"""Chat endpoint."""

from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core import cache, db
from app.services import rag

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # Force the reply language; omit or "auto" to mirror the question.
    language: Literal["auto", "en", "pl", "uk", "tr"] | None = None
    # Prior turns, oldest first, so follow-ups resolve against them.
    history: list[Turn] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    n: int
    title: str
    url: str
    # None when retrieval ran BM25-only (embedding quota spent): there is no
    # cosine score to report.
    similarity: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    # None in BM25-only mode — see Source.similarity.
    confidence: float | None = None
    answered: bool
    latency_ms: int
    # None when analytics logging failed — feedback is then simply unavailable.
    query_id: int | None = None
    # True when served from cache (no embedding or model call was spent).
    cached: bool = False


class FeedbackRequest(BaseModel):
    query_id: int
    helpful: bool


class SourceClickRequest(BaseModel):
    url: str = Field(max_length=2000)
    title: str = Field(default="", max_length=500)
    query_id: int | None = None


def _client_id(http: Request) -> str:
    """Caller identity for rate limiting. Behind Cloudflare/Caddy the socket peer
    is the proxy, so prefer the forwarded client address when present."""
    forwarded = http.headers.get("cf-connecting-ip") or http.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http.client.host if http.client else "unknown"


def _enforce_rate_limit(http: Request) -> None:
    if cache.rate_limited(_client_id(http)):
        raise HTTPException(
            status_code=429,
            detail="Too many questions in a short time. Please wait a moment.",
        )


def _graceful(question: str) -> dict:
    """A calm 'busy, try again' payload in the question's language. Used when
    something below the RAG layer raises — usually the database being briefly
    unreachable while the shared host is overloaded — so the visitor sees a
    message instead of a raw 500."""
    lang = rag._detect_language(question)
    return {
        "answer": rag._unavailable_answer(lang),
        "sources": [], "confidence": None, "answered": False,
        "latency_ms": 0, "query_id": None, "cached": False,
    }


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http: Request):
    _enforce_rate_limit(http)
    language = None if request.language in (None, "auto") else request.language
    try:
        return rag.answer(
            request.question,
            top_k=request.top_k,
            language=language,
            history=[t.model_dump() for t in request.history],
        )
    except Exception:
        logger.exception("answer failed unexpectedly")
        return _graceful(request.question)


@router.post("/ask/stream")
def ask_stream(request: AskRequest, http: Request):
    """Same answer, streamed as Server-Sent Events so the reply appears as it
    is written rather than after a multi-second wait."""
    _enforce_rate_limit(http)
    language = None if request.language in (None, "auto") else request.language

    def guarded():
        # An error below (e.g. the DB briefly unreachable under host load) must
        # not surface as a broken stream; close it cleanly with a busy message.
        try:
            yield from rag.stream(
                request.question,
                top_k=request.top_k,
                language=language,
                history=[t.model_dump() for t in request.history],
            )
        except Exception:
            logger.exception("stream failed unexpectedly")
            payload = _graceful(request.question)
            yield f"event: token\ndata: {json.dumps({'text': payload['answer']})}\n\n"
            yield ("event: done\ndata: " + json.dumps({
                "answered": False, "confidence": None,
                "latency_ms": 0, "query_id": None, "sources": [],
            }) + "\n\n")

    return StreamingResponse(
        guarded(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Proxies buffer by default, which would defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/suggestions")
def suggestions():
    """Starter questions shown in the empty chat window."""
    return {
        "pl": [
            "Ile kosztuje czesne na Informatyce?",
            "Jakie dokumenty są wymagane przy rekrutacji?",
            "Jakie kierunki studiów oferuje uczelnia?",
            "Gdzie znajduje się dziekanat?",
        ],
        "en": [
            "What is the tuition for Computer Engineering?",
            "What documents do I need to apply?",
            "Which study programmes are offered in English?",
            "How do I apply as an international student?",
        ],
    }


@router.post("/feedback")
def feedback(request: FeedbackRequest):
    try:
        with db.connection() as conn:
            updated = conn.execute(
                "UPDATE queries SET feedback = %s WHERE id = %s RETURNING id",
                (1 if request.helpful else -1, request.query_id),
            ).fetchone()
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not record feedback: {exc}")

    if not updated:
        raise HTTPException(status_code=404, detail="Unknown query id")
    return {"ok": True}


@router.post("/source-click", status_code=204)
def source_click(request: SourceClickRequest):
    """Record that a visitor opened a cited source (analytics only)."""
    db.log_source_click(url=request.url, title=request.title, query_id=request.query_id)
