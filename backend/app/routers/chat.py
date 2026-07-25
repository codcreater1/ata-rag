"""Chat endpoint."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core import db
from app.services import rag

router = APIRouter(prefix="/chat", tags=["chat"])


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # Force the reply language; omit or "auto" to mirror the question.
    language: Literal["auto", "en", "pl", "uk", "tr"] | None = None


class Source(BaseModel):
    n: int
    title: str
    url: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    answered: bool
    latency_ms: int
    # None when analytics logging failed — feedback is then simply unavailable.
    query_id: int | None = None


class FeedbackRequest(BaseModel):
    query_id: int
    helpful: bool


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    language = None if request.language in (None, "auto") else request.language
    return rag.answer(request.question, top_k=request.top_k, language=language)


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
