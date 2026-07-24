"""Gemini embeddings (Google AI Studio free tier).

Chosen because the site is multilingual (Polish, English, Ukrainian) and this
model handles all three in one shared vector space, so a Polish page can answer
an English question without a translation step.

Documents and queries are embedded with different task types — asymmetric
retrieval measurably beats embedding both sides identically.
"""

from __future__ import annotations

import logging
import math
import time

import httpx

from app.core.config import google_api_key, settings

logger = logging.getLogger(__name__)

_API = "https://generativelanguage.googleapis.com/v1beta"
_BATCH = 100          # API limit per batchEmbedContents call
_MAX_RETRIES = 4


class EmbeddingError(RuntimeError):
    pass


def is_enabled() -> bool:
    return bool(google_api_key())


def _normalize(vec: list[float]) -> list[float]:
    """Re-normalise to unit length.

    gemini-embedding-001 only returns a normalised vector at its native 3072
    dims; a Matryoshka-truncated 768-dim vector must be renormalised before it
    is used with cosine distance.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _post(path: str, payload: dict, timeout: float = 60) -> dict:
    key = google_api_key()
    if not key:
        raise EmbeddingError("GOOGLE_API_KEY is not set")

    url = f"{_API}/{path}?key={key}"
    delay = 2.0
    for attempt in range(_MAX_RETRIES):
        try:
            r = httpx.post(url, json=payload, timeout=timeout)
            # Free tier is rate-limited; back off rather than losing the batch.
            if r.status_code in (429, 500, 503):
                if attempt == _MAX_RETRIES - 1:
                    raise EmbeddingError(f"embedding API {r.status_code}: {r.text[:200]}")
                logger.warning("embedding API %s — retrying in %.0fs", r.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"embedding API {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise EmbeddingError(f"embedding request failed: {exc}") from exc
            time.sleep(delay)
            delay *= 2
    raise EmbeddingError("embedding request failed after retries")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages for storage. Order is preserved."""
    out: list[list[float]] = []
    model = settings.embedding_model

    for start in range(0, len(texts), _BATCH):
        batch = texts[start:start + _BATCH]
        payload = {
            "requests": [
                {
                    "model": model,
                    "content": {"parts": [{"text": t}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "outputDimensionality": settings.embedding_dim,
                }
                for t in batch
            ]
        }
        data = _post(f"{model}:batchEmbedContents", payload)
        vectors = [_normalize(e["values"]) for e in data.get("embeddings", [])]
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"expected {len(batch)} embeddings, got {len(vectors)}"
            )
        out.extend(vectors)
        logger.info("embedded %d/%d", min(start + _BATCH, len(texts)), len(texts))

    return out


def embed_query(text: str) -> list[float]:
    """Embed a question. RETRIEVAL_QUERY is the matching side of the pair."""
    model = settings.embedding_model
    data = _post(
        f"{model}:embedContent",
        {
            "model": model,
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": settings.embedding_dim,
        },
    )
    values = (data.get("embedding") or {}).get("values")
    if not values:
        raise EmbeddingError("empty query embedding")
    return _normalize(values)
