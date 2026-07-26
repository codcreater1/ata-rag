"""ATA RAG API — chat over the university website, plus dashboard analytics."""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.core.config import settings  # noqa: E402
from app.routers import chat, dashboard  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title=settings.app_title, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _tracing_status() -> str:
    """Whether answers are actually being traced.

    Tracing needs both the keys and an importable SDK; either missing means
    traces silently stop while answers keep working, which is exactly the
    failure that is hard to notice from the outside.
    """
    import os

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return "disabled: no LANGFUSE_* keys"
    try:
        import langfuse  # noqa: F401
        from langfuse.openai import OpenAI  # noqa: F401
    except Exception as exc:
        return f"broken: {type(exc).__name__}: {exc}"
    return "active"


@app.get("/health")
def health():
    from app.core import embeddings
    from app.core.config import database_url, llm_api_key

    return {
        "status": "healthy",
        "service": settings.app_title,
        "version": settings.app_version,
        "embeddings_enabled": embeddings.is_enabled(),
        "llm_enabled": bool(llm_api_key()),
        "database_configured": bool(database_url()),
        "tracing": _tracing_status(),
    }


app.include_router(chat.router)
app.include_router(dashboard.router)
