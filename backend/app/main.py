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
    }


app.include_router(chat.router)
app.include_router(dashboard.router)
