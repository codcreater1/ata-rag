"""Runtime configuration, sourced from the environment.

Override anything with the ``ATARAG_`` prefix, or a local ``.env`` file.
Secrets (API keys, database URL) are read without the prefix so they match the
names the providers themselves document.
"""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATARAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_title: str = "ATA RAG"
    app_version: str = "0.1.0"

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ---------------------------------------------------------------- #
    # Retrieval
    # ---------------------------------------------------------------- #
    top_k: int = 10          # the PDF spec's "top 10 chunks"
    # Below this cosine similarity the context is treated as "not found" and the
    # bot says so instead of guessing from weak matches (spec: < 0.65).
    min_similarity: float = 0.65

    # gemini-embedding-001 defaults to 3072 dims but supports Matryoshka
    # truncation; 768 keeps the pgvector index small with negligible quality
    # loss and stays under pgvector's 2000-dim ivfflat limit.
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 768

    # ---------------------------------------------------------------- #
    # Answering LLM (OpenAI-compatible endpoint; Groq by default)
    # ---------------------------------------------------------------- #
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    answer_language: str = "auto"   # auto = mirror the user's question

    # ---------------------------------------------------------------- #
    # Crawl
    # ---------------------------------------------------------------- #
    crawl_delay_seconds: float = 0.4


settings = Settings()


# Secrets — plain names, so they match provider docs and Coolify variables.
def google_api_key() -> str:
    """Gemini key (Google AI Studio) used for embeddings."""
    return os.getenv("GOOGLE_API_KEY", "")


def llm_api_key() -> str:
    """Key for the answering LLM (Groq by default)."""
    return os.getenv("LLM_API_KEY", "")


def database_url() -> str:
    """Postgres connection string (Neon), must have pgvector available."""
    return os.getenv("DATABASE_URL", "")
