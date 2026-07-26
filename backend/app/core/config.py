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

    # The deployed frontend's origin comes from CORS_ORIGINS in the environment.
    # These are the dev-server defaults: 5174 is the port vite.config.js pins,
    # 5173 is vite's own default and what a stray `vite` run lands on.
    cors_origins: list[str] = [
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ---------------------------------------------------------------- #
    # Retrieval
    # ---------------------------------------------------------------- #
    top_k: int = 10          # the PDF spec's "top 10 chunks"
    # Below this cosine similarity the context is treated as "not found" and the
    # bot says so instead of guessing from weak matches.
    #
    # The specification suggests 0.65, but that figure is calibrated for OpenAI
    # text-embedding-3-small; gemini-embedding-001 places its scores lower.
    # Measured against the live index:
    #     answerable questions   0.613 – 0.781  (min: "Erasmus exchange programme")
    #     irrelevant questions   0.521 – 0.579  (max: "how much is the rector's dog")
    # 0.65 sat above the weakest genuine question and rejected it. 0.60 sits in
    # the gap: every real question passes, every nonsense one is still refused.
    # Re-measure this if the embedding model changes.
    min_similarity: float = 0.60

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

    # Groq's free tier allows 100k tokens per day, which a single afternoon of
    # demoing can exhaust — and a dead LLM makes every answer look like a
    # knowledge gap. Gemini stands in via its OpenAI-compatible endpoint.
    fallback_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # Several models, not one: each Gemini model has its OWN free-tier daily
    # request bucket (the newest, e.g. 3.6-flash, is as low as 20/day), so
    # trying them in turn multiplies the fallback's capacity. Ordered lite-first
    # (higher free throughput). Comma-separated so it is env-overridable.
    fallback_models: str = ("gemini-flash-lite-latest,gemini-2.0-flash-lite,"
                            "gemini-flash-latest,gemini-2.0-flash")

    # Indexing and query-time retrieval draw on the same ~1000/day Gemini
    # embedding cap (one request per chunk). A greedy index run drains it and
    # every live question then falls back to BM25-only. Stopping a run short of
    # the cap leaves headroom so the daytime product keeps its dense retrieval.
    # 0 disables the reserve (a deliberate full re-index).
    embed_chunks_per_run: int = 850

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
