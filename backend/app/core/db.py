"""Postgres + pgvector access.

Deliberately plain ``psycopg`` rather than an ORM: the schema is two tables and
the only interesting query is a vector search, which is easier to read and tune
as SQL.

Schema
    documents  one row per crawled page / generated tuition card
    chunks     retrievable passages, with the embedding and a jsonb metadata bag
    queries    every question asked, for the dashboard (latency, scores, misses)
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import database_url, settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        url = database_url()
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = ConnectionPool(url, min_size=1, max_size=8, kwargs={"row_factory": dict_row})
    return _pool


@contextmanager
def connection():
    with get_pool().connection() as conn:
        yield conn


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    url         TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    language    TEXT NOT NULL DEFAULT 'pl',
    source_type TEXT NOT NULL DEFAULT 'page',
    lastmod     TEXT NOT NULL DEFAULT '',
    -- Hash of the extracted markdown: lets a re-crawl skip unchanged pages
    -- instead of re-embedding the whole site every night.
    content_hash TEXT NOT NULL DEFAULT '',
    markdown    TEXT NOT NULL DEFAULT '',
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- source_type is part of the key: a tuition card deliberately cites the
    -- programme page it describes, and the crawler fetches that same page. They
    -- are different documents about one URL and must coexist — keying on
    -- (url, language) alone let the crawled page overwrite the fee card.
    UNIQUE (url, language, source_type)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(%(dim)s),
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING GIN (metadata);

CREATE TABLE IF NOT EXISTS queries (
    id            BIGSERIAL PRIMARY KEY,
    asked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question      TEXT NOT NULL,
    language      TEXT NOT NULL DEFAULT '',
    answered      BOOLEAN NOT NULL DEFAULT TRUE,
    top_similarity REAL,
    latency_ms    INT,
    sources       JSONB NOT NULL DEFAULT '[]'::jsonb,
    feedback      SMALLINT
);
"""

# Built separately: ivfflat needs rows present to pick sensible lists, so we
# create it after the first index run rather than on an empty table.
VECTOR_INDEX = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = %(lists)s);
"""


# Databases created before source_type joined the key still carry the old
# constraint, under which crawled pages silently replaced tuition cards.
_MIGRATE_UNIQUE = """
DO $$
DECLARE
    old_name text;
BEGIN
    SELECT c.conname INTO old_name
      FROM pg_constraint c
      JOIN pg_class t ON t.oid = c.conrelid
     WHERE t.relname = 'documents'
       AND c.contype = 'u'
       AND (SELECT count(*) FROM unnest(c.conkey)) = 2;

    IF old_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE documents DROP CONSTRAINT %I', old_name);
        ALTER TABLE documents
            ADD CONSTRAINT documents_url_language_source_type_key
            UNIQUE (url, language, source_type);
    END IF;
END $$;
"""


def init_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA % {"dim": settings.embedding_dim})
        conn.execute(_MIGRATE_UNIQUE)
        conn.commit()
    logger.info("schema ready")


def build_vector_index() -> None:
    """Create the ANN index once there is data to size it against."""
    with connection() as conn:
        n = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        if n < 1000:
            logger.info("only %d chunks — exact search is fine, skipping ivfflat", n)
            return
        lists = max(10, min(1000, n // 100))
        conn.execute(VECTOR_INDEX % {"lists": lists})
        conn.commit()
    logger.info("vector index built (lists=%d)", lists)


def upsert_document(conn, *, url, title, language, source_type, lastmod,
                    content_hash, markdown) -> tuple[int, bool]:
    """Insert or update a document. Returns (id, changed)."""
    # Match the uniqueness key exactly, so a crawled page never displaces the
    # tuition card that cites the same URL.
    row = conn.execute(
        """SELECT id, content_hash FROM documents
            WHERE url = %s AND language = %s AND source_type = %s""",
        (url, language, source_type),
    ).fetchone()

    if row and row["content_hash"] == content_hash:
        return row["id"], False           # unchanged: caller can skip re-embedding

    if row:
        conn.execute(
            """UPDATE documents
                  SET title=%s, lastmod=%s, content_hash=%s,
                      markdown=%s, fetched_at=now()
                WHERE id=%s""",
            (title, lastmod, content_hash, markdown, row["id"]),
        )
        return row["id"], True

    new = conn.execute(
        """INSERT INTO documents (url, title, language, source_type, lastmod,
                                  content_hash, markdown)
           VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (url, title, language, source_type, lastmod, content_hash, markdown),
    ).fetchone()
    return new["id"], True


def replace_chunks(conn, document_id: int, rows: list[tuple[int, str, list[float], dict]]) -> None:
    """Swap in a document's chunks (delete-then-insert keeps it simple)."""
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (document_id, chunk_index, text, embedding, metadata)
               VALUES (%s,%s,%s,%s,%s)""",
            [(document_id, i, text, embedding, json.dumps(meta, ensure_ascii=False))
             for i, text, embedding, meta in rows],
        )


def search(query_embedding: list[float], *, top_k: int, language: str | None = None) -> list[dict]:
    """Cosine-similarity search, optionally restricted to one language."""
    sql = """
        SELECT c.text,
               c.metadata,
               d.url,
               d.title,
               d.language,
               1 - (c.embedding <=> %(q)s::vector) AS similarity
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         WHERE c.embedding IS NOT NULL
    """
    params: dict = {"q": str(query_embedding), "k": top_k}
    if language:
        sql += " AND d.language = %(lang)s"
        params["lang"] = language
    sql += " ORDER BY c.embedding <=> %(q)s::vector LIMIT %(k)s"

    with connection() as conn:
        return conn.execute(sql, params).fetchall()


def log_query(*, question, language, answered, top_similarity, latency_ms, sources) -> int | None:
    try:
        with connection() as conn:
            row = conn.execute(
                """INSERT INTO queries (question, language, answered, top_similarity,
                                        latency_ms, sources)
                   VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
                (question, language, answered, top_similarity, latency_ms,
                 json.dumps(sources, ensure_ascii=False)),
            ).fetchone()
            conn.commit()
            return row["id"]
    except psycopg.Error:
        # Analytics must never break answering.
        logger.exception("could not log query")
        return None
