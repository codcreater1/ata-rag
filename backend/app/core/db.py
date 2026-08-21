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
        _pool = ConnectionPool(
            url, min_size=1, max_size=8, kwargs={"row_factory": dict_row},
            # Neon (serverless) closes idle connections, so a pooled one can be
            # dead by the time it is handed out — which surfaced as intermittent
            # 500s under host load. check_connection validates each connection at
            # checkout and recycles a dead one, so the caller always gets a live
            # one instead of failing on first use.
            check=ConnectionPool.check_connection,
        )
    return _pool


@contextmanager
def connection():
    # check_connection above already retries at checkout: a dead connection is
    # validated, discarded and replaced with a live one (up to the pool timeout),
    # which is exactly the transient failure we were seeing under host load. So
    # the caller here just borrows a connection that is guaranteed live.
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

-- Lexical half of hybrid retrieval. 'simple' rather than a language-specific
-- configuration: the corpus mixes Polish, English and Ukrainian in one column,
-- and no single stemmer fits all three. Exact tokens (programme names, "1000
-- zł", "Erasmus") are what BM25 contributes here anyway.
CREATE INDEX IF NOT EXISTS chunks_fts_idx
    ON chunks USING GIN (to_tsvector('simple', text));

CREATE TABLE IF NOT EXISTS queries (
    id            BIGSERIAL PRIMARY KEY,
    asked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question      TEXT NOT NULL,
    language      TEXT NOT NULL DEFAULT '',
    answered      BOOLEAN NOT NULL DEFAULT TRUE,
    top_similarity REAL,
    latency_ms    INT,
    sources       JSONB NOT NULL DEFAULT '[]'::jsonb,
    feedback      SMALLINT,
    prompt_tokens     INT,
    completion_tokens INT
);

-- Which cited sources visitors actually open: the PDF spec asks for "top
-- clicked sources", which is a different signal from "most cited" — it shows
-- what people trust enough to go read.
CREATE TABLE IF NOT EXISTS source_clicks (
    id         BIGSERIAL PRIMARY KEY,
    clicked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    query_id   BIGINT REFERENCES queries(id) ON DELETE SET NULL,
    url        TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS source_clicks_url_idx ON source_clicks(url);

-- Persistent semantic answer cache. A question that was answered once should
-- never spend model credit again — not even after a redeploy (the in-process
-- cache resets then). Stored with the question's embedding so a *similar*
-- question ("how much is tuition" vs "what is the tuition") reuses the answer
-- too, matched by cosine distance in pgvector.
CREATE TABLE IF NOT EXISTS qa_cache (
    id            BIGSERIAL PRIMARY KEY,
    question_norm TEXT NOT NULL,
    language      TEXT NOT NULL DEFAULT 'auto',
    embedding     vector(%(dim)s),
    payload       JSONB NOT NULL,
    hits          INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_hit_at   TIMESTAMPTZ,
    UNIQUE (question_norm, language)
);
CREATE INDEX IF NOT EXISTS qa_cache_lang_idx ON qa_cache(language);
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


_MIGRATE_COLUMNS = """
ALTER TABLE queries ADD COLUMN IF NOT EXISTS prompt_tokens INT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS completion_tokens INT;
"""


def init_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA % {"dim": settings.embedding_dim})
        conn.execute(_MIGRATE_UNIQUE)
        conn.execute(_MIGRATE_COLUMNS)
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


def indexed_lastmods() -> dict[str, str]:
    """`{url: lastmod}` for pages that are fully indexed.

    The crawler compares these against the sitemap so an unchanged page is never
    fetched again. Only documents that actually have chunks count: one whose
    embeddings never landed must be retried, not skipped forever.
    """
    with connection() as conn:
        rows = conn.execute(
            """SELECT d.url, d.lastmod FROM documents d
                WHERE d.lastmod <> ''
                  AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"""
        ).fetchall()
    return {r["url"]: r["lastmod"] for r in rows}


def known_pdf_urls() -> set[str]:
    """PDF URLs discovered by earlier runs.

    PDF links are found by reading the pages that link to them. Once the crawler
    starts skipping unchanged pages it stops re-seeing those links, so the set
    has to persist across runs or the PDFs would drop out of the index.
    """
    with connection() as conn:
        rows = conn.execute(
            "SELECT url FROM documents WHERE source_type = 'pdf'"
        ).fetchall()
    return {r["url"] for r in rows}


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


# Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
# and the usual default: large enough that the top few ranks score similarly,
# so one retriever's confident hit is not automatically beaten by the other's.
_RRF_K = 60


def hybrid_search(query_embedding: list[float], query_text: str, *,
                  top_k: int, candidates: int = 30) -> list[dict]:
    """Vector + full-text search fused with Reciprocal Rank Fusion.

    Dense retrieval matches meaning but drifts on rare literal tokens — exact
    programme names, "Erasmus", a specific fee. Full-text nails those but misses
    paraphrase. RRF combines the two rankings without needing their scores to be
    on a comparable scale, which cosine distance and ts_rank are not.

    ``similarity`` stays the cosine score so the confidence gate keeps its
    meaning; a chunk found only by full-text carries its true (lower) cosine.
    """
    sql = """
        WITH vec AS (
            SELECT c.id,
                   row_number() OVER (ORDER BY c.embedding <=> %(q)s::vector) AS rank
              FROM chunks c
             WHERE c.embedding IS NOT NULL
             ORDER BY c.embedding <=> %(q)s::vector
             LIMIT %(cand)s
        ),
        fts AS (
            SELECT c.id,
                   row_number() OVER (
                       ORDER BY ts_rank(to_tsvector('simple', c.text),
                                        plainto_tsquery('simple', %(qt)s)) DESC
                   ) AS rank
              FROM chunks c
             WHERE to_tsvector('simple', c.text) @@ plainto_tsquery('simple', %(qt)s)
             LIMIT %(cand)s
        ),
        fused AS (
            SELECT COALESCE(vec.id, fts.id) AS id,
                   COALESCE(1.0 / (%(rrf)s + vec.rank), 0)
                 + COALESCE(1.0 / (%(rrf)s + fts.rank), 0) AS score
              FROM vec FULL OUTER JOIN fts ON vec.id = fts.id
        )
        SELECT c.text,
               c.metadata,
               d.url,
               d.title,
               d.language,
               1 - (c.embedding <=> %(q)s::vector) AS similarity,
               fused.score AS fusion_score
          FROM fused
          JOIN chunks c ON c.id = fused.id
          JOIN documents d ON d.id = c.document_id
         ORDER BY fused.score DESC
         LIMIT %(k)s
    """
    params = {
        "q": str(query_embedding), "qt": query_text,
        "k": top_k, "cand": candidates, "rrf": _RRF_K,
    }
    with connection() as conn:
        return conn.execute(sql, params).fetchall()


def top_by_type(query_embedding: list[float], source_types: tuple[str, ...],
                *, top_k: int) -> list[dict]:
    """Nearest chunks restricted to certain source types, by cosine.

    Fee cards are outnumbered by the programme and marketing pages that share a
    programme's name, so for a tuition question they fall out of the general
    candidate pool entirely. Fetching them on their own guarantees the figure is
    available to merge into the context.
    """
    # Filter to the type first (a MATERIALIZED CTE), then rank by distance. The
    # ivfflat index only scans a few lists, so a selective source_type filter on
    # top of an index scan can miss all matches and return nothing; materialising
    # the (small) typed set forces an exact nearest-neighbour over just those.
    sql = """
        WITH candidates AS MATERIALIZED (
            SELECT c.text, c.metadata, c.embedding, d.url, d.title, d.language
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE c.embedding IS NOT NULL AND d.source_type = ANY(%(types)s)
        )
        SELECT text, metadata, url, title, language,
               1 - (embedding <=> %(q)s::vector) AS similarity,
               NULL::float AS fusion_score
          FROM candidates
         ORDER BY embedding <=> %(q)s::vector
         LIMIT %(k)s
    """
    with connection() as conn:
        return conn.execute(
            sql, {"q": str(query_embedding), "types": list(source_types), "k": top_k},
        ).fetchall()


def _or_tsquery(query_text: str) -> str:
    """Build an OR tsquery from a question's content words.

    ``plainto_tsquery`` ANDs every token, and with the 'simple' config (no
    stopword removal) that means "what", "is", "the" all become required — so a
    natural-language question matches almost nothing. ORing the content words
    instead lets ts_rank reward chunks that cover more of them. \\w+ keeps the
    lexemes safe to hand to to_tsquery directly.
    """
    import re
    terms = [t for t in re.findall(r"\w+", query_text.lower()) if len(t) > 2]
    # Cap the term count so a pathological question can't build a huge query.
    return " | ".join(dict.fromkeys(terms[:24]))


def text_search(query_text: str, *, top_k: int) -> list[dict]:
    """BM25-only retrieval — no query embedding required.

    The serving path needs one embedding per question, from the same hard daily
    cap that indexing draws on; when that cap is spent, dense retrieval is simply
    unavailable. Full-text search still works and, for the literal questions this
    site gets (programme names, "Erasmus", a fee), is often enough on its own.

    ``similarity`` is reported as None because there is no cosine score to give.
    The LLM's own "answer only from the context, otherwise say you don't know"
    instruction remains the guard against answering from irrelevant chunks.
    """
    ts = _or_tsquery(query_text)
    if not ts:
        return []
    # Weight the title above the body (setweight A vs B). Without it a chatty
    # programme page that repeats "computer engineering" outranks the tuition
    # card actually titled "Tuition — Computer Engineering", so the fee numbers
    # never reach the context. With it the cards surface for a tuition question.
    doc = ("setweight(to_tsvector('simple', d.title), 'A') || "
           "setweight(to_tsvector('simple', c.text), 'B')")
    sql = f"""
        SELECT c.text, c.metadata, d.url, d.title, d.language,
               NULL::float AS similarity,
               ts_rank({doc}, to_tsquery('simple', %(ts)s)) AS fusion_score
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         WHERE {doc} @@ to_tsquery('simple', %(ts)s)
         ORDER BY fusion_score DESC
         LIMIT %(k)s
    """
    with connection() as conn:
        return conn.execute(sql, {"ts": ts, "k": top_k}).fetchall()


def qa_cache_get(question_norm: str, cache_lang: str,
                 embedding: list[float] | None,
                 threshold: float, max_age_days: int) -> dict | None:
    """A stored answer for this question, or None. Never raises.

    Two layers: an exact match on the normalised text (free, no embedding), then
    — if an embedding is given — the nearest past question by cosine distance,
    accepted only above *threshold*. Both are scoped to *cache_lang* so a reply
    is never reused across languages, and to entries younger than *max_age_days*
    so an answer is re-generated at least that often even if the site changed in
    a way the nightly re-index did not catch. A hit bumps the usage counter.
    """
    try:
        with connection() as conn:
            row = conn.execute(
                """SELECT id, payload FROM qa_cache
                    WHERE question_norm = %s AND language = %s
                      AND created_at > now() - make_interval(days => %s)""",
                (question_norm, cache_lang, max_age_days),
            ).fetchone()

            if not row and embedding is not None:
                row = conn.execute(
                    """SELECT id, payload, 1 - (embedding <=> %(q)s::vector) AS sim
                         FROM qa_cache
                        WHERE language = %(lang)s AND embedding IS NOT NULL
                          AND created_at > now() - make_interval(days => %(days)s)
                        ORDER BY embedding <=> %(q)s::vector
                        LIMIT 1""",
                    {"q": str(embedding), "lang": cache_lang, "days": max_age_days},
                ).fetchone()
                if not row or row["sim"] < threshold:
                    return None

            if not row:
                return None

            conn.execute(
                "UPDATE qa_cache SET hits = hits + 1, last_hit_at = now() WHERE id = %s",
                (row["id"],),
            )
            conn.commit()
            return row["payload"]
    except Exception:
        # A cache miss must never break answering; fall through to a fresh answer.
        logger.exception("qa_cache lookup failed")
        return None


def qa_cache_put(question_norm: str, cache_lang: str,
                 embedding: list[float] | None, payload: dict) -> None:
    """Store (or refresh) a cached answer. Never raises."""
    try:
        with connection() as conn:
            conn.execute(
                """INSERT INTO qa_cache (question_norm, language, embedding, payload)
                   VALUES (%(qn)s, %(lang)s, %(emb)s, %(payload)s)
                   ON CONFLICT (question_norm, language) DO UPDATE
                       SET payload = EXCLUDED.payload,
                           embedding = EXCLUDED.embedding,
                           created_at = now()""",
                {"qn": question_norm, "lang": cache_lang,
                 "emb": str(embedding) if embedding is not None else None,
                 "payload": json.dumps(payload, ensure_ascii=False)},
            )
            conn.commit()
    except Exception:
        logger.exception("qa_cache store failed")


def clear_qa_cache() -> None:
    """Drop the semantic cache — called after a re-index so new content shows."""
    try:
        with connection() as conn:
            conn.execute("TRUNCATE qa_cache")
            conn.commit()
    except Exception:
        logger.exception("qa_cache clear failed")


def log_query(*, question, language, answered, top_similarity, latency_ms, sources,
              prompt_tokens=None, completion_tokens=None) -> int | None:
    try:
        with connection() as conn:
            row = conn.execute(
                """INSERT INTO queries (question, language, answered, top_similarity,
                                        latency_ms, sources, prompt_tokens,
                                        completion_tokens)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (question, language, answered, top_similarity, latency_ms,
                 json.dumps(sources, ensure_ascii=False), prompt_tokens,
                 completion_tokens),
            ).fetchone()
            conn.commit()
            return row["id"]
    except psycopg.Error:
        # Analytics must never break answering.
        logger.exception("could not log query")
        return None


def log_source_click(*, url: str, title: str = "", query_id: int | None = None) -> None:
    """Record that a visitor opened a cited source. Never raises — analytics
    must not break the page."""
    try:
        with connection() as conn:
            conn.execute(
                "INSERT INTO source_clicks (query_id, url, title) VALUES (%s,%s,%s)",
                (query_id, url, title),
            )
            conn.commit()
    except psycopg.Error:
        logger.exception("could not log source click")
