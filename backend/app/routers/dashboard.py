"""Dashboard analytics.

Surfaces what the professor asked to track: common questions, unanswered ones,
retrieval quality, latency and feedback — plus what is actually in the index.
Unanswered questions are the most useful signal: they show where the knowledge
base (or the website itself) has a gap.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core import db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with db.connection() as conn:
        return conn.execute(sql, params).fetchall()


@router.get("/stats")
def stats():
    """Headline numbers for the dashboard cards."""
    try:
        index = _rows("""
            SELECT (SELECT count(*) FROM documents) AS documents,
                   (SELECT count(*) FROM chunks)    AS chunks,
                   (SELECT max(fetched_at) FROM documents) AS last_indexed
        """)[0]

        usage = _rows("""
            SELECT count(*)                                     AS questions,
                   count(*) FILTER (WHERE NOT answered)          AS unanswered,
                   round(avg(latency_ms))                        AS avg_latency_ms,
                   round(avg(top_similarity)::numeric, 3)        AS avg_similarity,
                   count(*) FILTER (WHERE feedback = 1)          AS helpful,
                   count(*) FILTER (WHERE feedback = -1)         AS not_helpful
              FROM queries
        """)[0]

        by_language = _rows("""
            SELECT language, count(*) AS n FROM documents
             GROUP BY language ORDER BY n DESC
        """)
        by_type = _rows("""
            SELECT source_type, count(*) AS n FROM documents
             GROUP BY source_type ORDER BY n DESC
        """)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    return {"index": index, "usage": usage,
            "documents_by_language": by_language, "documents_by_type": by_type}


@router.get("/questions")
def questions(limit: int = 50, only_unanswered: bool = False):
    """Recent questions, newest first."""
    sql = """
        SELECT id, asked_at, question, language, answered,
               top_similarity, latency_ms, feedback
          FROM queries
    """
    if only_unanswered:
        sql += " WHERE NOT answered"
    sql += " ORDER BY asked_at DESC LIMIT %s"
    try:
        return {"questions": _rows(sql, (min(limit, 200),))}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/top-questions")
def top_questions(limit: int = 20):
    """Most frequently asked questions (normalised, case-insensitive)."""
    try:
        return {"questions": _rows("""
            SELECT lower(btrim(question)) AS question,
                   count(*)               AS times_asked,
                   bool_or(answered)      AS ever_answered,
                   round(avg(top_similarity)::numeric, 3) AS avg_similarity
              FROM queries
             GROUP BY lower(btrim(question))
             ORDER BY times_asked DESC, question
             LIMIT %s
        """, (min(limit, 100),))}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/gaps")
def gaps(limit: int = 20):
    """Unanswered questions — where the knowledge base falls short."""
    try:
        return {"gaps": _rows("""
            SELECT lower(btrim(question)) AS question,
                   count(*)               AS times_asked,
                   round(avg(top_similarity)::numeric, 3) AS avg_similarity,
                   max(asked_at)          AS last_asked
              FROM queries
             WHERE NOT answered
             GROUP BY lower(btrim(question))
             ORDER BY times_asked DESC, last_asked DESC
             LIMIT %s
        """, (min(limit, 100),))}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/sources")
def top_sources(limit: int = 20):
    """Pages cited most often — which content actually answers questions."""
    try:
        return {"sources": _rows("""
            SELECT s->>'url'   AS url,
                   s->>'title' AS title,
                   count(*)    AS citations
              FROM queries q, jsonb_array_elements(q.sources) s
             GROUP BY 1, 2
             ORDER BY citations DESC
             LIMIT %s
        """, (min(limit, 100),))}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
