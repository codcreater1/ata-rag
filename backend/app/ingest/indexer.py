"""Full ingestion run: crawl + tuition data -> chunks -> embeddings -> Postgres.

Runs as a Coolify scheduled task (nightly). Pages whose extracted markdown is
byte-identical to the stored copy are skipped, so a routine run re-embeds only
what actually changed instead of the whole site.
"""

from __future__ import annotations

import hashlib
import logging

from app.core import db, embeddings
from app.core.config import settings
from app.ingest.chunker import chunk_page
from app.ingest.crawler import crawl
from app.ingest.prices import build_price_pages

logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(*, limit: int | None = None, skip_crawl: bool = False,
        force: bool = False) -> dict:
    """Index the site. Returns a summary for the dashboard / logs."""
    db.init_schema()

    pages = []
    if not skip_crawl:
        pages += crawl(limit=limit, delay=settings.crawl_delay_seconds)
    # Tuition cards are cheap and always worth refreshing — fees change.
    try:
        pages += build_price_pages()
    except Exception:
        logger.exception("tuition ingestion failed; continuing with crawled pages")

    stats = {"pages": len(pages), "indexed": 0, "skipped": 0,
             "chunks": 0, "failed": 0}

    for page in pages:
        content_hash = _hash(page.markdown)
        try:
            with db.connection() as conn:
                doc_id, changed = db.upsert_document(
                    conn, url=page.url, title=page.title, language=page.language,
                    source_type=page.source_type, lastmod=page.lastmod,
                    content_hash=content_hash, markdown=page.markdown,
                )
                if not changed and not force:
                    stats["skipped"] += 1
                    conn.commit()
                    continue

                chunks = chunk_page(page)
                if not chunks:
                    conn.commit()
                    continue

                vectors = embeddings.embed_documents([c.text for c in chunks])
                db.replace_chunks(
                    conn, doc_id,
                    [(c.order, c.text, v, c.metadata) for c, v in zip(chunks, vectors)],
                )
                conn.commit()

            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
        except Exception:
            logger.exception("failed to index %s", page.url)
            stats["failed"] += 1

    db.build_vector_index()
    logger.info("ingestion done: %s", stats)
    return stats


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Index akademiata.pl into pgvector")
    parser.add_argument("--limit", type=int, help="cap crawled pages (smoke test)")
    parser.add_argument("--skip-crawl", action="store_true",
                        help="refresh tuition data only")
    parser.add_argument("--force", action="store_true",
                        help="re-embed even unchanged pages")
    args = parser.parse_args()

    print(run(limit=args.limit, skip_crawl=args.skip_crawl, force=args.force))
