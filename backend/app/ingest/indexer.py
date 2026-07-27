"""Full ingestion run: crawl + tuition data -> chunks -> embeddings -> Postgres.

Runs as a Coolify scheduled task (nightly), and skips unchanged work twice over:
the sitemap's ``lastmod`` keeps an unmodified page from being fetched at all,
and a content hash keeps a page whose markdown came back identical from being
re-embedded. A routine run therefore costs minutes and almost no quota, instead
of half an hour and the whole site's worth of embeddings.
"""

from __future__ import annotations

import hashlib
import logging

from app.core import cache, db, embeddings
from app.core.config import settings
from app.ingest.chunker import chunk_page
from app.ingest.crawler import crawl
from app.ingest.pdfs import crawl_pdfs
from app.ingest.prices import build_price_pages

logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(*, limit: int | None = None, skip_crawl: bool = False,
        force: bool = False, pdf_limit: int | None = None,
        full: bool = False) -> dict:
    """Index the site. Returns a summary for the dashboard / logs.

    By default the crawl trusts the sitemap's ``lastmod`` and refetches only
    pages that changed. Pass *full* to refetch everything — worth doing after a
    change to extraction or chunking, where the source page is identical but our
    reading of it is not.
    """
    db.init_schema()

    pages = []
    if not skip_crawl:
        # Collect PDF links while crawling, then ingest them: regulations, fee
        # tables and forms are published as PDFs and are invisible to an
        # HTML-only crawl. Seed with the ones already known, since the pages
        # that link to them are usually skipped as unmodified.
        pdf_links: set[str] = set() if full else db.known_pdf_urls()
        pages += crawl(limit=limit, delay=settings.crawl_delay_seconds,
                       collect_pdf_links=pdf_links,
                       known_lastmod=None if full else db.indexed_lastmods())
        if pdf_links:
            try:
                pages += crawl_pdfs(pdf_links, delay=settings.crawl_delay_seconds,
                                    limit=pdf_limit)
            except Exception:
                logger.exception("PDF ingestion failed; continuing with pages")
    # Tuition cards are cheap and always worth refreshing — fees change.
    try:
        pages += build_price_pages()
    except Exception:
        logger.exception("tuition ingestion failed; continuing with crawled pages")

    stats = {"pages": len(pages), "indexed": 0, "skipped": 0,
             "chunks": 0, "failed": 0, "quota_reached": False,
             "budget_reached": False}

    # Embed the highest-value pages first so a day capped by the free-tier quota
    # still covers tuition, programmes and admissions before news posts.
    priority = {"tuition": 0, "page": 1, "pdf": 2, "bachelor": 2, "master": 2,
                "postgraduate": 3, "mba": 3, "courses": 3, "faq": 1,
                "contact": 4, "exams": 4, "post": 9}
    pages.sort(key=lambda p: priority.get(p.source_type, 5))

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

                # Stop before draining the day's embedding budget, so live
                # queries keep their dense retrieval. Checked before the call so
                # a page is never left with half its chunks embedded. A full
                # re-index is a deliberate admin action and lifts the reserve.
                budget = 0 if full else settings.embed_chunks_per_run
                if budget and stats["chunks"] + len(chunks) > budget:
                    conn.rollback()
                    stats["budget_reached"] = True
                    logger.info("per-run embed budget (%d chunks) reached — "
                                "stopping to leave quota for live queries", budget)
                    break

                vectors = embeddings.embed_documents([c.text for c in chunks])
                db.replace_chunks(
                    conn, doc_id,
                    [(c.order, c.text, v, c.metadata) for c, v in zip(chunks, vectors)],
                )
                conn.commit()

            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
        except embeddings.DailyQuotaExceeded:
            # Stop cleanly: today's budget is gone. Already-committed documents
            # persist, and tomorrow's run resumes from here (unchanged pages are
            # skipped by content hash). Never a crash, never a partial commit.
            stats["quota_reached"] = True
            logger.warning("daily embedding quota reached — stopping after %d docs",
                           stats["indexed"])
            break
        except Exception:
            logger.exception("failed to index %s", page.url)
            stats["failed"] += 1

    db.build_vector_index()
    if stats["indexed"]:
        # Cached answers were built from the previous corpus — drop both layers.
        cache.clear_answers()
        db.clear_qa_cache()
    logger.info("ingestion done: %s", stats)
    return stats


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    # Coolify injects the environment, but a local run has only .env — and the
    # API server is the only other thing that loads it.
    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Index akademiata.pl into pgvector")
    parser.add_argument("--limit", type=int, help="cap crawled pages (smoke test)")
    parser.add_argument("--skip-crawl", action="store_true",
                        help="refresh tuition data only")
    parser.add_argument("--force", action="store_true",
                        help="re-embed even unchanged pages")
    parser.add_argument("--pdf-limit", type=int, help="cap linked PDFs ingested")
    parser.add_argument("--full", action="store_true",
                        help="refetch every page instead of trusting sitemap lastmod")
    args = parser.parse_args()

    print(run(limit=args.limit, skip_crawl=args.skip_crawl, force=args.force,
              pdf_limit=args.pdf_limit, full=args.full))
