"""Sitemap-driven crawler for akademiata.pl.

The site is WordPress with a sitemap index, so we enumerate URLs from the
sitemaps instead of following links — faster, complete, and it gives us
``lastmod`` for free (used later to re-index only what changed).

Only content-bearing sitemaps are crawled. The site also publishes ~1.4k
taxonomy/filter URLs (tags, categories, price/duration facets) that carry no
prose and would only add noise to retrieval, so they are skipped by default.

Page HTML is reduced to the main article with trafilatura: every page on this
site starts with a large shared navigation menu, and naive tag-stripping would
make every chunk look alike and wreck retrieval.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

BASE = "https://akademiata.pl"
SITEMAP_INDEX = f"{BASE}/sitemap_index.xml"

# Sitemaps holding real prose, most useful first. Everything else on the site is
# taxonomy/faceting. Order matters: a capped run (limit=N) should get admissions
# and programme pages before the ~700 news posts.
CONTENT_SITEMAPS = [
    "page", "bachelor", "master", "postgraduate", "mba",
    "courses", "faq", "exams", "contact", "post",
]

# Never index these path segments even if a sitemap lists them.
EXCLUDE_PATTERNS = re.compile(
    r"/(wp-admin|wp-json|feed|search|tag|category|author|page/\d+)(/|$)", re.I
)

USER_AGENT = "ATA-RAG-Bot/1.0 (+akademiata.pl knowledge base)"
_HEADERS = {"User-Agent": USER_AGENT}


@dataclass
class Page:
    url: str
    title: str
    markdown: str
    language: str
    lastmod: str = ""
    source_type: str = "page"          # which sitemap it came from
    metadata: dict = field(default_factory=dict)


def _lang_of(url: str) -> str:
    """The site serves /en/... and /uk/... prefixes; default is Polish."""
    path = urlparse(url).path
    if path.startswith("/en/"):
        return "en"
    if path.startswith("/uk/"):
        return "uk"
    if path.startswith("/ru/"):
        return "ru"
    return "pl"


def _get(client: httpx.Client, url: str, *, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        try:
            r = client.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
            r.raise_for_status()
            return r.text
        except httpx.HTTPError as exc:
            if attempt == retries:
                logger.warning("fetch failed %s: %s", url, exc)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _sitemap_urls(client: httpx.Client) -> list[tuple[str, str, str]]:
    """Return (url, lastmod, source_type) for every content page."""
    index = _get(client, SITEMAP_INDEX)
    if not index:
        raise RuntimeError(f"Could not read sitemap index at {SITEMAP_INDEX}")

    available = {
        sm.rstrip("/").split("/")[-1].replace("-sitemap.xml", ""): sm
        for sm in re.findall(r"<loc>([^<]+)</loc>", index)
    }

    out: list[tuple[str, str, str]] = []
    for name in CONTENT_SITEMAPS:          # priority order, not sitemap order
        sm = available.get(name)
        if not sm:
            continue

        body = _get(client, sm)
        if not body:
            continue

        # <url><loc>..</loc><lastmod>..</lastmod></url>
        for block in re.findall(r"<url>(.*?)</url>", body, re.S):
            loc = re.search(r"<loc>([^<]+)</loc>", block)
            if not loc:
                continue
            url = loc.group(1).strip()
            if EXCLUDE_PATTERNS.search(urlparse(url).path):
                continue
            mod = re.search(r"<lastmod>([^<]+)</lastmod>", block)
            out.append((url, mod.group(1).strip() if mod else "", name))

    return out


def _extract(html: str, url: str) -> tuple[str, str]:
    """(title, markdown) for the page's main content, or ('', '')."""
    import trafilatura

    md = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_recall=True,
        url=url,
    )
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "").strip()
    if not title:
        m = re.search(r"<title>([^<]*)</title>", html, re.I)
        title = m.group(1).strip() if m else url

    # Strip the site-wide suffix WordPress appends to every <title>.
    title = re.sub(r"\s*[-|–]\s*(ATA|Akademia Techniczno-Artystyczna).*$", "", title).strip()
    return title, (md or "").strip()


def crawl(limit: int | None = None, delay: float = 0.4,
          collect_pdf_links: set[str] | None = None) -> list[Page]:
    """Crawl content pages. *limit* caps pages (useful for smoke tests).

    Pass a set as *collect_pdf_links* to also gather every linked PDF URL seen
    along the way; the caller can then ingest those separately (regulations and
    fee tables are published as PDFs, not pages).
    """
    from app.ingest.pdfs import discover_pdf_links

    pages: list[Page] = []
    with httpx.Client() as client:
        targets = _sitemap_urls(client)
        logger.info("sitemap listed %d content URLs", len(targets))
        if limit:
            targets = targets[:limit]

        for i, (url, lastmod, source_type) in enumerate(targets, 1):
            html = _get(client, url)
            if not html:
                continue

            if collect_pdf_links is not None:
                collect_pdf_links |= discover_pdf_links(html, url)

            title, md = _extract(html, url)
            # Very short extractions are nav-only stubs, not content.
            if len(md) < 200:
                logger.debug("skipping thin page (%d chars): %s", len(md), url)
                continue

            pages.append(Page(
                url=url, title=title, markdown=md,
                language=_lang_of(url), lastmod=lastmod, source_type=source_type,
            ))
            if i % 25 == 0:
                logger.info("crawled %d/%d", i, len(targets))
            time.sleep(delay)

    return pages
