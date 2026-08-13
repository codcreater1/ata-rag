"""Ingest the PDFs the site links to (regulations, fee tables, forms).

Much of what a student actually needs — the internship regulations, the fee
schedule, scholarship rules — is published as a PDF rather than a web page, so a
crawler that only reads HTML misses it entirely.

Text is extracted with PyMuPDF. Scanned, image-only PDFs yield nothing
extractable; those are reported and skipped rather than indexed as empty
documents (OCR would be the next step, and is deliberately out of scope here).
"""

from __future__ import annotations

import io
import logging
import re
import time
from urllib.parse import unquote, urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

BASE = "https://akademiata.pl"
USER_AGENT = "ATA-RAG-Bot/1.0 (+akademiata.pl knowledge base)"

# Large scans are rarely worth the download; the useful documents are text PDFs.
MAX_PDF_BYTES = 20 * 1024 * 1024
# Below this, extraction produced nothing usable (image-only scan).
MIN_TEXT_CHARS = 400


def discover_pdf_links(html: str, page_url: str) -> set[str]:
    """PDF URLs linked from one page, absolutised and de-duplicated."""
    found: set[str] = set()
    for href in re.findall(r'href="([^"]+)"', html, re.IGNORECASE):
        url = urljoin(page_url, href.strip())
        path = urlparse(url).path.lower()
        if path.endswith(".pdf") and urlparse(url).netloc.endswith("akademiata.pl"):
            found.add(url.split("#")[0])
    return found


def _title_from(url: str, first_page: str) -> str:
    """Prefer the document's own first heading; fall back to the filename."""
    for line in first_page.splitlines():
        line = line.strip()
        # A title line: substantial, not a lone number, not a date stamp.
        if 12 <= len(line) <= 120 and not re.fullmatch(r"[\d\s.,/-]+", line):
            return line
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    return re.sub(r"[-_]+", " ", name[:-4]).strip() or url


def _to_markdown(doc) -> tuple[str, str]:
    """(first_page_text, markdown) for a PyMuPDF document."""
    pages: list[str] = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            # Collapse the ragged line breaks PDF extraction produces, keeping
            # paragraph boundaries so heading-aware chunking still has something
            # to split on.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            pages.append(text)

    if not pages:
        return "", ""

    body = "\n\n".join(pages)
    return pages[0], body


def fetch_pdf_page(url: str, client: httpx.Client | None = None):
    """Download and extract one PDF into a ``crawler.Page``, or None."""
    import fitz  # PyMuPDF

    from app.ingest.crawler import Page, _lang_of

    owns_client = client is None
    client = client or httpx.Client()
    try:
        r = client.get(url, headers={"User-Agent": USER_AGENT},
                       timeout=60, follow_redirects=True)
        r.raise_for_status()
        data = r.content
    except httpx.HTTPError as exc:
        logger.warning("PDF fetch failed %s: %s", url, exc)
        return None
    finally:
        if owns_client:
            client.close()

    if len(data) > MAX_PDF_BYTES:
        logger.info("skipping oversized PDF (%.1f MB): %s", len(data) / 1e6, url)
        return None

    try:
        with fitz.open(stream=io.BytesIO(data), filetype="pdf") as doc:
            if doc.needs_pass or doc.is_encrypted:
                logger.info("skipping encrypted PDF: %s", url)
                return None
            first_page, markdown = _to_markdown(doc)
    except Exception as exc:
        logger.warning("PDF parse failed %s: %s", url, exc)
        return None

    if len(markdown) < MIN_TEXT_CHARS:
        # Almost certainly a scan. Indexing it would add an empty document that
        # can never answer anything.
        logger.info("skipping image-only/near-empty PDF: %s", url)
        return None

    title = _title_from(url, first_page)
    return Page(
        url=url,
        title=title,
        markdown=f"# {title}\n\n{markdown}",
        language=_lang_of(url),
        source_type="pdf",
        metadata={"kind": "document", "format": "pdf"},
    )


def crawl_pdfs(pdf_urls: set[str], *, delay: float = 0.4, limit: int | None = None) -> list:
    """Fetch and extract a set of PDF URLs."""
    pages = []
    targets = sorted(pdf_urls)[:limit] if limit else sorted(pdf_urls)
    with httpx.Client() as client:
        for i, url in enumerate(targets, 1):
            page = fetch_pdf_page(url, client)
            if page:
                pages.append(page)
            if i % 10 == 0:
                logger.info("PDFs: %d/%d processed, %d usable", i, len(targets), len(pages))
            time.sleep(delay)

    logger.info("extracted %d usable PDFs of %d linked", len(pages), len(targets))
    return pages
