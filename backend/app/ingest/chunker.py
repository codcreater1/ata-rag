"""Heading-aware chunking.

Fixed-size splitting cuts sentences and tables in half and loses the heading a
passage belongs to, which matters here: most answers live under a specific
section ("Wymagane dokumenty", "Opłaty"). We split on Markdown headings first,
then only sub-split sections that are too long, and we prefix every chunk with
its document title + heading path so an isolated chunk still says what it is
about — retrieval and the grounding prompt both rely on that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Roughly 1500 characters ~ 375 tokens: large enough to hold a full section,
# small enough that a top-k of 8 still fits comfortably in the prompt.
MAX_CHARS = 1500
MIN_CHARS = 80          # below this a chunk is a stub (stray heading, nav crumb)
OVERLAP_CHARS = 150     # carried between sub-splits so sentences aren't orphaned

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    heading_path: str
    order: int
    metadata: dict = field(default_factory=dict)


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split into (heading_path, body) pairs following the heading hierarchy."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(stack), body))
        buf.clear()

    for line in markdown.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            # Keep only ancestors shallower than this heading.
            stack[:] = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
        else:
            buf.append(line)
    flush()

    return sections or [("", markdown.strip())]


def _atoms(body: str) -> list[str]:
    """Break a section into pieces that each fit the budget.

    Paragraphs first, then sentences, then a hard wrap. Splitting to
    within-budget atoms up front means the packing loop below never has to
    handle an oversized piece — the case that previously let 18k-character
    chunks through.
    """
    atoms: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHARS:
            atoms.append(para)
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= MAX_CHARS:
                atoms.append(sentence)
            else:
                for i in range(0, len(sentence), MAX_CHARS):
                    atoms.append(sentence[i:i + MAX_CHARS])
    return atoms


def _pack(body: str) -> list[str]:
    """Greedily pack within-budget atoms into chunks, with a little overlap."""
    if len(body) <= MAX_CHARS:
        return [body]

    parts: list[str] = []
    current = ""

    for atom in _atoms(body):
        if not current:
            current = atom
        elif len(current) + len(atom) + 2 <= MAX_CHARS:
            current = f"{current}\n\n{atom}"
        else:
            parts.append(current)
            tail = current[-OVERLAP_CHARS:].lstrip()
            # Only carry overlap when the next chunk still fits with it.
            current = f"{tail}\n\n{atom}" if len(tail) + len(atom) + 2 <= MAX_CHARS else atom

    if current:
        parts.append(current)
    return parts


def chunk_page(page) -> list[Chunk]:
    """Turn a crawled/synthesised page into retrievable chunks."""
    chunks: list[Chunk] = []
    order = 0

    for heading_path, body in _split_sections(page.markdown):
        for part in _pack(body):
            if len(part.strip()) < MIN_CHARS:
                continue

            # Self-describing context: an isolated chunk must still identify its
            # page and section, both for the embedding and for the LLM prompt.
            header = page.title
            if heading_path:
                header = f"{page.title} — {heading_path}"
            text = f"{header}\n\n{part.strip()}"

            chunks.append(Chunk(
                text=text,
                heading_path=heading_path,
                order=order,
                metadata={
                    "url": page.url,
                    "title": page.title,
                    "section": heading_path,
                    "language": page.language,
                    "source_type": page.source_type,
                    **(page.metadata or {}),
                },
            ))
            order += 1

    return chunks


def chunk_pages(pages) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        out.extend(chunk_page(p))
    return out
