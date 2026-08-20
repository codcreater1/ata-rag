"""Chunking invariants — the budget guard that stopped 18k-char chunks."""

from app.ingest.chunker import MAX_CHARS, _atoms, _pack, chunk_page, faculty_of
from app.ingest.crawler import Page


def test_atoms_never_exceed_budget_even_without_punctuation():
    # A single 20k "sentence" with no breaks must still be wrapped to budget —
    # the exact shape that previously slipped through as one giant chunk.
    body = "x" * 20000
    for atom in _atoms(body):
        assert len(atom) <= MAX_CHARS


def test_pack_parts_within_budget():
    body = "\n\n".join("A sentence about tuition. " * 40 for _ in range(20))
    for part in _pack(body):
        assert len(part) <= MAX_CHARS


def test_chunk_page_produces_chunks_and_carries_metadata():
    page = Page(
        url="https://akademiata.pl/en/offer/bachelor/computer-engineering/",
        title="Computer Engineering",
        markdown="# Tuition\n\n" + ("The fee is 2900 EUR per year. " * 30),
        language="en",
        source_type="bachelor",
    )
    chunks = chunk_page(page)
    assert chunks
    first = chunks[0]
    assert page.title in first.text                      # self-describing header
    assert first.metadata["url"] == page.url
    assert first.metadata["language"] == "en"


def test_faculty_of():
    assert faculty_of("Computer Engineering") is not None
    assert faculty_of("") is None
