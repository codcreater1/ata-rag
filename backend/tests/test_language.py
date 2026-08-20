"""Language detection — the piece that has bitten us most.

A wrong guess here answers a Turkish student in Polish, or an English question
in Turkish. These cases pin the two directions that regressed in practice: a
stray Turkish ``ı`` in an English sentence ("tuttıons"), and a genuine Turkish
question whose only signal is ``ı`` ("evrak neler lazım").
"""

import pytest

from app.services.rag import _detect_language, _looks_polish

CASES = [
    # English — including the dotless-ı keyboard slip that must STAY English
    ("What is the tuition for Computer Engineering?", "en"),
    ("how do I apply as an international student", "en"),
    ("whats the tuttıons for computer eng", "en"),
    ("why did you answer in turkish", "en"),
    ("give me py code", "en"),
    # Turkish — strong letters, function words, and ı-only questions
    ("Erasmus hakkında bilgi istiyorum", "tr"),
    ("Bilgisayar mühendisliği ücreti ne kadar", "tr"),
    ("evrak neler lazım", "tr"),
    ("kaç yıl sürüyor", "tr"),
    ("türk öğrenciler burs alabilir mi", "tr"),
    # Polish
    ("Ile kosztuje czesne na Informatyce?", "pl"),
    ("Jakie dokumenty są wymagane?", "pl"),
    # Ukrainian (Cyrillic)
    ("Скільки коштує навчання?", "uk"),
    ("Які документи потрібні для вступу?", "uk"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_detect_language(text, expected):
    assert _detect_language(text) == expected


def test_looks_polish():
    assert _looks_polish("Ile kosztuje czesne na Informatyce")
    assert not _looks_polish("what is the tuition for computer engineering")
