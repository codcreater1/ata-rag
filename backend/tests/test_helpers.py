"""Pure retrieval / cache helpers — no DB, no network."""

from app.core import cache
from app.core.db import _or_tsquery
from app.ingest.prices import _fee_phrase
from app.services import rag


def test_or_tsquery_ors_content_words_and_drops_short_ones():
    ts = _or_tsquery("What is the tuition for Computer Engineering?")
    terms = ts.split(" | ")
    assert "tuition" in terms and "computer" in terms and "engineering" in terms
    assert "is" not in terms          # <=2 chars dropped
    assert " | " in ts                # OR, not AND


def test_or_tsquery_dedupes_and_caps():
    ts = _or_tsquery("fee fee fee " + " ".join(f"word{i}" for i in range(40)))
    terms = ts.split(" | ")
    assert terms.count("fee") == 1
    assert len(terms) <= 24


def test_or_tsquery_empty_on_no_content():
    assert _or_tsquery("is a to of") == ""


def test_is_follow_up():
    assert rag._is_follow_up("and in Wrocław?")
    assert rag._is_follow_up("what about part-time?")
    assert rag._is_follow_up("is it free?")
    assert not rag._is_follow_up("What is the tuition for Computer Engineering?")
    assert not rag._is_follow_up("What documents do I need to apply?")


def test_cacheable():
    history = [{"role": "user", "content": "tuition for computer science"}]
    # a follow-up depends on the conversation, so it must not be cached
    assert not rag._cacheable("and in Wrocław?", history)
    # a self-contained question is cacheable even mid-conversation
    assert rag._cacheable("What documents do I need to apply?", history)
    # no history: always cacheable
    assert rag._cacheable("anything self contained here", [])


def test_cache_language():
    # Auto: detected from the question, so a reply is never reused cross-language
    assert rag._cache_language("evrak neler lazım", None) == "tr"
    assert rag._cache_language("what is the tuition", None) == "en"
    # An explicit picker choice wins over detection
    assert rag._cache_language("what is the tuition", "pl") == "pl"


def test_retrieval_query_prepends_topic_only_for_followups():
    history = [{"role": "user", "content": "tuition for Computer Engineering"}]
    # follow-up gets the previous subject prepended
    q = rag._retrieval_query("and in Wrocław?", history)
    assert "Computer Engineering" in q and "Wrocław" in q
    # a self-contained question is left untouched
    assert rag._retrieval_query("How do I apply?", history) == "How do I apply?"


def test_normalise_question_collapses_case_space_punctuation():
    n = cache.normalise_question
    assert n("How much is TUITION?") == n("how  much   is tuition")
    assert n("  Ile kosztuje? ") == "ile kosztuje"


def test_fee_phrase():
    assert _fee_phrase({200}, "EUR") == "200 EUR"
    assert _fee_phrase({300, 200, 250}, "EUR") == "200–300 EUR"


def test_tuition_intent():
    for q in ["what is the tuition for architecture", "how much does it cost",
              "architecture ücreti ne kadar", "ile kosztuje architektura",
              "what are the fees"]:
        assert rag._tuition_intent(q), q
    for q in ["how do I apply", "which programmes are offered in english",
              "where is the university"]:
        assert not rag._tuition_intent(q), q


def test_merge_prefer_puts_cards_first_and_dedupes():
    cards = [{"url": "a", "title": "fee A"}, {"url": "b", "title": "fee B"}]
    hits = [{"url": "c", "title": "page C"}, {"url": "a", "title": "page A dup"}]
    merged = rag._merge_prefer(cards, hits, top_k=10)
    assert [m["url"] for m in merged] == ["a", "b", "c"]   # cards first, 'a' once
