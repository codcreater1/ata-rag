"""Evaluation harness for the ATA RAG assistant.

Runs a golden set of questions (``cases.json``) through the real answering
pipeline and scores each against declared expectations — the answer's language,
whether it answered or declined, and which facts (or forbidden strings) appear.
It exists to catch regressions: the tuition numbers, the language matching, the
scope guardrail and the injection defence all have cases here, so a change that
breaks one shows up as a red line instead of a support message weeks later.

Run from the backend directory:

    python -m eval.run_eval                 # score every case
    python -m eval.run_eval --category scope
    python -m eval.run_eval --api https://pomelo-8.codewithpeter.com

By default it calls ``rag.answer`` in-process (needs .env — DB and API keys).
Pass ``--api`` to test a deployed instance over HTTP instead. Exits non-zero if
the pass rate falls below ``--min`` (default 0.90), so it can gate CI or a
release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

CASES = Path(__file__).with_name("cases.json")

# Scope refusals and the "not found" fallback across the four languages — any of
# these means the assistant declined, which is what a scope/gate case expects.
_REFUSAL_MARKERS = (
    "only help with questions about",      # EN scope refusal
    "could not find", "contact the university",
    "nie znalazłem", "bulamadım", "не знайшов",
    "temporarily overloaded",              # provider down — still not an answer
)


# Function words are language-pure: unlike diacritics they are not dragged in by
# a place name, so an English answer that mentions "Wrocław" still scores as
# English. The language with the most function-word hits wins.
_LANG_WORDS = {
    "en": {"the", "is", "are", "for", "and", "of", "to", "you", "with", "per",
           "or", "a", "an", "in", "on", "this", "your", "can", "year", "semester"},
    "pl": {"na", "dla", "jest", "są", "oraz", "wynosi", "czesne", "roku", "lub",
           "przy", "się", "które", "można", "dokumenty", "studia", "w", "z", "i"},
    "tr": {"ve", "için", "ile", "bir", "bu", "olan", "göre", "veya", "öğrenim",
           "başvuru", "gerekli", "ücret", "belgeler", "kampüs", "üniversite"},
}


def _answer_language(text: str) -> str:
    """Language of an answer: en / pl / uk / tr, by function-word overlap."""
    t = text.lower()
    if len(re.findall(r"[а-яіїєґ]", t)) >= 3:
        return "uk"
    words = set(re.findall(r"\w+", t))
    scores = {lang: len(ws & words) for lang, ws in _LANG_WORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] else "en"


def _get_answer_local(question: str) -> dict:
    from app.services import rag

    return rag.answer(question)


def _get_answer_api(base: str):
    import httpx

    def call(question: str) -> dict:
        r = httpx.post(f"{base}/chat/ask", json={"question": question}, timeout=90)
        r.raise_for_status()
        return r.json()

    return call


def _check(case: dict, result: dict) -> list[str]:
    """Return a list of failure messages (empty means the case passed)."""
    checks = case.get("checks", {})
    answer = result.get("answer", "") or ""
    low = answer.lower()
    fails: list[str] = []

    if "answered" in checks and bool(result.get("answered")) != checks["answered"]:
        fails.append(f"answered={result.get('answered')} expected {checks['answered']}")

    if checks.get("refused"):
        refused = not result.get("answered") or any(m in low for m in _REFUSAL_MARKERS)
        if not refused:
            fails.append("expected a refusal, got an answer")

    if "lang" in checks:
        got = _answer_language(answer)
        if got != checks["lang"]:
            fails.append(f"language={got} expected {checks['lang']}")

    for kw in checks.get("contains_all", []):
        if kw.lower() not in low:
            fails.append(f"missing required '{kw}'")

    any_kw = checks.get("contains_any", [])
    if any_kw and not any(kw.lower() in low for kw in any_kw):
        fails.append(f"none of {any_kw} present")

    for kw in checks.get("not_contains", []):
        if kw.lower() in low:
            fails.append(f"forbidden '{kw}' present")

    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the ATA RAG assistant.")
    parser.add_argument("--api", help="Base URL of a deployed instance (else in-process)")
    parser.add_argument("--category", help="Only run cases in this category")
    parser.add_argument("--min", type=float, default=0.90, help="Minimum pass rate")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between cases (use ~3 against --api to stay "
                             "under the server's rate limit)")
    args = parser.parse_args()

    if args.api:
        get_answer = _get_answer_api(args.api.rstrip("/"))
        target = args.api
    else:
        from dotenv import load_dotenv

        load_dotenv()
        get_answer = _get_answer_local
        target = "in-process (rag.answer)"

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if not cases:
        print("no matching cases")
        return 1

    print(f"Running {len(cases)} cases against {target}\n")
    passed = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # [pass, total]
    started = time.perf_counter()

    for i, case in enumerate(cases):
        if args.delay and i:
            time.sleep(args.delay)
        cat = case["category"]
        by_cat[cat][1] += 1
        try:
            result = get_answer(case["question"])
            fails = _check(case, result)
        except Exception as exc:            # a broken case must not stop the run
            fails = [f"error: {type(exc).__name__}: {exc}"]

        if fails:
            print(f"  FAIL  [{cat}] {case['id']}")
            for f in fails:
                print(f"          - {f}")
        else:
            passed += 1
            by_cat[cat][0] += 1
            print(f"  ok    [{cat}] {case['id']}")

    total = len(cases)
    rate = passed / total
    elapsed = time.perf_counter() - started

    print("\nBy category:")
    for cat in sorted(by_cat):
        p, t = by_cat[cat]
        print(f"  {cat:14} {p}/{t}")

    print(f"\n{passed}/{total} passed ({rate:.0%}) in {elapsed:.0f}s")
    if rate < args.min:
        print(f"FAILED: below the {args.min:.0%} threshold")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
