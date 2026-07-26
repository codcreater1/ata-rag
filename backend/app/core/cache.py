"""In-process caches for answers and rate limiting.

Both exist for the same reason: every question spends a Gemini embedding, and
the free tier allows 1000 of those per day for the whole site. A handful of
visitors repeating popular questions ("how much is tuition?") would otherwise
burn the budget that everyone else needs.

In-process rather than Redis: one backend container serves the site, and a cache
that empties on redeploy is fine for content that changes nightly anyway.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import OrderedDict, deque

# Answers are only as fresh as the nightly crawl, so an hour is comfortably safe
# and still lets a same-day re-index show through reasonably soon.
ANSWER_TTL_SECONDS = 3600
ANSWER_CACHE_SIZE = 500

_lock = threading.Lock()
_answers: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()


def normalise_question(question: str) -> str:
    """Key form of a question, so trivial variations share one cache entry.

    "How much is tuition?", "how much is tuition" and "How  much is TUITION?"
    are the same question; casing, punctuation and spacing should not each cost
    a separate embedding.
    """
    text = unicodedata.normalize("NFKC", question).strip().lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def get_answer(question: str, language: str | None) -> dict | None:
    """Cached payload for this question, or None."""
    key = (normalise_question(question), language or "auto")
    now = time.time()
    with _lock:
        entry = _answers.get(key)
        if not entry:
            return None
        stored_at, payload = entry
        if now - stored_at > ANSWER_TTL_SECONDS:
            del _answers[key]
            return None
        _answers.move_to_end(key)          # keep hot entries alive
        return payload


def put_answer(question: str, language: str | None, payload: dict) -> None:
    key = (normalise_question(question), language or "auto")
    with _lock:
        _answers[key] = (time.time(), payload)
        _answers.move_to_end(key)
        while len(_answers) > ANSWER_CACHE_SIZE:
            _answers.popitem(last=False)   # evict least recently used


def clear_answers() -> None:
    """Drop cached answers — called after a re-index so new content shows."""
    with _lock:
        _answers.clear()


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

_hits: dict[str, deque[float]] = {}


def rate_limited(client: str) -> bool:
    """True when *client* has exceeded the per-window allowance.

    A sliding window over recent timestamps: generous enough that nobody asking
    questions in good faith will notice, tight enough that a script cannot drain
    the day's embedding quota in a minute.
    """
    now = time.time()
    with _lock:
        window = _hits.setdefault(client, deque())
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= RATE_LIMIT_REQUESTS:
            return True

        window.append(now)

        # Occasional sweep so idle clients do not accumulate forever.
        if len(_hits) > 1000:
            for ip in [k for k, v in _hits.items() if not v or v[-1] < cutoff]:
                _hits.pop(ip, None)
        return False
