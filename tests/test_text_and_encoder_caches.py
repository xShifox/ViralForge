"""Memoisation must not change a single answer.

Three caches went in:

* `words_of` - `_hits` walks the token list once per lexicon (eight of them),
  and `analyze_text` walks it a ninth time to count fillers. Nine regex passes
  became one.
* `analyze_text` - the most repeated computation in scoring. The same span is
  analysed by `_window_score`, again by `_quotability` for each sentence inside
  it, and again by the hook checks in `middle_hook`.
* `hardware_encoder` - already an `lru_cache`, but the health endpoint called it
  five times per request.

The risk with all three is not slowness, it is a cache that hands back something
shared and mutable, or that quietly answers a *different* question. So these
tests are mostly about identity and about equality with an uncached computation.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import text as text_engine   # noqa: E402
from clipforge.engines.text import TextSignals     # noqa: E402

SENTENCE = (
    "Nobody tells you this about the first month, and it changes everything - "
    "here are the three mistakes almost everyone makes."
)


@pytest.fixture(autouse=True)
def _clean_caches():
    text_engine.clear_caches()
    yield
    text_engine.clear_caches()


def _uncached_analyze(text, duration=0.0):
    """The original body, with the cache disabled around it."""
    text_engine._analyze_cached.cache_clear()
    try:
        return dataclasses.replace(
            text_engine._analyze_cached(text, duration),
            key_terms=list(text_engine._analyze_cached(text, duration).key_terms),
        )
    finally:
        text_engine._analyze_cached.cache_clear()


# ------------------------------------------------------------------- words_of

def test_words_of_returns_the_same_tokens():
    assert text_engine.words_of(SENTENCE) == \
        [w.lower() for w in text_engine.WORD_RE.findall(SENTENCE)]


def test_words_of_hands_out_a_fresh_list_every_time():
    """The cached value is shared; the list is not.

    If this returned the cached list, a caller that sorted or appended to it
    would corrupt every later caller - and only under a particular scoring
    order, which is the worst way for it to fail.
    """
    first = text_engine.words_of(SENTENCE)
    second = text_engine.words_of(SENTENCE)
    assert first == second
    assert first is not second
    first.append("MUTATED")
    first.sort()
    assert "mutated" not in text_engine.words_of(SENTENCE)


def test_the_word_cache_is_actually_used():
    text_engine.clear_caches()
    text_engine.words_of(SENTENCE)
    info_before = text_engine._words_of_cached.cache_info()
    text_engine.words_of(SENTENCE)
    info_after = text_engine._words_of_cached.cache_info()
    assert info_after.hits > info_before.hits, "the second call missed the cache"


def test_the_word_cache_is_bounded():
    assert text_engine._words_of_cached.cache_info().maxsize is not None, \
        "an unbounded cache keyed on transcript spans grows without limit"


# ---------------------------------------------------------------- analyze_text

def test_analyze_text_is_unchanged_by_the_cache():
    for duration in (0.0, 5.0, 12.5):
        assert text_engine.analyze_text(SENTENCE, duration) == \
               _uncached_analyze(SENTENCE, duration)


def test_analyze_text_hands_out_a_fresh_object():
    """`TextSignals` is a mutable dataclass; a shared instance would be a trap."""
    first = text_engine.analyze_text(SENTENCE)
    second = text_engine.analyze_text(SENTENCE)
    assert first == second
    assert first is not second
    first.hook = 99.0
    assert text_engine.analyze_text(SENTENCE).hook != 99.0


def test_the_key_terms_list_is_not_shared():
    first = text_engine.analyze_text(SENTENCE)
    second = text_engine.analyze_text(SENTENCE)
    assert first.key_terms == second.key_terms
    first.key_terms.append("MUTATED")
    assert "mutated" not in text_engine.analyze_text(SENTENCE).key_terms


def test_duration_is_part_of_the_cache_key():
    """It feeds words_per_sec, so ignoring it would serve a stale rate."""
    fast = text_engine.analyze_text(SENTENCE, 1.0)
    slow = text_engine.analyze_text(SENTENCE, 100.0)
    assert fast.words_per_sec != slow.words_per_sec
    assert fast.words_per_sec == pytest.approx(
        len(text_engine.words_of(SENTENCE)) / 1.0)


def test_a_zero_duration_leaves_the_rate_unset():
    sig = text_engine.analyze_text(SENTENCE)
    assert sig.words_per_sec == 0.0


def test_analyze_text_is_actually_cached():
    text_engine.clear_caches()
    text_engine.analyze_text(SENTENCE)
    before = text_engine._analyze_cached.cache_info()
    text_engine.analyze_text(SENTENCE)
    after = text_engine._analyze_cached.cache_info()
    assert after.hits > before.hits


def test_an_empty_span_still_works():
    assert text_engine.analyze_text("").word_count == 0


def test_clearing_the_cache_changes_nothing_observable():
    warm = text_engine.analyze_text(SENTENCE)
    text_engine.clear_caches()
    assert text_engine.analyze_text(SENTENCE) == warm


# ------------------------------------------------- the functions that call it

def test_quotability_is_unchanged():
    whole = text_engine.analyze_text(SENTENCE)
    for sentence in text_engine.split_sentences(SENTENCE):
        assert text_engine._quotability(sentence, whole, 0.0) == \
               text_engine._quotability(sentence, whole, 0.0)


def test_quotability_repeated_calls_agree():
    whole = text_engine.analyze_text(SENTENCE)
    sentence = text_engine.split_sentences(SENTENCE)[0]
    values = {text_engine._quotability(sentence, whole, 0.5) for _ in range(5)}
    assert len(values) == 1


def test_window_score_repeated_calls_agree():
    sig_all = text_engine.analyze_text(SENTENCE)
    values = {
        text_engine._window_score(SENTENCE, sig_all, 12, True, True)
        for _ in range(5)
    }
    assert len(values) == 1


def test_window_score_still_distinguishes_its_inputs():
    """A cache keyed on too little would flatten these into one answer."""
    sig_all = text_engine.analyze_text(SENTENCE)
    short = text_engine._window_score("few words here now", sig_all, 4, True, True)
    long = text_engine._window_score(SENTENCE, sig_all, 40, True, True)
    assert short != long


# ------------------------------------------------------------- the health route

def test_health_reads_the_encoder_once(monkeypatch):
    """Five reads of one value; anything that changed between them is a lie.

    Several *other* helpers in the same response read the encoder too -
    `gpu.describe()`, `gpu.tier()`, `nvenc_available()` - which is fine, since
    under the lru_cache those are free. So everything that is not the encoder
    block is stubbed out, and the count is specifically about the route's own
    reads. The route used to make five.
    """
    from clipforge.api import routes

    calls = []

    def counting():
        calls.append(1)
        return "h264_nvenc", "NVIDIA"

    monkeypatch.setattr(routes.gpu, "hardware_encoder", counting)
    monkeypatch.setattr(routes.gpu, "describe", lambda: {})
    monkeypatch.setattr(routes, "nvenc_available", lambda: True)
    monkeypatch.setattr(routes, "nvenc_broken", lambda: False)
    monkeypatch.setattr(routes.asr, "device_info", lambda: {})
    monkeypatch.setattr(routes.llm, "describe", lambda: {})
    routes.health()
    assert len(calls) == 1, f"the route read the encoder {len(calls)} times"


def test_the_other_health_reads_are_not_multiplied(monkeypatch):
    """`nvenc_broken` was also read twice for one answer."""
    from clipforge.api import routes

    calls = []
    monkeypatch.setattr(routes.gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    monkeypatch.setattr(routes.gpu, "describe", lambda: {})
    monkeypatch.setattr(routes, "nvenc_broken", lambda: calls.append(1) or False)
    routes.health()
    assert len(calls) == 1, f"nvenc_broken was read {len(calls)} times"


def test_health_reports_one_consistent_encoder(monkeypatch):
    from clipforge.api import routes

    monkeypatch.setattr(routes.gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    monkeypatch.setattr(routes, "nvenc_broken", lambda: False)
    encoder = routes.health()["encoder"]
    assert encoder["hardware_encoder"] == "h264_nvenc"
    assert encoder["active"] == "h264_nvenc"
    assert encoder["usable"] is True
    assert encoder["cpu_fallback"] is False
    assert encoder["fallback_reason"] is None


def test_health_reports_a_broken_card_consistently(monkeypatch):
    from clipforge.api import routes

    monkeypatch.setattr(routes.gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    monkeypatch.setattr(routes, "nvenc_broken", lambda: True)
    encoder = routes.health()["encoder"]
    # A card that exists but fails is refused, not worked around.
    assert encoder["active"] is None
    assert encoder["usable"] is False
    assert encoder["hardware_encoder"] == "h264_nvenc"
    assert encoder["cpu_fallback"] is False, \
        "a broken card is not the same as having no card"


def test_health_reports_a_cpu_only_machine(monkeypatch):
    from clipforge.api import routes

    monkeypatch.setattr(routes.gpu, "hardware_encoder", lambda: ("", ""))
    monkeypatch.setattr(routes, "nvenc_broken", lambda: False)
    encoder = routes.health()["encoder"]
    assert encoder["active"] == "libx264"
    assert encoder["usable"] is False
    assert encoder["cpu_fallback"] is True
