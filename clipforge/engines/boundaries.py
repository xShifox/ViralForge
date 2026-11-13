"""Where a clip may begin and end, so it does not need editing afterwards.

The complaint this answers is specific: clips that open on "master of being
gross." or "so you can feel that.", and close on "…who lack the means to slay".
Nothing is wrong with those clips technically - they are the right length, they
are compliant, they have subtitles - but they start and finish in the middle of a
thought, and the first thing anybody does with them is trim them by hand.

Two things were causing it, and only one of them was visible in the code.

The visible one was that a clip end which found no boundary nearby was kept
anyway, at its raw timestamp. ``_snap`` returned the value unchanged when
nothing was within tolerance, so "end here unless you happen to have a sentence
ending nearby" was the rule in all but name.

The less visible one was that the scorer never saw the audio. There were four
hundred detected pauses sitting in the analysis, each with a duration, and
``generate_candidates`` was handed the transcript, the settings and the video
length - not the pauses. Whisper's punctuation is a guess derived from the audio
it already had, and it puts full stops at every breath, so a transcript alone
cannot tell a finished thought from a pause in the middle of one. A pause the
detector measured can.

So boundaries come from both, and each is labelled with how much it can be
trusted. A long pause is strong. A full stop is strong. A whisper segment edge is
weak, because segments are cut for transcription convenience and routinely begin
mid-sentence - which is exactly how "master of being gross." got to be a clip
opening in the first place.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

#: Returned by WordTimeline.last_before when the word list is unsorted and the
#: scan fallback is required.
_LAST_BEFORE_UNSUPPORTED = object()

#: A pause at least this long is treated as a possible boundary. Below it the
#: detector is reporting breath, not a break between thoughts.
PAUSE_WEAK_S = 0.35
#: A pause this long is a boundary on its own, punctuation or not.
PAUSE_STRONG_S = 0.60

#: Terminal punctuation. A comma is deliberately absent: a clip that ends on one
#: is a clause, and a clause is not a thought finished.
TERMINAL = (".", "!", "?", "…")

#: Words that cannot open a self-contained thought. A clip starting on one of
#: these is a continuation, and the viewer arrives after the setup.
CONTINUATION = frozenset("""
    so and but or nor yet because since although though while whereas if unless
    until after before when where which who whom whose that what why how
    with without within into onto from for to of in on at by as than then
    there here their his her its our your my he she it they we you them him us
    me also just even only very such both each any all some most many much
    about again still now being been having doing going getting making taking
    using saying telling seeing knowing because although since
""".split())

#: Words that are fine at the start even though they are short and common.
ALWAYS_STARTS = frozenset({"i", "a", "an", "the", "no", "ok", "okay", "yes",
                           "well", "right", "look", "listen", "stop", "wait",
                           "imagine", "remember", "consider", "meanwhile",
                           "basically", "honestly", "anyway", "alright"})

_WORD = re.compile(r"[A-Za-z']+")


def _first_word(text: str) -> str:
    """The first alphabetic word of a phrase, lowercased and stripped of marks."""
    match = _WORD.search(text or "")
    return match.group(0).lower() if match else ""


def _is_capitalised(text: str) -> bool:
    stripped = (text or "").lstrip("\"'([{")
    return bool(stripped) and stripped[0].isupper()


#: Words a clip must not *end* on. A boundary after one of these is a real pause
#: and still a bad ending: the viewer is left holding "…nine people living in it,
#: and". This is the last of the dangling endings, and it needed its own rule
#: because the pause is genuine - the speaker really did stop there - so every
#: check that only asks "did somebody pause" says yes.
DANGLING_END = frozenset("""
    and but or nor so yet because since which who whom whose that while though
    although if unless when as than then with without to of in on at by for from
    into onto about over under through during against toward towards until
    very just also even only such both each any all some most many much
""".split())


def ends_cleanly(last_word: str) -> bool:
    """Whether a clip may finish on the word before this boundary."""
    word = _first_word(last_word)
    return bool(word) and word not in DANGLING_END


def _last_before_by_scan(words, threshold: float):
    """The last word starting before ``threshold``, by linear scan."""
    found = None
    for word in words or ():
        if word.start < threshold:
            found = word
    return found


class WordTimeline:
    """Bisected index over word start times.

    A lookup is O(log n) when the list is sorted. An unsorted list is reported
    (``sorted`` is False, ``last_before`` returns ``_LAST_BEFORE_UNSUPPORTED``)
    rather than answered with a confident wrong answer - the caller must fall
    back to the scan.
    """

    def __init__(self, words):
        self.words = list(words or ())
        self._starts = [float(w.start) for w in self.words]
        self.sorted = all(
            a <= b for a, b in zip(self._starts, self._starts[1:]))

    def last_before(self, threshold: float):
        if not self.sorted:
            return _LAST_BEFORE_UNSUPPORTED
        index = bisect.bisect_left(self._starts, float(threshold)) - 1
        if index < 0:
            return None
        return self.words[index]


def _ends_cleanly_at(words, time: float, index: WordTimeline | None = None) -> bool:
    """Whether the last word before ``time`` is one a clip can finish on.

    ``words`` is the flat word list. When an index is supplied it is used for
    the lookup (with scan fallback for unsorted lists); otherwise the scan
    runs directly.
    """
    threshold = float(time) - 0.05
    if index is not None:
        found = index.last_before(threshold)
        if found is _LAST_BEFORE_UNSUPPORTED:
            found = _last_before_by_scan(words, threshold)
    else:
        found = _last_before_by_scan(words, threshold)
    if found is None:
        return True                       # nothing spoken yet; nothing to dangle
    return ends_cleanly(found.text)


@dataclass(frozen=True)
class Boundary:
    """A moment a clip may start or end on.

    ``strength`` is 2 for a break that can be trusted on its own and 1 for one
    that cannot. Callers that care about correctness use only the strong ones;
    the weak ones exist so the search has somewhere to widen to, and so a
    transcript with no usable punctuation still produces something.
    """

    time: float
    strength: int          # 2 strong, 1 weak
    reason: str

    @property
    def strong(self) -> bool:
        return self.strength >= 2


class Boundaries:
    """Every place a clip could begin or end, from the audio and the transcript."""

    def __init__(self, boundaries: list[Boundary]):
        self.all = sorted(boundaries, key=lambda b: b.time)
        self.strong = [b for b in self.all if b.strong]
        self.times = [b.time for b in self.all]
        self.strong_times = [b.time for b in self.strong]

    def __len__(self) -> int:
        return len(self.all)

    def nearest_strong(self, value: float, tol: float,
                       after: bool = False) -> Boundary | None:
        """The strong boundary closest to ``value`` within ``tol``.

        ``after`` looks only at boundaries at or past ``value``, which is what a
        clip end wants: finishing just *after* a complete thought is right,
        finishing just before one throws the ending away.
        """
        best: Boundary | None = None
        best_gap = tol
        for b in self.strong:
            gap = (b.time - value) if after else abs(b.time - value)
            if gap < 0:
                continue
            if gap <= best_gap:
                best, best_gap = b, gap
        return best

    def snap_end(self, value: float, low: float, high: float,
                 words: list | None = None) -> Boundary | None:
        """A boundary the clip can finish on, or None if there is not one.

        The tolerance widens as the search goes on rather than being fixed,
        because a fixed tolerance means a clip whose natural ending is four
        seconds away gets no ending at all. Each step is still bounded by the
        duration band, so widening never produces an over-long clip - it only
        reaches further for a boundary that was always inside the band.

        ``words`` enables the dangling-ending check: a boundary whose preceding
        word is a conjunction is a real pause and still a bad ending, so the
        next boundary is tried instead. Without it, "…living in it, and" is a
        perfectly good boundary by every other measure.
        """
        attempts = 0
        index = WordTimeline(words) if words is not None else None
        for tol in (1.2, 2.5, 4.0, 6.5, 10.0):
            for want_after in (True, False):
                found = self.nearest_strong(value, tol, after=want_after)
                if found is None:
                    continue
                if not (low - 0.01 <= found.time <= high + 0.01):
                    continue
                if words is not None and not _ends_cleanly_at(words, found.time, index):
                    # Try the next boundary instead of accepting a dangling
                    # ending. The search spans the rest of the duration band
                    # rather than reusing the current tolerance: the good
                    # ending may be any distance away, and reusing a 1.2 s
                    # tolerance here meant a clip whose next clean ending was
                    # fifteen seconds on simply got none at all.
                    attempts += 1
                    if attempts > 3:
                        continue
                    later = self.nearest_strong(found.time + 0.05,
                                                 max(0.5, high - found.time),
                                                 after=True)
                    if later is None or not (low - 0.01 <= later.time <= high + 0.01):
                        continue
                    if not _ends_cleanly_at(words, later.time, index):
                        continue
                    return later
                return found
        return None

    def start_ok(self, time: float, first_word: str, sentence_cased: bool) -> str:
        """Why this cannot be a clip opening, or "" when it can.

        Two independent ways to be wrong, and both had to be closed: the opening
        could sit on a weak boundary, or it could sit on a strong boundary that
        is nonetheless in the middle of a sentence because Whisper put a full
        stop in the wrong place.
        """
        at = self.nearest_strong(time, 0.35)
        if at is None:
            return "the clip would open mid-thought: no pause or full stop there"

        word = _first_word(first_word)
        if word in CONTINUATION and word not in ALWAYS_STARTS:
            return f"the clip would open on '{word}', which continues a thought"
        # Whisper capitalises what it believes starts a sentence, so a lowercase
        # opener is a continuation - but only on a transcript where it does that
        # reliably, or an all-lowercase transcript would have no clips at all.
        if sentence_cased and word and word != "i" and not _is_capitalised(first_word):
            return f"the clip would open on lowercase '{word}', which continues a thought"
        return ""


def build(transcript_segments, pauses) -> Boundaries:
    """Collect boundaries from the transcript's punctuation and the audio's pauses.

    Both sources are used and neither is trusted alone. Punctuation alone puts a
    full stop at every breath; pauses alone miss the places where somebody
    finished a sentence and immediately carried on without a gap.
    """
    found: list[Boundary] = []

    for pause in pauses or ():
        duration = float(getattr(pause, "duration", 0.0) or 0.0)
        if duration < PAUSE_WEAK_S:
            continue
        # A clip starts when speech resumes, not when the silence does: leading
        # a clip with a second of dead air looks like a mistake.
        found.append(Boundary(
            time=round(float(pause.end), 3),
            strength=2 if duration >= PAUSE_STRONG_S else 1,
            reason=f"a {duration:.2f}s pause",
        ))

    for segment in transcript_segments or ():
        words = list(getattr(segment, "words", ()) or ())
        if not words:
            continue
        # Segment edges are recorded, but only as weak: they are cut for
        # transcription convenience and are the source of most mid-sentence
        # clip openings.
        found.append(Boundary(time=round(float(segment.start), 3), strength=1,
                              reason="transcript segment"))
        found.append(Boundary(time=round(float(segment.end), 3), strength=1,
                              reason="transcript segment"))
        for word in words:
            text = (word.text or "").strip()
            if not text:
                continue
            if text.endswith(TERMINAL):
                found.append(Boundary(time=round(float(word.end), 3), strength=2,
                                      reason="a full stop"))
            elif text.endswith((";", ":")):
                found.append(Boundary(time=round(float(word.end), 3), strength=1,
                                      reason="a clause break"))

    # Deduplicate on time, keeping the strongest reason, so a pause and a full
    # stop at the same moment count once and the stronger one explains it.
    best: dict[float, Boundary] = {}
    for b in found:
        current = best.get(b.time)
        if current is None or b.strength > current.strength:
            best[b.time] = b
    return Boundaries(list(best.values()))


def is_sentence_cased(segments) -> bool:
    """Whether this transcript marks its sentence starts with capitals.

    Checked before the lowercase rule is applied, because a transcript that never
    uses capitals would otherwise have every single candidate rejected.
    """
    capitals = 0
    checked = 0
    for segment in segments or ():
        for word in (getattr(segment, "words", ()) or ())[:1]:
            text = (word.text or "").strip()
            if not text:
                continue
            checked += 1
            capitals += 1 if _is_capitalised(text) else 0
            if checked >= 120:
                break
        if checked >= 120:
            break
    if checked < 20:
        return False
    return capitals / checked >= 0.6
