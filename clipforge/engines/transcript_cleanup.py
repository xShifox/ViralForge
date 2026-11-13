"""Remove transcription doubling: one laugh heard as three.

Only *short adjacent* repeats collapse - two copies of the same word in a
row. Naming a score never deletes speech, so anything long or spaced apart is
left alone.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.models import Transcript


@dataclass
class Report:
    removed_words: int = 0
    changed: bool = False
    samples: tuple = ()

    def __init__(self, removed_words: int = 0, changed: bool = False, samples: tuple = ()):
        self.removed_words = int(removed_words)
        self.changed = bool(changed)
        self.samples = tuple(samples or ())


def _key(word: str) -> str:
    return (word or "").strip(" .,-!?;:\"'()[]{}").lower()


def clean(transcript: Transcript):
    """Return (cleaned Transcript, Report) without mutating the original."""
    if transcript is None or not getattr(transcript, "segments", None):
        return Transcript() if transcript is None else transcript.model_copy(deep=True), Report()

    new_segments = []
    removed = 0
    samples: list[str] = []
    prev_key: str = ""

    for segment in transcript.segments:
        kept_words = []
        for word in segment.words or []:
            key = _key(word.text)
            # A short word doubled in a row is a filler doubling. A long word
            # doubled is emphasis, which is real speech.
            # ``prev_key`` resets per segment: the same short word ending
            # one segment and starting the next is two utterances, not a
            # doubling.
            if key and key == prev_key and len(key) <= 4:
                removed += 1
                samples.append(word.text)
                continue
            kept_words.append(word)
            if key:
                prev_key = key
        prev_key = ""
        if kept_words:
            text = " ".join(w.text for w in kept_words)
            new_segments.append(segment.model_copy(update={"words": kept_words, "text": text}))
        # Empty segments after cleaning are dropped.

    cleaned = Transcript(
        language=transcript.language,
        duration=transcript.duration,
        segments=new_segments,
        engine=transcript.engine,
    )
    return cleaned, Report(removed_words=removed, changed=removed > 0,
                           samples=tuple(samples[:5]))
