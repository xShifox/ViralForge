"""Publishing metadata: hashtags, CTA quotes and the sidecar block.

Hashtags are picked from the clip's own words - named entities and nouns first,
fillers never - so a tag names the subject rather than the platform.
"""
from __future__ import annotations

import re

from . import arabic

MAX_HASHTAGS = 5
FORMAT_TAGS: tuple[str, ...] = ("fyp",)

_ENG_GENERIC = frozenset({
    "today", "want", "know", "thing", "people", "so", "the", "a", "an", "and",
    "or", "but", "if", "of", "to", "in", "on", "at", "for", "with", "that",
    "this", "it", "is", "was", "are", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "very", "just", "then", "than", "there", "here", "what", "when",
    "who", "how", "why", "all", "some", "one", "two", "get", "got", "go",
    "going", "went", "like", "about", "from", "by", "as", "up", "down", "out",
    "over", "can", "will", "would", "could", "should", "am",
})

_TOKEN = re.compile(r"[^\s#]+")

# English CTA phrases the speaker might actually say. Quoted back verbatim.
_CTA_PHRASES = (
    "follow the channel", "follow me", "subscribe", "link in bio",
    "check out", "let me know", "tell me", "comment below", "save this",
    "part two", "part 2",
)

# Arabic CTA stems (prefix match after normalisation, so clitic forms are
# reached: والاشتراك finds اشتراك).
_ARABIC_CTA_STEMS = (
    "اشترك", "تابع", "شارك", "اضغط", "حمّل", "حمل", "شاهد", "ادخل",
)


def _filler(word: str) -> bool:
    """Whether a word is conversational filler (never a topic)."""
    if not word:
        return True
    folded = arabic.normalise(word)
    if not folded:
        return True
    if arabic.is_arabic(word):
        return arabic.is_filler(word)
    return folded in _ENG_GENERIC


def _clean_tag(word: str) -> str:
    """Strip punctuation a hashtag cannot hold."""
    cleaned = ""
    for ch in (word or "").strip():
        if ch.isalnum() or ch == "_":
            cleaned += ch
    return cleaned


def topic_words(text: str) -> list[str]:
    """Rank candidate topic words: named entities, then nouns, then verbs.

    Returns normalised words, deduplicated, most topic-like first.
    """
    if not text:
        return []
    tokens = [t for t in _TOKEN.findall(text) if t.strip("#")]
    words: list[str] = []
    for token in tokens:
        cleaned = token.strip(" \t\r\n،;:!?.,()[]{}<>\"'").strip("#")
        if not cleaned:
            continue
        words.append(cleaned)
    if not words:
        return []

    # Rank: prefix-capitalised (named entity) > Arabic noun / English topic >
    # repeated words. Captured before sorting so the sort never looks a word
    # up in the list being reordered.
    def rank(item):
        index, word = item
        folded = arabic.normalise(word)
        score = 0
        if word[:1].isupper():
            score += 10
        # Frequency bonus (captured counts, not list order).
        score += counts[folded] * 2
        if arabic.is_arabic(word):
            if arabic.is_probably_noun(word):
                score += 5
            folded_stem = arabic.light_stem(word)
            if folded_stem.startswith("ال") or folded.endswith("ه"):
                score += 2
        else:
            if folded not in _ENG_GENERIC and len(folded) >= 4:
                score += 3
        return -score

    counts: dict[str, int] = {}
    folded_words = [arabic.normalise(w) for w in words]
    for folded in folded_words:
        counts[folded] = counts.get(folded, 0) + 1

    indexed = list(enumerate(words))
    ranked = sorted(indexed, key=rank)
    seen: set[str] = set()
    out: list[str] = []
    for _i, word in ranked:
        folded = arabic.normalise(word)
        if folded in seen:
            continue
        if _filler(word):
            continue
        if len(folded) < 3:
            # Short names still count (a person is a topic).
            if not (word[:1].isupper() and len(folded) >= 2):
                continue
        seen.add(folded)
        tag = arabic.normalise(word) if arabic.is_arabic(word) else word
        # Strip a leading conjunction from the tag (والحراره -> الحراره).
        if arabic.is_arabic(word):
            if len(tag) >= 4 and tag[:1] in ("و", "ف"):
                tag = tag[1:]
        out.append(tag)
    return out


def hashtags(text: str, niche_label: str = "") -> list[str]:
    """Pick up to MAX_HASHTAGS tags from text, plus the niche when given."""
    topics = topic_words(text)
    tags: list[str] = []
    for topic in topics:
        tag = _clean_tag(topic)
        if not tag:
            continue
        if tag not in tags:
            tags.append(tag)
        if len(tags) >= MAX_HASHTAGS - (1 if niche_label else 0) - len(FORMAT_TAGS):
            break
    for tag in FORMAT_TAGS:
        if tag not in tags and len(tags) < MAX_HASHTAGS:
            tags.append(tag)
    if niche_label and niche_label not in tags and len(tags) < MAX_HASHTAGS:
        tags.append(niche_label)
    return tags[:MAX_HASHTAGS]


def cta_from_dialogue(text: str) -> str:
    """The call the speaker actually made, quoted verbatim - or "".

    Never invented: the returned string is always a slice of ``text``.
    """
    if not text:
        return ""
    lower = text.lower()
    for phrase in _CTA_PHRASES:
        index = lower.find(phrase)
        if index >= 0:
            end = index + len(phrase)
            # Extend to the end of the sentence.
            sentence_end = re.search(r"[.!؟?।]", text[end:])
            if sentence_end:
                end += sentence_end.end()
            return text[index:end].strip()
    # Arabic: prefix match on a normalised word, but quote the original span.
    for match in re.finditer(r"[^\s]+", text):
        original = match.group(0)
        folded = arabic.normalise(original.strip("،;:!?.,()[]{}<>\"'"))
        for stem in _ARABIC_CTA_STEMS:
            if folded.startswith(arabic.normalise(stem)):
                # Extend the quote to the end of the sentence.
                tail = text[match.start():]
                sentence_end = re.search(r"[.!؟?。]", tail)
                if sentence_end:
                    return tail[: sentence_end.end()].strip()
                return tail.strip()
    return ""


def metadata_block(tags: list[str], cta: str = "") -> str:
    """The sidecar text: CTA first, then the hashtags."""
    parts: list[str] = []
    if cta:
        parts.append(cta)
    if tags:
        parts.append(" ".join(f"#{t}" for t in tags))
    return "\n".join(parts)


def build(text: str = "", niche_label: str = "", niche_key: str = "",
          title: str = "") -> dict:
    """Everything the runner needs: hashtags, cta, sidecar."""
    tags = hashtags(text, niche_label)
    if title and len(title.split()) >= 2:
        headline = hashtags(title, "")
        for tag in headline:
            if tag not in tags:
                tags.insert(0, tag)
        tags = tags[:MAX_HASHTAGS]
    cta = cta_from_dialogue(text)
    sidecar = metadata_block(tags, cta)
    return {"hashtags": tags, "cta": cta, "sidecar": sidecar}
