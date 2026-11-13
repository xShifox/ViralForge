"""Lexical analysis and catchy-phrase generation.

Pure-python + regex, no model downloads. The lexicons below encode what
actually tends to make a short clip spread: an explicit hook, emotional
charge, humour, surprise, concrete numbers, and a strong claim rather than
small talk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------- lexicons

HOOK_PHRASES = [
    "i can't believe", "i cannot believe", "you won't", "you know what", "here's why",
    "this is why", "the reason", "the best part", "the worst part", "it gets worse",
    "the truth is", "the secret", "nobody tells you", "nobody tells me", "let me tell you",
    "guess what", "i was wrong", "i finally", "i never thought", "what happened",
    "here's what", "imagine this", "listen to me", "the funniest", "the craziest",
    "the moment", "turns out", "it turns out", "plot twist",
    "the weirdest", "the strangest", "i still think about", "changed everything",
    "biggest mistake", "one thing i", "if you only", "the only way", "nobody expects",
]

# Connectives that look like hooks but are extremely common in ordinary speech.
# They must not count as much as a genuinely arresting opening line.
SOFT_HOOK_PHRASES = [
    "and then", "but then", "what happened", "here's what", "that", "so",
]

EMOTION_WORDS = {
    "love": 1.0, "hate": 0.9, "scared": 0.9, "terrified": 1.0, "afraid": 0.8,
    "incredible": 0.9, "amazing": 0.85, "beautiful": 0.8, "heartbreaking": 1.0,
    "angry": 0.85, "furious": 0.9, "excited": 0.85, "proud": 0.8, "lonely": 0.85,
    "grateful": 0.8, "devastated": 1.0, "hopeful": 0.7, "nervous": 0.8, "relief": 0.75,
    "shocked": 0.95, "stunned": 0.9, "disappointed": 0.8, "anxious": 0.8,
    "overwhelmed": 0.9, "obsessed": 0.85, "gutted": 0.9, "thrilled": 0.9,
    "heartwarming": 0.95, "moving": 0.7, "grateful": 0.8, "worth it": 0.75,
    "meaningful": 0.75, "beautiful": 0.8, "peaceful": 0.7, "freedom": 0.7,
}

HUMOR_MARKERS = [
    "haha", "hahaha", "lol", "funny", "hilarious", "ridiculous", "absurd",
    "laughing", "joke", "comedy", "silly", "crazy", "this is so", "i cannot",
    "i can't", "no way", "are you serious", "stop", "come on", "killing me",
    "dying", "crack up", "😂",
]

SURPRISE_WORDS = [
    "actually", "unexpected", "unexpectedly", "suddenly", "out of nowhere",
    "wait", "what", "seriously", "incredible", "never seen", "first time ever",
    "nobody", "no one", "unbelievable", "wild", "insane", "bizarre", "mysterious",
    "it just", "and then", "turns out", "secretly", "apparently",
]

SUPERLATIVES = [
    "best", "worst", "biggest", "smallest", "most", "least", "greatest", "largest",
    "never", "always", "everyone", "no one", "impossible", "insane", "crazy",
    "extreme", "ultimate", "perfect", "worst", "entirely", "completely", "totally",
    "absolutely", "literally", "hundred percent", "one hundred percent",
]

CTA_WORDS = [
    "follow", "subscribe", "like this", "comment", "share", "link in", "check out",
    "part two", "part 2", "tell me", "let me know", "drop a", "save this",
    "my other video", "more videos", "watch this",
]

# Only genuinely content-free fillers live here: this list is used to build the
# on-screen headline, never the subtitles, so it can be conservative. Words like
# "like", "so", "well" and "right" were removed because they often carry real
# meaning ("it looked like a ghost movie").
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "erm", "ah", "hmm", "mm", "mhm", "you know", "i mean",
    "sort of", "kind of", "alright", "anyway", "you see", "if you will", "you know what",
}

# words that signal a "conclusion/payoff" - good places to end a clip
PAYOFF_WORDS = [
    "and that's", "that's why", "so i", "and then", "the lesson", "the point is",
    "which is why", "the truth", "in the end", "turns out", "and the answer",
    "the best part", "so the lesson", "anyway", "the takeaway",
]

WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
# The look-ahead must also accept a digit, otherwise "…for the next 20 years.
# 100%." never splits and the two unrelated statements get glued together.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…؟?])\s+(?=[A-Z0-9\"'(])|\n+")

_caches_cleared = 0


def clear_caches() -> None:
    """Drop cached analyses (the multilingual tables live in lexicons)."""
    global _caches_cleared
    _caches_cleared += 1
NUMBER_RE = re.compile(r"\b\d[\d,.]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|dozen|half|twice|three times)\b", re.I)


@dataclass
class TextSignals:
    """Normalised 0..1 linguistic features for a stretch of dialogue."""

    hook: float = 0.0
    emotion: float = 0.0
    humor: float = 0.0
    surprise: float = 0.0
    superlative: float = 0.0
    number: float = 0.0
    cta: float = 0.0
    payoff: float = 0.0
    question: float = 0.0
    exclam: float = 0.0
    word_count: int = 0
    words_per_sec: float = 0.0
    filler_ratio: float = 0.0
    key_terms: list[str] = field(default_factory=list)


# ----------------------------------------------------------------- primitives

def words_of(text: str) -> list[str]:
    """Tokenise text into words.

    Arabic is tokenised too - an ASCII-only pattern tokenised *nothing* out
    of Arabic, leaving every text signal zero. Arabic tokens are normalised
    first so a vowel mark never splits a word in two.
    """
    from . import arabic

    out: list[str] = []
    for match in WORD_RE.finditer(text or ""):
        token = match.group(0)
        # An apostrophe inside an English word belongs to the word ("can't").
        if token == "'":
            continue
        if arabic.is_arabic(token):
            token = arabic.normalise(token)
            if token:
                out.append(token)
        else:
            out.append(token.lower())
    return out


# Unicode ranges that imply a different writing system, and therefore a
# different font. A Latin display face (Segoe UI Black) has no Arabic or CJK
# glyphs at all, so drawing non-Latin text with it produces tofu boxes.
SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "arabic": (0x0600, 0x06FF),
    "hebrew": (0x0590, 0x05FF),
    "cyrillic": (0x0400, 0x04FF),
    "greek": (0x0370, 0x03FF),
    "devanagari": (0x0900, 0x097F),
    "thai": (0x0E00, 0x0E7F),
    "kana": (0x3040, 0x30FF),          # hiragana + katakana
    "hangul": (0xAC00, 0xD7AF),
    "cjk": (0x4E00, 0x9FFF),            # unified ideographs
    "cjk_ext": (0x3400, 0x4DBF),
}


def detect_scripts(text: str) -> list[str]:
    """Return the non-Latin writing systems present in ``text``."""
    found: set[str] = set()
    for char in text:
        code = ord(char)
        for name, (low, high) in SCRIPT_RANGES.items():
            if low <= code <= high:
                found.add(name)
                break
    return sorted(found)


def is_rtl(text: str) -> bool:
    """True when the text is predominantly right-to-left.

    A line that merely *contains* a few Arabic words inside mostly-Latin text
    is still laid out left-to-right, so this looks at the ratio rather than
    mere presence.
    """
    rtl_chars = sum(1 for c in text if _bidi_class(c) == "rtl")
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and rtl_chars / letters > 0.4


# ------------------------------------------------------- bidirectional reordering

# Direction classes used while reordering a line.
_RTL, _LTR, _NEUTRAL = "rtl", "ltr", "neutral"

# Explicit direction marks carry no visible glyph but do state an intent, so
# they must survive the reordering pass rather than being dropped as neutrals.
_DIR_MARKS = {"\u200f": _RTL, "\u200e": _LTR, "\u061c": _RTL}

# Arabic-Indic and Extended Arabic-Indic digits belong to the Arabic run, not to
# a separate left-to-right island, unlike ASCII digits.
_RTL_DIGIT_RANGES = ((0x0660, 0x0669), (0x06F0, 0x06F9), (0x0750, 0x077F))


def _bidi_class(char: str) -> str:
    """Classify one character as right-to-left, left-to-right or neutral."""
    if char in _DIR_MARKS:
        return _DIR_MARKS[char]
    if char.isdigit():
        code = ord(char)
        if any(low <= code <= high for low, high in _RTL_DIGIT_RANGES):
            return _RTL
        return _LTR
    if not char.isalpha():
        return _NEUTRAL
    code = ord(char)
    if (0x0590 <= code <= 0x08FF or 0xFB1D <= code <= 0xFEFC
            or any(low <= code <= high for low, high in _RTL_DIGIT_RANGES)):
        return _RTL
    return _LTR


def _token_direction(token: str, base: str) -> str:
    """Direction of a whitespace-free token, from its first strong character.

    Trailing punctuation rides along with the token, which is what keeps a comma
    glued to the word it follows after the word order is reversed.
    """
    for char in token:
        kind = _bidi_class(char)
        if kind != _NEUTRAL:
            return kind
    return base


# Zero-width left-to-right mark. Strong LTR, so it splits a line into one bidi
# run per word.
LRM = "\u200e"


def bidi_reorder(text: str) -> str:
    """Return ``text`` with its words in the order libass will display them.

    libass cannot be asked to lay out a right-to-left paragraph: it ignores
    RLM/LRM/ALM marks, ignores a ``Language`` field in ``[Script Info]`` (every
    combination renders a byte-identical frame), and its own base-direction guess
    is inconsistent - a short Arabic line comes out right-to-left while a longer
    one silently flips to left-to-right.

    So the order is decided here instead. The words are reversed, because the
    first word of a right-to-left sentence belongs on the right, and each word is
    then separated by an LRM. Being strong LTR, the mark makes every word its own
    bidi run: libass lays the runs out in the order given and shapes each
    right-to-left word correctly inside its run. The result is the same whatever
    base direction libass picks, which is the point - it stops depending on it.
    """
    stripped = text.strip()
    if not stripped or not is_rtl(stripped):
        return text

    words = re.findall(r"\S+", stripped)
    if len(words) < 2:
        # A single word has no order to get wrong; libass shapes it correctly.
        return text
    return LRM + (LRM + " ").join(reversed(words))


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def strip_fillers(text: str) -> str:
    out = text
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        out = re.sub(rf"(?<!\w){re.escape(filler)}(?!\w)[, ]*", "", out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out)
    return re.sub(r"\s+([,.!?])", r"\1", out).strip(" ,")


def _hits(lower_text: str, table) -> int:
    if isinstance(table, dict):
        return sum(1 for w in words_of(lower_text) if w in table)
    return sum(1 for phrase in table if phrase in lower_text)


def _saturate(count: int, scale: float) -> float:
    """Map a raw hit count onto 0..1 with diminishing returns."""
    if count <= 0:
        return 0.0
    return min(1.0, count / scale)


# ------------------------------------------------------------------- analysis

def analyze_text(text: str, duration: float = 0.0) -> TextSignals:
    """Compute linguistic engagement signals for a span of dialogue.

    English keeps the phrase-substring path. Other languages take the
    maximum of the English count and the multilingual stem count, so the
    multilingual pass can only raise a signal, never subtract from one the
    English path already found.
    """
    from . import lexicons

    sig = TextSignals()
    lower = text.lower()
    tokens = words_of(lower)
    sig.word_count = len(tokens)
    if not tokens:
        return sig

    sig.hook = max(
        _saturate(_hits(lower, HOOK_PHRASES), 1.5),
        0.45 * _saturate(_hits(lower, SOFT_HOOK_PHRASES), 2.0),
    )
    sig.emotion = _saturate(_hits(lower, EMOTION_WORDS), 2.0)
    sig.humor = _saturate(_hits(lower, HUMOR_MARKERS), 2.0)
    sig.surprise = _saturate(_hits(lower, SURPRISE_WORDS), 3.0)
    sig.superlative = _saturate(_hits(lower, SUPERLATIVES), 3.0)
    sig.number = min(1.0, 0.45 * len(NUMBER_RE.findall(lower)))
    sig.cta = _saturate(_hits(lower, CTA_WORDS), 1.0)
    sig.payoff = _saturate(_hits(lower, PAYOFF_WORDS), 1.0)

    # Multilingual pass: maximum with the English count (never a replacement).
    for signal, scale in (("hook", 1.5), ("emotion", 2.0), ("humor", 2.0),
                          ("surprise", 3.0), ("superlative", 3.0),
                          ("cta", 1.0), ("payoff", 1.0)):
        table = lexicons.MULTILINGUAL_STEMS.get(signal, {})
        if not table:
            continue
        best = lexicons.best_multilingual_hits(text, table)
        candidate = _saturate(best, scale)
        current = getattr(sig, signal)
        setattr(sig, signal, max(current, candidate))

    sentences = split_sentences(text)
    if sentences:
        sig.question = min(1.0, sum(1 for s in sentences if s.rstrip().endswith(("?", "؟", "？"))) / 2.0)
        sig.exclam = min(1.0, sum(1 for s in sentences if "!" in s or "！" in s) / 2.0)

    fillers = sum(1 for w in tokens if w in FILLER_WORDS)
    sig.filler_ratio = fillers / len(tokens)
    if duration > 0:
        sig.words_per_sec = len(tokens) / duration

    sig.key_terms = key_terms(text)
    return sig


def key_terms(text: str, limit: int = 6) -> list[str]:
    """Frequency-ranked content words, used for fallback titles and tags."""
    stop = {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
        "for", "with", "that", "this", "it", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "i", "you",
        "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
        "his", "its", "our", "their", "so", "not", "no", "yes", "very", "just",
        "then", "than", "there", "here", "what", "when", "who", "how", "why",
        "all", "some", "one", "two", "get", "got", "go", "going", "went", "like",
        "about", "from", "by", "as", "up", "down", "out", "over", "can", "will",
        "would", "could", "should", "am", "s", "t", "don", "didn", "doesn",
    }
    counts: dict[str, int] = {}
    for word in words_of(text):
        if len(word) < 4 or word in stop:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


# ------------------------------------------------------------ phrase generation

VIRAL_PREFIXES = [
    "This is why", "Nobody tells you", "I was completely wrong about",
    "The moment everything changed", "Plot twist", "Wait for it",
    "The part nobody tells you", "Everyone gets this wrong",
    "The truth about", "I can't believe this happened",
]

MAX_PHRASE_CHARS = 46
MIN_PHRASE_WORDS = 3

# A sentence opening with one of these depends on the previous line for its
# meaning; lifting it onto a banner reads as broken English.
DANGLING_OPENERS = (
    "so ", "and ", "but ", "because ", "since ", "which ", "that ", "those ",
    "he ", "she ", "they ", "we ", "it was ", "it is ", "there was ", "there is ",
    "he said", "she said", "they said", "i said", "you said", "and then", "so then",
)

# Openers that make a line self-contained and quotable. "that" is deliberately
# absent: after a run-on split it almost always leans on the previous sentence
# ("...the best of it. That I decided to..."), which reads as broken on a banner.
STRONG_OPENERS = (
    "i ", "i'm", "i've", "i'll", "this ", "these ", "you ", "we ", "my ",
    "the ", "it ", "nothing ", "everything ", "everyone ", "no one ", "nobody ",
    "never ", "always ", "if ", "don't ", "do not ", "stop ", "listen ", "imagine ",
    "what ", "why ", "how ", "here ", "every ", "most ", "wait ",
)


def _truncate_words(text: str, max_chars: int) -> str:
    text = text.strip(" .,-")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,.;:-")


def _is_dangling(sentence: str) -> bool:
    low = sentence.lower()
    return any(low.startswith(op) for op in DANGLING_OPENERS)


def _opener_strength(sentence: str) -> float:
    low = sentence.lower()
    if any(low.startswith(op) for op in STRONG_OPENERS):
        return 1.0
    if _is_dangling(low):
        return 0.0
    return 0.5


def _quotability(sentence: str, sig_all: TextSignals, position: float) -> float:
    """How good is this sentence as a big on-screen headline?"""
    words = sentence.split()
    n = len(words)
    if n < 2 or n > 22:
        return 0.0

    sig = analyze_text(sentence)
    score = (
        0.30 * sig.hook
        + 0.22 * sig.emotion
        + 0.20 * sig.humor
        + 0.18 * sig.superlative
        + 0.12 * sig.surprise
        + 0.10 * sig.exclam
        + 0.08 * sig.question
    )

    # ideal headline length: punchy but not a fragment
    if 4 <= n <= 12:
        score += 0.28
    elif n <= 16:
        score += 0.12
    else:
        score -= 0.10

    # a headline has to stand on its own without the preceding line
    score += 0.20 * _opener_strength(sentence)

    # reward statements over questions for the big top banner
    if sentence.rstrip().endswith("?"):
        score -= 0.08
    if sentence.rstrip().endswith(("!", ".")):
        score += 0.05

    # slightly favour earlier lines: they read as an opening hook
    score += 0.18 * (1.0 - position)

    if sig_all.humor > 0.4 and sig.humor == 0:
        score -= 0.20
    return score


ATTRIBUTION_RE = re.compile(
    r"^\s*(?:and\s+|but\s+|so\s+|then\s+)?(?:(?:i|you|he|she|they|we)\s+)?"
    r"(?:said|says|asked|told me|tells me|whispered|added|explained|goes|replies|answered)\s*"
    r"(?:that\s+|to\s+me\s+|me\s+)?[:,]?\s*",
    re.I,
)

# Leading connectives: harmless in conversation, awkward on a banner.
LEADING_CONNECTIVE_RE = re.compile(
    r"^\s*(?:and|but|so|then|plus|also|now|well|okay|ok|anyway|actually|honestly|but yeah)\b[,:]?\s*",
    re.I,
)

# a clause starting with one of these cannot stand on its own
PRONOUN_LEAD_RE = re.compile(
    r"^\s*(?:it|he|she|they|them|him|her|this|that|these|those|which|who)\b", re.I
)

CLAUSE_SPLIT_RE = re.compile(
    r",\s+(?:and|but|so|then|because|which|while|although)\s+|;\s+", re.I
)


def _strip_attribution(sentence: str) -> str:
    """Turn "He said, do not measure your life" into the bare quote."""
    stripped = ATTRIBUTION_RE.sub("", sentence, count=1).strip()
    # only accept the strip if it removed something real
    if len(stripped.split()) >= 3:
        return stripped
    return sentence.strip()


def _clause_candidates(sentence: str) -> list[str]:
    """Break a long sentence into standalone clauses worth headlining.

    A 40-word sentence truncated to 46 characters always cuts mid-thought, so
    a single punchy clause reads far better on screen.
    """
    out: list[str] = []
    for part in CLAUSE_SPLIT_RE.split(sentence):
        part = part.strip(" ,;")
        if not part:
            continue
        n = len(part.split())
        if not (3 <= n <= 14):
            continue
        if PRONOUN_LEAD_RE.match(part):
            continue
        out.append(part)
    return out


# A banner fragment must not begin or end on a word that needs a neighbour.
BAD_START_WORDS = {
    "and", "but", "so", "or", "because", "if", "when", "while", "which", "who",
    "whom", "whose", "then", "than", "there", "here", "of", "in", "on", "at", "to",
    "for", "with", "from", "by", "as", "into", "onto", "over", "under", "up", "down",
    "out", "off", "about", "after", "before", "just", "also", "very", "really",
    "quite", "even", "still", "ever", "never", "not", "no", "yes", "is", "are", "was",
    "were", "be", "been", "being", "am", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
}

BAD_END_WORDS = {
    "and", "but", "or", "so", "because", "if", "when", "while", "which", "who",
    "that", "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "into",
    "the", "a", "an", "my", "your", "his", "her", "its", "our", "their", "is", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "than", "then", "this", "these", "those",
    # a headline that trails off on a bare object pronoun reads as truncated
    "it", "them", "him", "her", "us", "me", "this", "one", "something", "anything",
}

BARE_PRONOUNS = {"it", "this", "that", "they", "he", "she", "we", "i", "you", "there"}
PREPOSITIONS = {
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "into", "onto",
    "over", "under", "about", "after", "before", "to", "than", "through", "between",
}

# "...can become" / "...will be" reads as a cut-off thought, so a span ending on
# an auxiliary is treated as an incomplete idea.
AUXILIARIES = {
    "can", "could", "will", "would", "should", "must", "may", "might", "shall",
    "is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had",
}


def _span_is_grammatical(words: list[str]) -> bool:
    """Reject spans that would look like a fragment on screen."""
    if len(words) < 3:
        return False
    first = words[0].lower().strip(",.;:!?'\"")
    last = words[-1].lower().strip(",.;:!?'\"")
    if first in BAD_START_WORDS or last in BAD_END_WORDS:
        return False
    # "it in moments" / "they with me" are word salad
    if len(words) > 1 and first in BARE_PRONOUNS and words[1].lower() in PREPOSITIONS:
        return False
    # "can become" / "will be" - the thought continues past the span
    if len(words) > 1 and words[-2].lower() in AUXILIARIES and first not in AUXILIARIES:
        return False
    return True


def _window_score(span: str, sig_all: TextSignals, n_words: int, is_start: bool, ends_sentence: bool) -> float:
    """Score a candidate window.

    Weighted differently from whole-sentence quotability: here the lexical
    signal has to dominate, otherwise the structural terms (which are nearly
    identical for competing spans) end up decided by trivia and the headline
    drifts to a bland fragment.
    """
    sig = analyze_text(span)
    signal = (
        0.45 * sig.hook
        + 0.40 * sig.emotion
        + 0.40 * sig.humor
        + 0.35 * sig.superlative
        + 0.25 * sig.surprise
        + 0.20 * sig.number
        + 0.10 * sig.exclam
    )
    # 4-12 words is the sweet spot for a headline
    if 4 <= n_words <= 12:
        length_fit = 0.10
    elif n_words <= 15:
        length_fit = 0.04
    else:
        length_fit = -0.04
    # more information is generally better, all else equal
    content = min(0.16, 0.02 * n_words)
    opener = 0.10 * _opener_strength(span)
    trailing = 0.02 if ends_sentence else 0.0
    head = 0.01 if is_start else 0.0
    return signal + length_fit + content + opener + trailing + head


def _best_window(sentence: str, max_chars: int, sig_all: TextSignals) -> str:
    """Pull the strongest contiguous word span that fits on the banner.

    Truncating a long line at ``max_chars`` always cuts mid-thought
    ("...can become" / "...years, measure it"). Sliding a window over the words
    and scoring each span keeps the most quotable *complete* idea instead.
    """
    words = sentence.split()
    if len(words) < 6:
        return _truncate_words(sentence, max_chars)

    # a window may not cross a sentence boundary, otherwise unrelated clauses
    # get glued together ("next 20 years. 100%")
    sentence_ends = {
        index for index, word in enumerate(words)
        if word.rstrip("!?.").rstrip(",;:").endswith((".", "!", "?", "…"))
        or word.rstrip().endswith((".", "!", "?", "…"))
    }

    best: tuple[float, str] | None = None
    for start in range(len(words)):
        limit = len(words)
        for end_index in sorted(sentence_ends):
            if end_index >= start:
                limit = end_index + 1
                break
        for stop in range(start + 3, min(len(words), start + 15, limit) + 1):
            span_words = words[start:stop]
            span = " ".join(span_words)
            if len(span) > max_chars:
                break
            if not _span_is_grammatical(span_words):
                continue
            score = _window_score(
                span,
                sig_all,
                len(span_words),
                start == 0,
                span_words[-1].endswith((".", "!", "?")),
            )
            if best is None or score > best[0]:
                best = (score, span)

    if best is None or best[0] < 0.20:
        return _truncate_words(sentence, max_chars)

    # strip trailing punctuation pulled in from the source text
    return best[1].strip(" ,;-")


def _strip_leading_connective(sentence: str) -> str:
    """Drop a leading "and/but/so" and return the remainder if usable."""
    candidate = LEADING_CONNECTIVE_RE.sub("", sentence).strip()
    if len(candidate.split()) >= 3 and not _is_dangling(candidate):
        return candidate
    return sentence


def generate_phrase(
    text: str,
    max_chars: int = MAX_PHRASE_CHARS,
    uppercase: bool = True,
    seed: int = 0,
    topic_hint: str = "",
) -> tuple[str, str]:
    """Build the bold on-screen phrase from the clip's own dialogue.

    Quality beats cleverness here: a verbatim self-contained line from the clip
    almost always reads better than a viral frame wrapped around a fragment, so
    the "hook" style only fires when the core is genuinely self-contained.

    Returns ``(phrase, style)`` where style is one of ``quote`` / ``hook`` /
    ``topic`` / ``fallback`` so the UI can label the provenance.
    """
    cleaned = strip_fillers(text)
    sentences = split_sentences(cleaned)
    sig_all = analyze_text(cleaned)
    if not sentences:
        if topic_hint:
            return (topic_hint.upper() if uppercase else topic_hint), "topic"
        return "HIGHLIGHT", "fallback"

    def finish(value: str, style: str) -> tuple[str, str]:
        return (value.upper() if uppercase else value), style

    scored = [
        (_quotability(s, sig_all, i / max(1, len(sentences) - 1)), s)
        for i, s in enumerate(sentences)
    ]
    scored.sort(key=lambda kv: kv[0], reverse=True)
    best_score, best = scored[0]

    # 1. a complete, self-contained line becomes the headline verbatim
    if best_score >= 0.45 and _opener_strength(best) >= 0.5 and not _is_dangling(best):
        phrase = _best_window(best, max_chars, sig_all)
        if len(phrase.split()) >= MIN_PHRASE_WORDS:
            return finish(phrase, "quote")

    # 2. a long line yields to its single most quotable clause
    for _score, sentence in scored:
        for clause in _clause_candidates(sentence):
            if len(clause) > max_chars:
                continue
            clause_score = _quotability(clause, sig_all, 0.0)
            if clause_score >= 0.45 and not _is_dangling(clause):
                if len(clause.split()) >= MIN_PHRASE_WORDS:
                    return finish(clause, "quote")

    # 3. drop the attribution and headline the quote itself
    for _score, sentence in scored:
        if not _is_dangling(sentence):
            continue
        bare = _strip_attribution(sentence)
        bare = _strip_leading_connective(bare)
        if _is_dangling(bare):
            continue
        phrase = _best_window(bare, max_chars, sig_all)
        if len(phrase.split()) >= MIN_PHRASE_WORDS:
            return finish(phrase, "quote")

    # 4. a long line that never became self-contained still has a good window
    for _score, sentence in scored:
        if len(sentence) > max_chars:
            phrase = _best_window(sentence, max_chars, sig_all)
            if len(phrase.split()) >= MIN_PHRASE_WORDS and not _is_dangling(phrase):
                return finish(phrase, "quote")

    # 5. a short, self-contained claim can carry a viral frame
    for _score, sentence in scored:
        if not (3 <= len(sentence.split()) <= 9):
            continue
        if _opener_strength(sentence) < 0.9:
            continue
        core = _truncate_words(sentence.rstrip(".!?"), max_chars - 24)
        if len(core.split()) < 3:
            continue
        prefix = VIRAL_PREFIXES[seed % len(VIRAL_PREFIXES)]
        return finish(f"{prefix}: {core}", "hook")

    # 6. last resort: the topic of the clip
    if sig_all.key_terms:
        return finish(", ".join(sig_all.key_terms[:3]), "topic")
    return finish(topic_hint or "HIGHLIGHT", "fallback")


def subtitle_text(text: str) -> str:
    """Light normalisation for on-screen subtitles (no content changes)."""
    out = re.sub(r"\s+", " ", text).strip()
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    return out


def describe_speaker_turns(transcript) -> list[dict]:
    """Flat list of speaker turns for the UI timeline."""
    turns = []
    previous = None
    for seg in transcript.segments:
        if seg.speaker and seg.speaker != previous:
            turns.append({
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
            })
            previous = seg.speaker
        elif turns:
            turns[-1]["end"] = seg.end
    return turns
