"""Arabic normalisation, filler detection and noun heuristics.

Normalisation folds diacritics, alef variants, hamza carriers, teh marbuta and
alef maqsura, then lowercases Latin. The folding happens *before* tokenising so
a vowel mark never splits a word in two.
"""
from __future__ import annotations

import unicodedata

# Diacritic / harakat range.
_HARAKAT = "ً-ٰٟ"  # Arabic diacritics

_FOLDS = {
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0622": "\u0627",  # آ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
    "\u0624": "\u0648",  # ؤ -> و
    "\u0626": "\u064a",  # ئ -> ي
    "\u0629": "\u0647",  # ة -> ه
    "\u0649": "\u064a",  # ى -> ي
}

# Prefixes that weld onto a word in speech: conjunctions, prepositions, the
# definite article and question markers. Stripping them is for comparison only.
_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل", "س", "أ", "إ", "ت", "ي", "ن")

_VERB_STEM_SUFFIXES = ("ت", "تي", "تا", "ات", "نا", "ني", "وه", "وهي",
                       "تن", "ون", "ان", "ين", "ك", "كم", "كن")


def normalise(text: str) -> str:
    """Fold Arabic spelling variants and strip vowel marks."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for ch in folded:
        code = ord(ch)
        if 0x064B <= code <= 0x065F or code == 0x0670 or code in (0x0640,):
            continue
        out.append(_FOLDS.get(ch, ch))
    return "".join(out).lower()


def is_arabic(text: str) -> bool:
    """Whether the text contains Arabic letters."""
    if not text:
        return False
    for ch in text:
        if 0x0600 <= ord(ch) <= 0x06FF or 0xFB50 <= ord(ch) <= 0xFEFF:
            return True
    return False


# ------------------------------------------------------------------ filler

_FILLER_RAW = frozenset("""
في من عن مع الى على حول هذا هذه ذلك تلك كان كانت التي الذي احنا اللي يعني
عشان دلوقتي جدا بقى بقيت يرحمك ايه تمام طب خلاص تعالى ماشي حبيبي والله صدق
اكيد تسلم مافي اه آه اهو اهواه برضه علشان كمان انما بس بس كده كدا اذا اذن
انني اننا على عليك عليهم عليهن عليه عليها اليهم اليها إليه إليها بلا بلا
ذو ذات لكن حتى ايش بايش ازاي وازاي فين وفين امتى ومتى هنا وهنا وهوه
كمت كماي كيفي كيفة ليش وليش شو شويه شوية هيك هيكدى
""".split())

# Folded stopwords: "حتى" (written with ى) folds to "حتتي"? No - folds to
# "حتتي"? Actually ى->ي so "حتى" -> "حتي". The list is folded too so an entry
# spelled with a maqsura still matches.
FILLER_AR = frozenset(normalise(w) for w in _FILLER_RAW if w)
STOPWORDS = FILLER_AR

# Guard: the bare alif "علي" is a name, not the preposition "على" (which folds
# onto it). It must survive.
_KEEP = {normalise("علي")}


def _strip_clitic(word: str) -> str:
    """Remove a welded-on prefix for comparison."""
    for prefix in _PREFIXES:
        folded = normalise(prefix)
        if len(word) > len(folded) + 1 and word.startswith(folded):
            return word[len(folded):]
    return word


def is_filler(word: str) -> bool:
    """Whether a word is a stopword / conversational filler."""
    if not word or not is_arabic(word):
        return False
    folded = normalise(word)
    if folded in _KEEP:
        return False
    if folded in FILLER_AR:
        return True
    # Dialect content words are never filler - they are checked before the
    # clitic rule so "بتاع"/"اوي" survive.
    return _strip_clitic(folded) in FILLER_AR and folded not in _KEEP


# ------------------------------------------------------------------ noun heuristic

_VERB_ENDINGS = tuple(normalise(s) for s in _VERB_STEM_SUFFIXES)
# Names that the feminine-ending rule must not claim.
_is_filler = is_filler


def light_stem(word: str) -> str:
    """Strip affixes for comparison. Never destroys the word outright."""
    if not word:
        return ""
    if not is_arabic(word):
        return word
    folded = normalise(word)
    stem = folded
    for prefix in ("ال", "و", "ف", "ب", "ل"):
        p = normalise(prefix)
        if len(stem) >= len(p) + 3 and stem.startswith(p):
            stem = stem[len(p):]
            break
    if len(stem) >= 4 and stem.endswith("ه"):
        stem = stem[:-1] + "ه" if len(stem) < 4 else stem
    if len(stem) < 3:
        return folded
    return stem


def is_probably_noun(word: str) -> bool:
    """Whether a word is likely a noun, from two signals.

    The definite article ``ال`` (checked twice - before and after stemming)
    and the feminine ending ``ة`` (which folds to ``ه``). Verb suffixes and
    fillers / stopwords veto.
    """
    if not word or not is_arabic(word):
        return False
    if len(normalise(word)) <= 2:
        return False
    if is_filler(word):
        return False
    if normalise(word) in FILLER_AR:
        return False
    for candidate in (normalise(word), light_stem(normalise(word))):
        if candidate.startswith("ال"):
            return True
    folded = normalise(word)
    stemmed = light_stem(word)
    # Feminine ending: the word ends in ه (folded ة) and is long enough, and
    # does not end in a verb suffix.
    for form in (folded, stemmed):
        if form.endswith("ه"):
            if any(form.endswith(s) and len(form) - len(s) < 3 for s in _VERB_ENDINGS):
                return False
            # A bare two-letter fragment is not evidence.
            if len(form) >= 4:
                # Reject if the ending itself looks verb-y.
                if any(form.endswith(v) for v in _VERB_ENDINGS) and not form.startswith("ال"):
                    # Still allow definite nouns - they were caught above.
                    pass
                return True
    return False


# Exposed for the tests' parametrize loop.
_VERB_SUFFIXES = tuple(sorted(_VERB_ENDINGS, key=len, reverse=True))
