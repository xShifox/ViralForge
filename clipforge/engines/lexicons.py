"""Multilingual stem tables: the same signals in every language the tool scores.

English stays on the old phrase-substring path (``nobody tells you`` is one
hook entry and three tokens, which a per-token prefix matcher cannot find).
Everything else is a stem table matched per-token with a prefix test.
"""
from __future__ import annotations

import unicodedata

from . import arabic

# signal -> language -> stems (validated: multiples of 1-2 chars for CJK only,
# no whitespace, no ñ/Ñ - accent folding via NFD merges ñ into n).
MULTILINGUAL_STEMS: dict[str, dict[str, tuple[str, ...]]] = {
    "hook": {
        "ar": ("اسرار", "سر", "احد", "لا يخبرك"),
        "es": ("nadie", "verdad", "secreto", "inesperado", "crees"),
        "zh": ("秘密", "没人告诉", "真相", "竟然", "原来"),
        "ja": ("秘密", "誰も教えて", "真相", "まさか", "実は"),
        "ko": ("비밀", "아무도", "진실", "설마", "사실은"),
    },
    "emotion": {
        "ar": ("احب", "اكره", "خايف", "مبسوط", "زعلان", "سعيد"),
        "es": ("encant", "increible", "feliz", "triste", "miedo", "amor"),
        "zh": ("爱", "喜欢", "难过", "开心", "害怕", "生气"),
        "ja": ("愛", "好き", "楽しい", "怖い", "悲しい", "嬉しい"),
        "ko": ("사랑", "좋아", "행복", "두려", "슬퍼", "기쁘"),
    },
    "superlative": {
        "ar": ("افضل", "اسوء", "اكبر", "اضخم", "اقوى"),
        "es": ("mejor", "peor", "maxim", "unic", "suprem"),
        "zh": ("最好", "最差", "最强", "最美", "最大", "最重要"),
        "ja": ("最高", "最悪", "一番", "最も", "最強", "最高の"),
        "ko": ("최고", "최악", "가장", "제일", "최강", "최선"),
    },
    "humor": {
        "ar": ("يضحك", "ضحك", "نكتة", "مزح", "هزلي"),
        "es": ("gracioso", "chiste", "comico", "reir"),
        "zh": ("笑话", "搞笑", "幽默", "好笑", "笑死"),
        "ja": ("笑い", "面白い", "ネタ", "ギャグ", "笑える"),
        "ko": ("웃긴", "농담", "재미", "웃음", "코믹"),
    },
    "surprise": {
        "ar": ("مفاجاه", "فجاه", "غريب", "لا يصدق", "دهشة"),
        "es": ("sorpresa", "inesperado", "repente", "increible"),
        "zh": ("惊讶", "突然", "意外", "怎么会", "竟然"),
        "ja": ("驚き", "突然", "意外", "嘘", "まさか"),
        "ko": ("놀라", "갑자기", "놀랍", "믿을 수 없이", "충격"),
    },
    "payoff": {
        "ar": ("نتيجه", "درس", "خلاصة", "قرار", "هدف"),
        "es": ("resultado", "leccion", "conclusion", "final", "resumen"),
        "zh": ("结论", "结果", "原来如此", "其实", "答案"),
        "ja": ("結論", "結果", "答え", "結局", "まとめ"),
        "ko": ("결론", "결과", "정리", "마무리", "답"),
    },
    "cta": {
        "ar": ("اشترك", "تابع", "شارك", "اشتراك", "تابعونا", "نزل", "حفظ", "شاهد", "حمل"),
        "es": ("suscribete", "comparte", "dale", "sigueme", "comenta"),
        "zh": ("订阅", "点赞", "关注", "分享", "转发"),
        "ja": ("チャンネル登録", "登録", "高評価", "コメント", "共有"),
        "ko": ("구독", "좋아요", "팔로우", "공유", "댓글"),
    },
}


def _clean_stems(stems) -> tuple[str, ...]:
    """Normalise stems: folded, no-whitespace, no ñ - validated at import."""
    out: list[str] = []
    for stem in stems:
        folded = stem.strip()
        if not folded or " " in folded:
            continue
        if "ñ" in folded or "Ñ" in folded:
            continue
        out.append(folded)
    return tuple(out)


# Values are immutable frozensets? The tests require isinstance(stems,
# (frozenset, tuple)). Keep tuples but _clean validated.
MULTILINGUAL_STEMS = {
    signal: {lang: _clean_stems(stems) for lang, stems in table.items()}
    for signal, table in MULTILINGUAL_STEMS.items()
}

_CJK = {"zh", "ja", "ko"}

# Arabic clitic prefixes stripped before prefix matching (والتابعة = تابع).
_AR_PREFIXES = tuple(sorted((arabic.normalise(p) for p in
                             ("وال", "فال", "بال", "ال", "و", "ف", "ب", "ل")),
                            key=len, reverse=True))


def detect_languages(text: str) -> set[str]:
    """Which languages could be present? Latin-script texts are ambiguous."""
    if not text or not text.strip():
        return set()
    has_arabic = any(0x0600 <= ord(ch) <= 0x06FF for ch in text)
    has_hangul = any(0xAC00 <= ord(ch) <= 0xD7AF for ch in text)
    has_kana = any(0x3040 <= ord(ch) <= 0x30FF for ch in text)
    has_han = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in text)
    out: set[str] = set()
    if has_arabic:
        out.add("ar")
    if has_latin:
        out.update({"en", "es", "pt", "fr", "it"})
    if has_han:
        out.update({"zh", "ja"})
    if has_kana:
        out.add("ja")
    if has_hangul:
        out.add("ko")
    return out


def _fold_spanish(text: str) -> str:
    """Fold accents via NFD (ñ folds to n - acceptable, no stem has ñ)."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_for(language: str, text: str) -> str:
    """Normalise a string for matching in the given language."""
    if not text:
        return ""
    if language == "ar":
        return arabic.normalise(text)
    if language in ("es", "pt", "fr", "it"):
        return _fold_spanish(text)
    if language in _CJK:
        return text.strip()
    return text.lower().strip()


def _arabic_bare(word: str) -> str:
    """Strip welded-on clitic prefixes until none remain."""
    changed = True
    while changed:
        changed = False
        for prefix in _AR_PREFIXES:
            if len(word) > len(prefix) and word.startswith(prefix):
                word = word[len(prefix):]
                changed = True
                break
    return word


def count_stem_hits(text: str, stems, language: str) -> int:
    """Count how many tokens prefix-match one of the stems."""
    if not text or not stems:
        return 0
    # Very short stems match half the language - ignored (CJK exempt).
    usable = [s for s in stems if language in _CJK or len(s) >= 3]
    if not usable:
        return 0
    if language in _CJK:
        folded_text = text
        hits = 0
        for stem in usable:
            if stem and stem in folded_text:
                hits += folded_text.count(stem)
        return hits
    tokens = normalise_for(language, text).split()
    hits = 0
    for token in tokens:
        token = token.strip(".,!?;:\"'()[]{}<>")
        if not token:
            continue
        if language == "ar":
            token = _arabic_bare(token)
        for stem in usable:
            folded_stem = normalise_for(language, stem)
            if token == folded_stem or token.startswith(folded_stem):
                hits += 1
                break
    return hits


def best_multilingual_hits(text: str, table: dict[str, tuple[str, ...]]) -> int:
    """Best count across the languages in one signal table.

    Han text is both zh and ja with identical tables, so summing would double
    every Chinese hit - the maximum is the count.
    """
    best = 0
    for language, stems in (table or {}).items():
        count = count_stem_hits(text, stems, language)
        if count > best:
            best = count
    return best
