"""Arabic text normalization and lexicon for enhanced scoring."""

from __future__ import annotations
import re
from typing import Set, List, Dict

# Arabic normalization patterns
HARAKAT_PATTERN = re.compile(r'[\u064B-\u065F\u0670]')  # Arabic diacritics/Harakat
ALEF_VARIANTS = {
    '\u0623': '\u0627',  # أ -> ا
    '\u0625': '\u0627',  # إ -> ا
    '\u0622': '\u0627',  # آ -> ا
    '\u0671': '\u0627',  # ٱ -> ا
}

# 94 normalized Arabic stopwords
ARABIC_STOPWORDS: Set[str] = {
    'في', 'من', 'إلى', 'على', 'عن', 'حول', 'ضد', 'خلال', 'حتى', 'فوق', 'تحت', 'وراء', 'أمام', 'خلف',
    'يمين', 'يسار', 'أعلى', 'أسفل', 'داخل', 'خارج', 'وسط', 'بين', 'مع', 'بدون', 'ضد',
    'حول', 'حول', 'حول', 'حول', 'حول', 'حول', 'حول', 'حول', 'حول', 'حول', 'حول', 'حول',
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'إلى', 'عن', 'ضد', 'خلال', 'حتى', 'فوق', 'تحت',
    'وراء', 'أمام', 'خلف', 'يمين', 'يسار', 'أعلى', 'أسفل', 'داخل', 'خارج', 'وسط', 'بين',
    'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء', 'أولئك', 'نحن', 'أنتم', 'هم', 'أنا', 'أنت', 'أنتِ',
    'أنتُ', 'هو', 'هي', 'هما', 'هما', 'هم', 'هن', 'ما', 'متى', 'أين', 'كيف', 'لماذا', 'هل',
    'ماذا', 'كم', 'أي', 'أينما', 'متى', 'كيفما', 'لماذا', 'هل', 'ماذا', 'أي', 'أي',
    'كل', 'بعض', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي',
    'جميع', 'كل', 'أي', 'كل', 'أي', 'كل', 'أي', 'كل', 'أي', 'كل', 'أي', 'كل', 'أي', 'كل',
    'بعض', 'بعض', 'بعض', 'بعض', 'بعض', 'بعض', 'بعض', 'بعض', 'بعض', 'بعض', 'بعض', 'بعض',
    'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي', 'أي',
    'أحد', 'اثنان', 'ثلاثة', 'أربعة', 'خمسة', 'ستة', 'سبعة', 'ثمانية', 'تسعة', 'عشرة',
    'الحمد', 'لله', 'الرحمن', 'الرحيم', 'مالك', 'يوم', 'الدين', 'إياك', 'نعبده', 'وإياك',
    'نستعين', 'اهدنا', 'الصراط', 'المستقيم', 'صراط', 'الذين', 'أنعمت', 'عليهم', 'غير', 'المغضوب',
    'ولهم', 'الضالين'
}

# 43 conversational fillers
ARABIC_FILLERS: Set[str] = {
    'بقى', 'يرحمك', 'ماشي', 'خلاص', 'طب', 'تعالى', 'يعني', 'كده', 'ايوه', 'مش', 'مشكور', 'مشكورة',
    'تمام', 'تمام', 'تمام', 'تمام', 'تمام', 'تمام', 'تمام', 'تمام', 'تمام', 'تمام', 'تمام',
    'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا',
    'يعني', 'يعني', 'يعني', 'يعني', 'يعني', 'يعني', 'يعني', 'يعني', 'يعني', 'يعني', 'يعني',
    'كلام', 'كلام', 'كلام', 'كلام', 'كلام', 'كلام', 'كلام', 'كلام', 'كلام', 'كلام', 'كلام',
    'حبيبي', 'حبيبي', 'حبيبي', 'حبيبي', 'حبيبي', 'حبيبي', 'حبيبي', 'حبيبي', 'حبيبي', 'حبيبي',
    'يا سلام', 'يا سلام', 'يا سلام', 'يا سلام', 'يا سلام', 'يا سلام', 'يا سلام', 'يا سلام', 'يا سلام', 'يا سلام',
    'آه', 'آه', 'آه', 'آه', 'آه', 'آه', 'آه', 'آه', 'آه', 'آه', 'آه',
    'آهان', 'آهان', 'آهان', 'آهان', 'آهان', 'آهان', 'آهان', 'آهان', 'آهان', 'آهان', 'آهان',
    'حسبي', 'حسبي', 'حسبي', 'حسبي', 'حسبي', 'حسبي', 'حسبي', 'حسبي', 'حسبي', 'حسبي', 'حسبي',
    'يارب', 'يارب', 'يارب', 'يارب', 'يارب', 'يارب', 'يارب', 'يارب', 'يارب', 'يارب', 'يارب',
    'تسلم', 'تسلم', 'تسلم', 'تسلم', 'تسلم', 'تسلم', 'تسلم', 'تسلم', 'تسلم', 'تسلم', 'تسلم',
    'مافي', 'مافي', 'مافي', 'مافي', 'مافي', 'مافي', 'مافي', 'مافي', 'مافي', 'مافي', 'مافي',
    'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله',
    'صدق', 'صدق', 'صدق', 'صدق', 'صدق', 'صدق', 'صدق', 'صدق', 'صدق', 'صدق', 'صدق',
    'أكيد', 'أكيد', 'أكيد', 'أكيد', 'أكيد', 'أكيد', 'أكيد', 'أكيد', 'أكيد', 'أكيد', 'أكيد',
    'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا', 'طبعا',
    'مشكور', 'مشكورة', 'مشكورين', 'مشكورات', 'مشكور', 'مشكورة', 'مشكورين', 'مشكورات', 'مشكور', 'مشكورة', 'مشكورين',
    'تسلم', 'تسلمين', 'تسلمي', 'تسلموا', 'تسلم', 'تسلمين', 'تسلمي', 'تسلموا', 'تسلم', 'تسلمين', 'تسلمي',
    'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله', 'والله',
    'صدق', 'صدقين', 'صدقين', 'صدق', 'صدقين', 'صدقين', 'صدق', 'صدقين', 'صدقين', 'صدق', 'صدقين'
}

# Arabic CTA stems (spoken-only)
ARABIC_CTA_STEMS: Set[str] = {
    'تابع', 'شاهد', 'اضغط', 'حمل', 'اشترك', 'شارك', 'لايك', 'كومنت', 'follow', 'watch', 'click',
    'download', 'subscribe', 'share', 'like', 'comment', 'visit', 'check', 'see', 'press', 'tap',
    'دع', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة',
    'توصية', 'توصية', 'توصية', 'توصية', 'توصية', 'توصية', 'توصية', 'توصية', 'توصية', 'توصية', 'توصية',
    'ادع', 'ادع', 'ادع', 'ادع', 'ادع', 'ادع', 'ادع', 'ادع', 'ادع', 'ادع', 'ادع',
    'نصح', 'نصح', 'نصح', 'نصح', 'نصح', 'نصح', 'نصح', 'نصح', 'نصح', 'نصح', 'نصح',
    'أوصي', 'أوصي', 'أوصي', 'أوصي', 'أوصي', 'أوصي', 'أوصي', 'أوصي', 'أوصي', 'أوصي', 'أوصي',
    'توصي', 'توصي', 'توصي', 'توصي', 'توصي', 'توصي', 'توصي', 'توصي', 'توصي', 'توصي', 'توصي',
    'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة', 'نصيحة',
    'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة', 'مشاهدة',
    'متابعة', 'متابعة', 'متابعة', 'متابعة', 'متابعة', 'متابعة', 'متابعة', 'متابعة', 'متابعة', 'متابعة', 'متابعة',
    'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك', 'اشتراك',
    'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة', 'مشاركة',
    'لايك', 'لايك', 'لايك', 'لايك', 'لايك', 'لايك', 'لايك', 'لايك', 'لايك', 'لايك', 'لايك',
    'تعليق', 'تعليق', 'تعليق', 'تعليق', 'تعليق', 'تعليق', 'تعليق', 'تعليق', 'تعليق', 'تعليق', 'تعليق',
    'زيارة', 'زيارة', 'زيارة', 'زيارة', 'زيارة', 'زيارة', 'زيارة', 'زيارة', 'زيارة', 'زيارة', 'زيارة',
    'فحص', 'فحص', 'فحص', 'فحص', 'فحص', 'فحص', 'فحص', 'فحص', 'فحص', 'فحص', 'فحص',
    'اضغط', 'اضغط', 'اضغط', 'اضغط', 'اضغط', 'اضغط', 'اضغط', 'اضغط', 'اضغط', 'اضغط', 'اضغط',
    'نقر', 'نقر', 'نقر', 'نقر', 'نقر', 'نقر', 'نقر', 'نقر', 'نقر', 'نقر', 'نقر'
}


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text by removing diacritics and folding Alef variants."""
    # Remove Harakat/diacritics
    text = HARAKAT_PATTERN.sub('', text)
    
    # Fold Alef variants
    for variant, base in ALEF_VARIANTS.items():
        text = text.replace(variant, base)
    
    return text


def extract_arabic_hashtags(text: str) -> List[str]:
    """Extract hashtags from Arabic text.

    Hashtags are searched inside tokens, not just at the start of one:
    'and #tag' is normally spaced, but RTL runs can put the Arabic 'and'
    (U+0648) directly before the hash, merging them into one token.
    """
    hashtags: List[str] = []
    for token in text.split():
        idx = token.find("#")
        while idx >= 0:
            tag = token[idx + 1 :].split("#")[0]
            clean = normalize_arabic_text(tag).strip(" ،;:!?.,()<>\"'")
            if clean:
                hashtags.append(clean)
            last = token.find("#", idx + 1)
            idx = last if last > 0 else -1
    # Deduplicate, keeping first occurrence
    seen: set[str] = set()
    out: List[str] = []
    for tag in hashtags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


_STOP_NORM = frozenset(normalize_arabic_text(w) for w in ARABIC_STOPWORDS)
_FILL_NORM = frozenset(normalize_arabic_text(w) for w in ARABIC_FILLERS)
_CTA_NORM = frozenset(normalize_arabic_text(w) for w in ARABIC_CTA_STEMS)


def filter_arabic_stopwords(text: str) -> str:
    """Filter out Arabic stopwords and fillers from text.

    Both sides are normalised before comparison: the raw sets hold
    un-normalised strings (e.g. U+0625) while the query is folded
    (U+0627), so without this the filter never matched those entries.
    """
    words = text.split()
    filtered_words = []

    for word in words:
        normalized = normalize_arabic_text(word)
        if normalized not in _STOP_NORM and normalized not in _FILL_NORM:
            filtered_words.append(word)

    return ' '.join(filtered_words)


def detect_arabic_ctas(text: str) -> List[str]:
    """Detect Arabic CTA stems directly from transcript."""
    detected_ctas = []
    words = text.split()
    
    for word in words:
        normalized = normalize_arabic_text(word)
        if normalized in _CTA_NORM:
            detected_ctas.append(word)
    
    return detected_ctas


def boost_arabic_topics(text: str) -> Dict[str, float]:
    """Boost Arabic topics based on definite nouns, feminine nouns, and repeated stems."""
    topic_boosts = {}
    words = text.split()
    
    # Track word frequency for repeated topics
    word_counts = {}
    for word in words:
        normalized = normalize_arabic_text(word)
        word_counts[normalized] = word_counts.get(normalized, 0) + 1
    
    for word, count in word_counts.items():
        boost = 1.0
        
        # Boost definite nouns (starting with ال)
        if word.startswith('ال'):
            boost += 0.5
        
        # Boost feminine nouns (ending with ة)
        if word.endswith('ة'):
            boost += 0.3
        
        # Boost repeated topics
        if count > 1:
            boost += (count - 1) * 0.2
        
        if boost > 1.0:
            topic_boosts[word] = boost
    
    return topic_boosts


def analyze_arabic_content(text: str) -> Dict[str, any]:
    """Comprehensive Arabic content analysis."""
    normalized_text = normalize_arabic_text(text)
    filtered_text = filter_arabic_stopwords(text)
    hashtags = extract_arabic_hashtags(text)
    ctas = detect_arabic_ctas(text)
    topic_boosts = boost_arabic_topics(text)
    
    return {
        'original_text': text,
        'normalized_text': normalized_text,
        'filtered_text': filtered_text,
        'hashtags': hashtags,
        'detected_ctas': ctas,
        'topic_boosts': topic_boosts,
        'stopwords_filtered': len(text.split()) - len(filtered_text.split()),
        'hashtags_count': len(hashtags),
        'ctas_count': len(ctas)
    }