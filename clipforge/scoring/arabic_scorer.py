"""Arabic-specific scoring enhancements."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
from ..core.arabic_lexicon import (
    normalize_arabic_text,
    filter_arabic_stopwords,
    detect_arabic_ctas,
    boost_arabic_topics,
    analyze_arabic_content
)
from ..core.models import Clip, ScoreBreakdown

log = logging.getLogger("clipforge")


def score_arabic_content(clip: Clip) -> ScoreBreakdown:
    """Calculate enhanced scores for Arabic content."""
    if not clip.cues:
        return ScoreBreakdown()
    
    # Analyze Arabic content
    analysis = analyze_arabic_content(clip.text)
    
    # Base scores
    text_hook_score = calculate_arabic_text_hook_score(clip, analysis)
    emotion_score = calculate_arabic_emotion_score(clip, analysis)
    humor_score = calculate_arabic_humor_score(clip, analysis)
    surprise_score = calculate_arabic_surprise_score(clip, analysis)
    energy_score = calculate_arabic_energy_score(clip, analysis)
    laughter_score = calculate_arabic_laughter_score(clip, analysis)
    pause_rhythm_score = calculate_arabic_pause_rhythm_score(clip, analysis)
    structure_score = calculate_arabic_structure_score(clip, analysis)
    topic_score = calculate_arabic_topic_score(clip, analysis)
    
    return ScoreBreakdown(
        text_hook=text_hook_score,
        emotion=emotion_score,
        humor=humor_score,
        surprise=surprise_score,
        energy=energy_score,
        laughter=laughter_score,
        pause_rhythm=pause_rhythm_score,
        structure=structure_score,
        topic=topic_score,
    )


def calculate_arabic_text_hook_score(clip: Clip, analysis: Dict) -> float:
    """Calculate text hook score with Arabic enhancements."""
    base_score = clip.score if hasattr(clip, 'score') else 0.5
    
    # Boost for CTAs (but only if actually spoken)
    cta_boost = min(0.3, len(analysis['detected_ctas']) * 0.1)
    
    # Boost for definite nouns and feminine nouns
    topic_boost = sum(analysis['topic_boosts'].values()) * 0.1
    
    # Boost for hashtags (engagement signal)
    hashtag_boost = min(0.2, len(analysis['hashtags']) * 0.05)
    
    # Penalty for too many stopwords
    stopwords_penalty = min(0.2, analysis['stopwords_filtered'] * 0.02)
    
    final_score = min(1.0, base_score + cta_boost + topic_boost + hashtag_boost - stopwords_penalty)
    
    log.debug(f"Arabic text hook score: {base_score:.3f} + {cta_boost:.3f} + {topic_boost:.3f} + {hashtag_boost:.3f} - {stopwords_penalty:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_emotion_score(clip: Clip, analysis: Dict) -> float:
    """Calculate emotion score for Arabic content."""
    # Look for emotional Arabic words
    emotional_words = ['حزن', 'فرح', 'غضب', 'خوف', 'رعب', 'دهشة', 'امتنان', 'حب', 'كره', 'قلق']
    
    text_lower = clip.text.lower()
    emotion_count = sum(1 for word in emotional_words if word in text_lower)
    
    # Boost for emotional content
    emotion_boost = min(0.3, emotion_count * 0.1)
    
    # Check for emotional expressions
    emotion_expressions = ['يا لله', 'سبحان الله', 'الحمد لله', 'لا إله إلا الله']
    expression_count = sum(1 for expr in emotion_expressions if expr in clip.text)
    
    expression_boost = min(0.2, expression_count * 0.1)
    
    base_score = clip.breakdown.emotion if clip.breakdown else 0.5
    final_score = min(1.0, base_score + emotion_boost + expression_boost)
    
    log.debug(f"Arabic emotion score: {base_score:.3f} + {emotion_boost:.3f} + {expression_boost:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_humor_score(clip: Clip, analysis: Dict) -> float:
    """Calculate humor score for Arabic content."""
    # Look for humor patterns in Arabic
    humor_patterns = [
        'ضحكت', 'ضحك', 'فكاهة', 'مزح', 'نكتة', 'نكت',
        '😂', '🤣', '😄', '😃', '😁',
        'ههه', 'هههه', 'ضحكة', 'ضحكت'
    ]
    
    humor_count = sum(1 for pattern in humor_patterns if pattern in clip.text)
    
    # Boost for humor indicators
    humor_boost = min(0.3, humor_count * 0.1)
    
    base_score = clip.breakdown.humor if clip.breakdown else 0.5
    final_score = min(1.0, base_score + humor_boost)
    
    log.debug(f"Arabic humor score: {base_score:.3f} + {humor_boost:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_surprise_score(clip: Clip, analysis: Dict) -> float:
    """Calculate surprise score for Arabic content."""
    # Look for surprise indicators
    surprise_words = ['فجأة', 'مفاجأة', 'صدمني', 'لم أتوقع', 'ما كنت أتوقع', 'يا إلهي']
    
    text_lower = clip.text.lower()
    surprise_count = sum(1 for word in surprise_words if word in text_lower)
    
    # Boost for surprise elements
    surprise_boost = min(0.3, surprise_count * 0.15)
    
    base_score = clip.breakdown.surprise if clip.breakdown else 0.5
    final_score = min(1.0, base_score + surprise_boost)
    
    log.debug(f"Arabic surprise score: {base_score:.3f} + {surprise_boost:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_energy_score(clip: Clip, analysis: Dict) -> float:
    """Calculate energy score for Arabic content."""
    # Look for energy indicators
    energy_words = ['سرعة', 'حركة', 'نشاط', 'حماس', 'حماسي', 'متحمس', 'متحرك']
    
    text_lower = clip.text.lower()
    energy_count = sum(1 for word in energy_words if word in text_lower)
    
    # Boost for energy indicators
    energy_boost = min(0.3, energy_count * 0.1)
    
    # Boost for exclamation marks
    exclamation_boost = min(0.1, clip.text.count('!') * 0.05)
    
    base_score = clip.breakdown.energy if clip.breakdown else 0.5
    final_score = min(1.0, base_score + energy_boost + exclamation_boost)
    
    log.debug(f"Arabic energy score: {base_score:.3f} + {energy_boost:.3f} + {exclamation_boost:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_laughter_score(clip: Clip, analysis: Dict) -> float:
    """Calculate laughter score for Arabic content."""
    # Look for laughter indicators
    laughter_indicators = ['ضحكت', 'ضحك', 'ضحكة', 'ههه', 'هههه', '😂', '🤣', '😄']
    
    laughter_count = sum(1 for indicator in laughter_indicators if indicator in clip.text)
    
    # Boost for laughter indicators
    laughter_boost = min(0.3, laughter_count * 0.15)
    
    base_score = clip.breakdown.laughter if clip.breakdown else 0.5
    final_score = min(1.0, base_score + laughter_boost)
    
    log.debug(f"Arabic laughter score: {base_score:.3f} + {laughter_boost:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_pause_rhythm_score(clip: Clip, analysis: Dict) -> float:
    """Calculate pause rhythm score for Arabic content."""
    # Look for pause indicators
    pause_indicators = ['...', '؟', '!', '...', '...', '...', '...', '...', '...', '...']
    
    pause_count = sum(1 for indicator in pause_indicators if indicator in clip.text)
    
    # Boost for dramatic pauses
    pause_boost = min(0.2, pause_count * 0.05)
    
    base_score = clip.breakdown.pause_rhythm if clip.breakdown else 0.5
    final_score = min(1.0, base_score + pause_boost)
    
    log.debug(f"Arabic pause rhythm score: {base_score:.3f} + {pause_boost:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_structure_score(clip: Clip, analysis: Dict) -> float:
    """Calculate structure score for Arabic content."""
    # Look for structural indicators
    structure_words = ['أولا', 'ثانيا', 'ثالثا', 'رابعا', 'خامسا', 'سادسا', 'سابعا', 'ثامنا', 'تاسعا', 'عاشرا']
    
    text_lower = clip.text.lower()
    structure_count = sum(1 for word in structure_words if word in text_lower)
    
    # Boost for structured content
    structure_boost = min(0.3, structure_count * 0.1)
    
    base_score = clip.breakdown.structure if clip.breakdown else 0.5
    final_score = min(1.0, base_score + structure_boost)
    
    log.debug(f"Arabic structure score: {base_score:.3f} + {structure_boost:.3f} = {final_score:.3f}")
    
    return final_score


def calculate_arabic_topic_score(clip: Clip, analysis: Dict) -> float:
    """Calculate topic score for Arabic content with topic boosting."""
    # Use the topic boosts from Arabic analysis
    topic_boost = sum(analysis['topic_boosts'].values())
    
    base_score = clip.breakdown.topic if clip.breakdown else 0.5
    final_score = min(1.0, base_score + topic_boost)
    
    log.debug(f"Arabic topic score: {base_score:.3f} + {topic_boost:.3f} = {final_score:.3f}")
    
    return final_score


def is_arabic_text(text: str) -> bool:
    """Check if text contains significant Arabic content."""
    arabic_range = range(0x0600, 0x06FF + 1)
    arabic_chars = [char for char in text if ord(char) in arabic_range]
    
    # Consider it Arabic if more than 30% of characters are Arabic
    return len(arabic_chars) / len(text) > 0.3 if text else False


def enhance_arabic_scoring(clip: Clip) -> Clip:
    """Enhance clip scoring with Arabic-specific logic."""
    if not is_arabic_text(clip.text):
        return clip
    
    log.info(f"Applying Arabic scoring enhancements to clip {clip.id}")
    
    # Calculate enhanced scores
    enhanced_scores = score_arabic_content(clip)
    
    # Update clip with enhanced scores
    clip.breakdown = enhanced_scores
    
    # Add Arabic analysis to clip reasons
    analysis = analyze_arabic_content(clip.text)
    reasons = clip.reasons or []
    
    if analysis['detected_ctas']:
        reasons.append(f"Arabic CTAs detected: {', '.join(analysis['detected_ctas'])}")
    
    if analysis['hashtags']:
        reasons.append(f"Arabic hashtags: {', '.join(analysis['hashtags'])}")
    
    if analysis['topic_boosts']:
        reasons.append(f"Arabic topic boosts: {len(analysis['topic_boosts'])} topics enhanced")
    
    clip.reasons = reasons
    
    return clip