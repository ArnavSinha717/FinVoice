"""Per-segment language tagging for code-switching detection.

Uses langdetect (Google's language-detection, pure Python) to tag each
transcript segment with its detected language. Detects code-switching
when segments alternate between languages (e.g., Hindi ↔ English).

Stores per-segment `lang` field and computes call-level metadata:
  - primary_language: most frequent language across segments
  - has_code_switching: True if >10% of segments differ from primary
  - language_distribution: {lang: count} across all segments
"""

from loguru import logger

_HAS_LANGDETECT = False

try:
    from langdetect import detect, detect_langs, DetectorFactory
    # Make langdetect deterministic (it uses random sampling internally)
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:
    logger.warning("langdetect not installed — per-segment language tagging disabled")


# Minimum text length for reliable detection (short phrases are noisy)
_MIN_TEXT_LENGTH = 15

# Minimum langdetect probability before believing a non-fallback language.
_MIN_CONFIDENCE = 0.85


def tag_segment_languages(
    segments: list[dict],
    fallback_lang: str = "en",
) -> dict:
    """Tag each segment with detected language and compute code-switching metadata.

    Args:
        segments: Transcript segments with 'text' field
        fallback_lang: Language to use when detection fails or text is too short

    Returns:
        Dict with:
        - primary_language: str (most common language)
        - has_code_switching: bool (>10% segments differ from primary)
        - language_distribution: dict[str, int]
        - segments_tagged: int (how many segments were tagged)

    Side effect: Adds 'lang' field to each segment dict in-place.
    """
    if not _HAS_LANGDETECT:
        # Fallback: tag all segments with the WhisperX-detected language
        for seg in segments:
            seg["lang"] = fallback_lang
        return {
            "primary_language": fallback_lang,
            "has_code_switching": False,
            "language_distribution": {fallback_lang: len(segments)},
            "segments_tagged": len(segments),
        }

    lang_counts: dict[str, int] = {}
    tagged = 0

    for seg in segments:
        text = seg.get("text", "").strip()

        if len(text) < _MIN_TEXT_LENGTH:
            # Too short for reliable detection — use fallback
            seg["lang"] = fallback_lang
            lang_counts[fallback_lang] = lang_counts.get(fallback_lang, 0) + 1
            tagged += 1
            continue

        try:
            # langdetect is unreliable on short, domain-specific utterances and will
            # confidently mislabel English call fragments as Somali/French/German.
            # Only accept a non-fallback language when it is high-confidence.
            ranked = detect_langs(text)
            best = ranked[0]
            detected = _normalize_lang_code(best.lang)
            if detected != fallback_lang and best.prob < _MIN_CONFIDENCE:
                detected = fallback_lang
            seg["lang"] = detected
            lang_counts[detected] = lang_counts.get(detected, 0) + 1
        except Exception:
            seg["lang"] = fallback_lang
            lang_counts[fallback_lang] = lang_counts.get(fallback_lang, 0) + 1

        tagged += 1

    # Compute primary language (most frequent)
    if lang_counts:
        primary = max(lang_counts, key=lang_counts.get)
    else:
        primary = fallback_lang

    # Code-switching: >10% of segments in a different language than primary
    total = sum(lang_counts.values())
    non_primary = total - lang_counts.get(primary, 0)
    has_code_switching = (non_primary / total) > 0.10 if total > 0 else False

    if has_code_switching:
        logger.info(
            f"Code-switching detected: primary={primary}, "
            f"distribution={lang_counts}"
        )

    return {
        "primary_language": primary,
        "has_code_switching": has_code_switching,
        "language_distribution": lang_counts,
        "segments_tagged": tagged,
    }


def _normalize_lang_code(code: str) -> str:
    """Normalize langdetect language codes to match WhisperX codes."""
    # langdetect returns ISO 639-1 codes which mostly match WhisperX
    # But some need mapping
    mapping = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "pt-br": "pt",
        "pt-pt": "pt",
    }
    return mapping.get(code, code)
