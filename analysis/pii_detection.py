"""PII Detection & Masking — Microsoft Presidio + Indian Financial Recognizers.

Uses Presidio (enterprise-grade PII engine) with custom recognizers for:
- Aadhaar numbers (12-digit with Verhoeff checksum)
- Indian PAN (ABCDE1234F)
- Indian phone numbers (+91 / 10-digit)
- UPI IDs (name@bank)
- IFSC codes (ABCD0123456)
- Bank account numbers (9-18 digits in financial context)

Presidio uses spaCy NER + regex + checksum validation internally.
"""

import re
from loguru import logger
from pydantic import BaseModel, Field
from typing import Optional

from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# ── CUSTOM INDIAN FINANCIAL RECOGNIZERS ──

# Verhoeff checksum tables for Aadhaar validation
_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8],
]
_VERHOEFF_INV = [0,4,3,2,1,5,6,7,8,9]


def _verhoeff_checksum(number: str) -> bool:
    """Validate Aadhaar number using Verhoeff algorithm."""
    try:
        c = 0
        for i, digit in enumerate(reversed(number)):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
        return c == 0
    except (ValueError, IndexError):
        return False


def _build_aadhaar_recognizer() -> PatternRecognizer:
    """Aadhaar: 12 digits, optionally spaced as XXXX XXXX XXXX, with Verhoeff checksum."""
    return PatternRecognizer(
        supported_entity="IN_AADHAAR",
        name="Indian Aadhaar Recognizer",
        patterns=[
            Pattern(
                name="aadhaar_spaced",
                regex=r"(?<![\d+])([2-9]\d{3}\s\d{4}\s\d{4})(?!\d)",
                score=0.7,
            ),
            # Dictated digit-by-digit. Whisper renders a spoken Aadhaar this way
            # far more often than as a clean 12-digit string, and the evaluation
            # harness measured 0.50 recall on ASR-rendered identifiers before this.
            Pattern(
                name="aadhaar_dictated",
                regex=r"(?<![\d+])([2-9](?:\s\d){11})(?!\s?\d)",
                score=0.6,
            ),
            # A 12-digit run that is NOT part of a longer number and is not preceded
            # by "+" — otherwise an E.164 Indian mobile (+91 + 10 digits) is exactly
            # 12 digits and gets read as an Aadhaar. UIDAI also never issues a number
            # starting with 0 or 1, so require a leading 2-9.
            Pattern(
                name="aadhaar_continuous",
                regex=r"(?<![\d+])([2-9]\d{11})(?!\d)",
                score=0.5,  # Lower score — needs context or checksum to confirm
            ),
        ],
        context=["aadhaar", "aadhar", "uid", "uidai", "identity", "id number", "verification"],
    )


def _build_pan_recognizer() -> PatternRecognizer:
    """Indian PAN: ABCDE1234F format."""
    return PatternRecognizer(
        supported_entity="IN_PAN",
        name="Indian PAN Recognizer",
        patterns=[
            Pattern(
                name="pan_standard",
                regex=r"\b[A-Z]{5}\d{4}[A-Z]\b",
                score=0.85,
            ),
            # Dictated letter-by-letter / digit-by-digit, e.g. "A B C D E 1 2 3 4 F".
            Pattern(
                name="pan_dictated",
                regex=r"(?<![A-Za-z0-9])((?:[A-Za-z]\s){5}(?:\d\s){4}[A-Za-z])(?![A-Za-z0-9])",
                score=0.6,
            ),
        ],
        context=["pan", "permanent account", "tax", "income tax", "PAN card"],
    )


def _build_indian_phone_recognizer() -> PatternRecognizer:
    """Indian phone: +91 XXXXX XXXXX or 10-digit starting with 6-9."""
    return PatternRecognizer(
        supported_entity="IN_PHONE",
        name="Indian Phone Recognizer",
        patterns=[
            Pattern(
                name="phone_with_country",
                regex=r"\+91[\s-]?[6-9]\d{4}[\s-]?\d{5}",
                score=0.85,
            ),
            Pattern(
                name="phone_10digit",
                regex=r"\b[6-9]\d{9}\b",
                score=0.6,
            ),
        ],
        context=["phone", "mobile", "call", "number", "contact", "whatsapp"],
    )


def _build_upi_recognizer() -> PatternRecognizer:
    """UPI ID: name@bankhandle."""
    return PatternRecognizer(
        supported_entity="IN_UPI_ID",
        name="Indian UPI ID Recognizer",
        patterns=[
            # Known handles — high confidence.
            Pattern(
                name="upi_id_known_handle",
                regex=r"\b[\w.-]+@(?:ybl|okhdfcbank|okicici|oksbi|okaxis|okbizaxis|paytm|"
                      r"apl|ibl|axl|axisbank|upi|phonepe|gpay|hdfcbank|icici|sbi|yesbank|"
                      r"kotak|federal|idfcbank|indus|barodampay|cnrb|airtel|freecharge|"
                      r"abfspay|jupiteraxis|naviaxis|fam|slc|rapl|waaxis|waicici|wasbi)\b",
                score=0.9,
            ),
            # Any handle of the right shape. The previous allowlist silently missed
            # real VPAs (e.g. @okaxis), and new PSP handles appear regularly, so match
            # the format and lean on Presidio's context words for confidence.
            Pattern(
                name="upi_id_generic",
                regex=r"\b[\w.-]{2,64}@[a-z]{2,30}\b",
                score=0.5,
            ),
        ],
        context=["upi", "payment", "pay", "transfer", "gpay", "phonepe", "paytm"],
    )


def _build_ifsc_recognizer() -> PatternRecognizer:
    """IFSC code: 4 letters + 0 + 6 digits."""
    return PatternRecognizer(
        supported_entity="IN_IFSC",
        name="Indian IFSC Recognizer",
        patterns=[
            Pattern(
                name="ifsc_code",
                regex=r"\b[A-Z]{4}0\d{6}\b",
                score=0.9,
            ),
        ],
        context=["ifsc", "branch", "bank", "transfer", "neft", "rtgs", "imps"],
    )


def _build_bank_account_recognizer() -> PatternRecognizer:
    """Bank account number: 9-18 digits in financial context."""
    return PatternRecognizer(
        supported_entity="IN_BANK_ACCOUNT",
        name="Indian Bank Account Recognizer",
        patterns=[
            Pattern(
                name="account_number",
                regex=r"\b\d{9,18}\b",
                score=0.3,  # Low base score — needs context
            ),
        ],
        context=[
            "account", "a/c", "account number", "savings", "current",
            "bank", "deposit", "credit", "debit",
        ],
    )


def _build_credit_card_recognizer() -> PatternRecognizer:
    """Credit/debit card: 16 digits, optionally grouped."""
    return PatternRecognizer(
        supported_entity="CREDIT_CARD",
        name="Card Number Recognizer",
        patterns=[
            Pattern(
                name="card_spaced",
                regex=r"\b(\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4})\b",
                score=0.8,
            ),
            Pattern(
                name="card_continuous",
                regex=r"\b(\d{16})\b",
                score=0.4,
            ),
        ],
        context=["card", "credit", "debit", "visa", "mastercard", "rupay", "cvv", "expiry"],
    )


# ── PII RESULT MODEL ──

class PIIEntity(BaseModel):
    """A detected PII entity."""
    entity_type: str = Field(description="PII type: IN_AADHAAR, IN_PAN, IN_PHONE, EMAIL_ADDRESS, PERSON, etc.")
    text: str = Field(description="The detected PII text")
    masked_text: str = Field(description="Masked version: ABCDE1234F → XXXXX****X")
    start: int = Field(description="Character start position in segment text")
    end: int = Field(description="Character end position in segment text")
    score: float = Field(ge=0, le=1, description="Detection confidence")
    segment_id: int


# ── ENGINE SETUP ──

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_analyzer() -> AnalyzerEngine:
    """Initialize Presidio analyzer with Indian financial recognizers."""
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    logger.info("Initializing Presidio PII analyzer with Indian financial recognizers...")

    _analyzer = AnalyzerEngine()

    # Register custom Indian recognizers
    _analyzer.registry.add_recognizer(_build_aadhaar_recognizer())
    _analyzer.registry.add_recognizer(_build_pan_recognizer())
    _analyzer.registry.add_recognizer(_build_indian_phone_recognizer())
    _analyzer.registry.add_recognizer(_build_upi_recognizer())
    _analyzer.registry.add_recognizer(_build_ifsc_recognizer())
    _analyzer.registry.add_recognizer(_build_bank_account_recognizer())
    _analyzer.registry.add_recognizer(_build_credit_card_recognizer())

    logger.info("Presidio analyzer ready with 7 custom Indian recognizers")
    return _analyzer


def _get_anonymizer() -> AnonymizerEngine:
    """Initialize Presidio anonymizer."""
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


# ── PUBLIC API ──

def detect_pii(segments: list, score_threshold: float = 0.5, lang: str = "en") -> list[PIIEntity]:
    """Detect PII entities across all transcript segments.

    Uses Presidio with Indian financial recognizers. Detects:
    - Standard PII: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME
    - Indian financial: IN_AADHAAR, IN_PAN, IN_PHONE, IN_UPI_ID, IN_IFSC,
      IN_BANK_ACCOUNT, CREDIT_CARD

    Custom regex recognizers (Aadhaar, PAN, phone, etc.) are format-based and
    work regardless of language. Presidio's NER-based detection uses English
    spaCy model (best available — no hi/ta Presidio support).

    Args:
        segments: Transcript segments with 'text' field
        score_threshold: Minimum confidence to report (default 0.5)
        lang: Detected language (for future multilingual enhancements)

    Returns:
        List of PIIEntity objects with masked versions
    """
    analyzer = _get_analyzer()
    pii_entities = []

    # All entity types we care about
    entities_to_detect = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
        "DATE_TIME", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
        # Custom Indian
        "IN_AADHAAR", "IN_PAN", "IN_PHONE", "IN_UPI_ID",
        "IN_IFSC", "IN_BANK_ACCOUNT",
    ]

    # Per-entity-type score thresholds. DATE_TIME is raised to 0.7 because
    # Presidio's default recognizer is too aggressive on financial calls —
    # catches "2020", "third quarter", "last year" etc. as PII dates.
    entity_score_overrides = {
        "DATE_TIME": max(score_threshold, 0.7),
    }

    for _pos, seg in enumerate(segments):
        # Prefer the segment's own id; fall back to position. Attributing PII to
        # a positional index silently mislabels callers that pass filtered or
        # re-indexed segments.
        seg_idx = seg.get("segment_id", seg.get("id", _pos))
        text = seg.get("text", "")
        if not text.strip():
            continue

        results = analyzer.analyze(
            text=text,
            entities=entities_to_detect,
            language="en",
            score_threshold=score_threshold,
        )

        for r in results:
            # Apply per-entity-type score threshold
            min_score = entity_score_overrides.get(r.entity_type, score_threshold)
            if r.score < min_score:
                continue

            detected_text = text[r.start:r.end]

            # Validate Aadhaar with Verhoeff if detected
            if r.entity_type == "IN_AADHAAR":
                digits = re.sub(r"\s", "", detected_text)
                if len(digits) == 12 and not _verhoeff_checksum(digits):
                    continue  # Failed checksum — not a real Aadhaar

            pii_entities.append(PIIEntity(
                entity_type=r.entity_type,
                text=detected_text,
                masked_text=_mask_pii(detected_text, r.entity_type),
                start=r.start,
                end=r.end,
                score=round(r.score, 3),
                segment_id=seg_idx,
            ))

    # For non-English: supplement with multilingual NER for PERSON/LOCATION
    # Presidio's spaCy NER is English-only — Hindi/Tamil names go undetected.
    if lang != "en":
        multilingual_pii = _detect_multilingual_ner(segments, lang)
        pii_entities.extend(multilingual_pii)

    # Deduplicate by (entity_type, text, segment_id)
    seen = set()
    unique = []
    for e in pii_entities:
        key = (e.entity_type, e.text, e.segment_id)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    logger.info(f"PII detection: {len(unique)} entities found across {len(segments)} segments")
    return unique


def mask_transcript(segments: list, pii_entities: list[PIIEntity]) -> list[dict]:
    """Return a copy of segments with PII masked.

    Useful for creating shareable/export versions of transcripts.
    """
    # Group PII by segment
    pii_by_seg = {}
    for e in pii_entities:
        if e.segment_id not in pii_by_seg:
            pii_by_seg[e.segment_id] = []
        pii_by_seg[e.segment_id].append(e)

    masked_segments = []
    for i, seg in enumerate(segments):
        new_seg = dict(seg)
        text = seg.get("text", "")

        if i in pii_by_seg:
            # Sort by start position descending to replace from end
            entities = sorted(pii_by_seg[i], key=lambda e: e.start, reverse=True)
            for e in entities:
                text = text[:e.start] + e.masked_text + text[e.end:]
            new_seg["text"] = text
            new_seg["pii_masked"] = True

        masked_segments.append(new_seg)

    return masked_segments


def _mask_pii(text: str, entity_type: str) -> str:
    """Generate appropriate mask for different PII types."""
    if entity_type == "IN_AADHAAR":
        digits = re.sub(r"\s", "", text)
        return f"XXXX XXXX {digits[-4:]}" if len(digits) >= 4 else "XXXX XXXX XXXX"
    elif entity_type == "IN_PAN":
        return f"XXXXX{text[5:9]}X" if len(text) == 10 else "XXXXXXXXXX"
    elif entity_type in ("PHONE_NUMBER", "IN_PHONE"):
        digits = re.sub(r"\D", "", text)
        return f"XXXXXX{digits[-4:]}" if len(digits) >= 4 else "XXXXXXXXXX"
    elif entity_type == "EMAIL_ADDRESS":
        parts = text.split("@")
        if len(parts) == 2:
            return f"{parts[0][0]}***@{parts[1]}"
        return "***@***"
    elif entity_type == "CREDIT_CARD":
        digits = re.sub(r"\D", "", text)
        return f"XXXX XXXX XXXX {digits[-4:]}" if len(digits) >= 4 else "XXXX XXXX XXXX XXXX"
    elif entity_type == "IN_UPI_ID":
        return "****@****"
    elif entity_type == "IN_IFSC":
        return "XXXX0XXXXXX"
    elif entity_type == "IN_BANK_ACCOUNT":
        return f"XXXXXX{text[-4:]}" if len(text) >= 4 else "XXXXXXXXXX"
    elif entity_type == "PERSON":
        return "[PERSON]"
    elif entity_type == "LOCATION":
        return "[LOCATION]"
    else:
        return f"[{entity_type}]"


# ── MULTILINGUAL NER (Hindi/Tamil PERSON + LOCATION) ──

_multilingual_ner = None


def _get_multilingual_ner():
    """Load multilingual NER pipeline for Hindi/Tamil names and locations.

    Uses XLM-RoBERTa-based NER (ai4bharat/IndicNER or xlm-roberta-large-finetuned-conll03).
    Falls back gracefully if not available.
    """
    global _multilingual_ner
    if _multilingual_ner is not None:
        return _multilingual_ner

    try:
        from transformers import pipeline as hf_pipeline

        # Try AI4Bharat IndicNER first (best for Indian languages)
        try:
            _multilingual_ner = hf_pipeline(
                "token-classification",
                model="ai4bharat/IndicNER",
                device=-1,
                aggregation_strategy="simple",
                truncation=True,
                max_length=512,
            )
            logger.info("IndicNER loaded for Hindi/Tamil PII (PERSON/LOCATION)")
            return _multilingual_ner
        except Exception:
            pass

        # Fallback: XLM-R multilingual NER
        _multilingual_ner = hf_pipeline(
            "token-classification",
            model="Davlan/xlm-roberta-large-finetuned-conll03-english",
            device=-1,
            aggregation_strategy="simple",
            truncation=True,
            max_length=512,
        )
        logger.info("XLM-R NER loaded as fallback for multilingual PII")
        return _multilingual_ner

    except Exception as e:
        logger.warning(f"Multilingual NER not available for PII: {e}")
        _multilingual_ner = False  # Sentinel to avoid retrying
        return None


def _detect_multilingual_ner(segments: list, lang: str) -> list[PIIEntity]:
    """Run multilingual NER to detect PERSON and LOCATION in non-English text.

    Presidio's spaCy NER is English-only. This supplements it for Hindi/Tamil
    names and locations using IndicNER or XLM-R.
    """
    ner = _get_multilingual_ner()
    if not ner:
        return []

    entities = []
    # NER label mapping (different models use different labels)
    person_labels = {"PER", "PERSON", "B-PER", "I-PER"}
    location_labels = {"LOC", "LOCATION", "B-LOC", "I-LOC", "GPE"}

    for _pos, seg in enumerate(segments):
        # Prefer the segment's own id; fall back to position. Attributing PII to
        # a positional index silently mislabels callers that pass filtered or
        # re-indexed segments.
        seg_idx = seg.get("segment_id", seg.get("id", _pos))
        text = seg.get("text", "").strip()
        if not text or len(text) < 3:
            continue

        try:
            results = ner(text[:512])
            for ent in results:
                label = ent.get("entity_group", ent.get("entity", "")).upper()
                score = ent.get("score", 0.0)
                word = ent.get("word", "").strip()

                if score < 0.5 or len(word) < 2:
                    continue

                if label in person_labels or "PER" in label:
                    entity_type = "PERSON"
                elif label in location_labels or "LOC" in label:
                    entity_type = "LOCATION"
                else:
                    continue

                # Clean up tokenizer artifacts (## prefixes, spaces)
                word = word.replace("##", "").strip()
                if not word:
                    continue

                start = ent.get("start", 0)
                end = ent.get("end", start + len(word))

                entities.append(PIIEntity(
                    entity_type=entity_type,
                    text=word,
                    masked_text=_mask_pii(word, entity_type),
                    start=start,
                    end=end,
                    score=round(score, 3),
                    segment_id=seg_idx,
                ))
        except Exception:
            continue

    if entities:
        logger.info(f"Multilingual NER ({lang}): {len(entities)} PERSON/LOCATION entities")

    return entities
