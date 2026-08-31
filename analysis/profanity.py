"""Profanity & Toxicity Detection — Detoxify multilingual classifier.

Uses Unitary AI's Detoxify model (trained on Jigsaw's 1.8M multilingual dataset)
for robust toxicity detection across 6 dimensions:
  - toxicity, severe_toxicity, obscene, identity_attack, insult, threat

Handles Hindi, English, and other languages out of the box.
Runs on CPU (~50ms per batch of 32 sentences).
"""

from loguru import logger
from pydantic import BaseModel, Field

try:
    from detoxify import Detoxify
    _model = None

    def _get_model():
        global _model
        if _model is None:
            logger.info("Loading Detoxify multilingual toxicity model on CPU...")
            _model = Detoxify("multilingual", device="cpu")
            logger.info("Detoxify model loaded")
        return _model

    HAS_DETOXIFY = True
except ImportError:
    HAS_DETOXIFY = False
    logger.warning("detoxify not installed — profanity detection disabled")


# Toxicity thresholds
TOXICITY_THRESHOLD = 0.7       # Flag as toxic
THREAT_THRESHOLD = 0.5         # Lower threshold for threats (serious)
SEVERE_THRESHOLD = 0.3         # Lower for severe toxicity


class ToxicityFlag(BaseModel):
    """A detected toxicity/profanity instance."""
    segment_id: int
    speaker: str = Field(description="Who said it")
    text: str = Field(description="The flagged text")
    toxicity_score: float = Field(ge=0, le=1)
    categories: list[str] = Field(description="Which categories triggered: toxicity, threat, insult, etc.")
    severity: str = Field(description="'low', 'medium', 'high', 'critical'")
    is_agent: bool = Field(description="True if said by agent — more serious for compliance")


def detect_profanity(segments: list) -> list[ToxicityFlag]:
    """Detect toxic, abusive, threatening, or profane language in transcript.

    Uses Detoxify multilingual model (Jigsaw-trained) for 6-dimensional
    toxicity classification. Much more robust than keyword matching.

    Args:
        segments: Transcript segments with 'text' and 'speaker' fields

    Returns:
        List of ToxicityFlag objects for flagged segments
    """
    if not HAS_DETOXIFY:
        from pipeline.degradations import report
        report("toxicity", "detoxify not installed",
               "toxicity_flags empty; agent_toxicity review reason never fires")
        return []

    model = _get_model()
    flags = []

    # Batch all segment texts
    texts = []
    indices = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if text and len(text) > 2:
            texts.append(text)
            indices.append(i)

    if not texts:
        return []

    # Run batch prediction (truncate texts to avoid embedding OOB errors)
    texts = [t[:512] for t in texts]
    try:
        results = model.predict(texts)
    except Exception as e:
        logger.warning(f"Detoxify batch prediction failed ({e}), trying individually")
        # Fallback: predict one at a time, skip failures
        keys = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]
        results = {k: [] for k in keys}
        skip_indices = set()
        for idx, text in enumerate(texts):
            try:
                r = model.predict([text])
                for k in keys:
                    results[k].append(r[k][0])
            except Exception:
                skip_indices.add(idx)
                for k in keys:
                    results[k].append(0.0)
        if skip_indices:
            logger.warning(f"Detoxify: {len(skip_indices)} segments skipped due to errors")

    for batch_idx, seg_idx in enumerate(indices):
        seg = segments[seg_idx]
        speaker = seg.get("speaker", "unknown").lower()

        scores = {
            "toxicity": results["toxicity"][batch_idx],
            "severe_toxicity": results["severe_toxicity"][batch_idx],
            "obscene": results["obscene"][batch_idx],
            "identity_attack": results["identity_attack"][batch_idx],
            "insult": results["insult"][batch_idx],
            "threat": results["threat"][batch_idx],
            "sexual_explicit": results["sexual_explicit"][batch_idx],
        }

        # Check which categories triggered
        triggered = []
        if scores["toxicity"] >= TOXICITY_THRESHOLD:
            triggered.append("toxicity")
        if scores["threat"] >= THREAT_THRESHOLD:
            triggered.append("threat")
        if scores["severe_toxicity"] >= SEVERE_THRESHOLD:
            triggered.append("severe_toxicity")
        if scores["obscene"] >= TOXICITY_THRESHOLD:
            triggered.append("obscene")
        if scores["insult"] >= TOXICITY_THRESHOLD:
            triggered.append("insult")
        if scores["identity_attack"] >= TOXICITY_THRESHOLD:
            triggered.append("identity_attack")
        if scores["sexual_explicit"] >= TOXICITY_THRESHOLD:
            triggered.append("sexual_explicit")

        if not triggered:
            continue

        # Determine severity
        max_score = scores["toxicity"]
        is_agent = speaker in ("agent", "speaker_00")

        if scores["severe_toxicity"] >= 0.5 or scores["threat"] >= 0.8:
            severity = "critical"
        elif max_score >= 0.9 or (is_agent and max_score >= 0.7):
            severity = "high"
        elif max_score >= 0.7:
            severity = "medium"
        else:
            severity = "low"

        # Agent toxicity is always escalated
        if is_agent and severity == "medium":
            severity = "high"

        flags.append(ToxicityFlag(
            segment_id=seg_idx,
            speaker=speaker,
            text=seg.get("text", ""),
            toxicity_score=round(max_score, 3),
            categories=triggered,
            severity=severity,
            is_agent=is_agent,
        ))

    # ── MuRIL abuse detection (Hindi/Tamil/Indian languages) ──
    # Detoxify covers English well but misses Hindi code-mixed abuse.
    # MuRIL handles 10 Indian languages natively — shared model with fraud module.
    try:
        from analysis.indic_abuse_detector import detect_abuse_batch
        muril_flagged = detect_abuse_batch(texts, threshold=0.6)
        flagged_seg_ids = {f.segment_id for f in flags}  # Already flagged by Detoxify

        for item in muril_flagged:
            batch_idx = item["index"]
            if batch_idx >= len(indices):
                continue
            seg_idx = indices[batch_idx]

            # Skip if Detoxify already flagged this segment
            if seg_idx in flagged_seg_ids:
                continue

            seg = segments[seg_idx]
            speaker = seg.get("speaker", "unknown").lower()
            is_agent = speaker in ("agent", "speaker_00", "speaker a")

            severity = "high" if is_agent else "medium"
            if item["score"] >= 0.9:
                severity = "critical" if is_agent else "high"

            flags.append(ToxicityFlag(
                segment_id=seg_idx,
                speaker=speaker,
                text=seg.get("text", ""),
                toxicity_score=round(item["score"], 3),
                categories=["muril_abuse", item["label"]],
                severity=severity,
                is_agent=is_agent,
            ))
    except ImportError:
        pass  # MuRIL not available — Detoxify-only mode

    if flags:
        agent_flags = sum(1 for f in flags if f.is_agent)
        logger.info(
            f"Profanity detection: {len(flags)} toxic segments "
            f"({agent_flags} by agent — compliance risk)"
        )
    else:
        logger.info("Profanity detection: no toxic content found")

    return flags
