"""Zero-shot Scam Classification — mDeBERTa multilingual NLI.

Uses `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` for
zero-shot scam/fraud classification across 100+ languages including Hindi
and Tamil. No training data needed — define labels and classify.

Custom labels for financial call context:
  - scam_call: caller is running a scam
  - social_engineering: caller is manipulating the victim
  - credential_phishing: caller is extracting passwords/OTPs/PINs
  - identity_theft: caller is collecting personal information for fraud
  - legitimate_banking: normal banking interaction

Runs on CPU (~560MB). ~50ms per sentence.
"""

from loguru import logger

_pipeline = None
HAS_ZERO_SHOT = False

_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Labels for financial call scam detection
SCAM_LABELS = [
    "This is a scam call",
    "This is social engineering or manipulation",
    "This is credential phishing for passwords or OTPs",
    "This is identity theft collecting personal information",
    "This is a legitimate banking interaction",
]

# Map hypothesis labels back to signal types
_LABEL_TO_TYPE = {
    "This is a scam call": "scam_call",
    "This is social engineering or manipulation": "social_engineering",
    "This is credential phishing for passwords or OTPs": "credential_phishing",
    "This is identity theft collecting personal information": "identity_theft",
    "This is a legitimate banking interaction": "legitimate",
}


def _get_pipeline():
    """Load zero-shot classification pipeline (lazy, cached)."""
    global _pipeline, HAS_ZERO_SHOT

    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline as hf_pipeline

        logger.info(f"Loading zero-shot scam classifier: {_MODEL_NAME}...")
        _pipeline = hf_pipeline(
            "zero-shot-classification",
            model=_MODEL_NAME,
            device=-1,  # CPU
        )
        HAS_ZERO_SHOT = True
        logger.info("Zero-shot scam classifier loaded (100+ languages, CPU)")
        return _pipeline
    except Exception as e:
        logger.warning(f"Zero-shot scam classifier not available: {e}")
        HAS_ZERO_SHOT = False
        from pipeline.degradations import report
        report("zero_shot_scam", f"mDeBERTa NLI model unavailable: {e}",
               "scam_* fraud signals unavailable")
        return None


def classify_scam_batch(
    texts: list[str],
    threshold: float = 0.7,
) -> list[dict]:
    """Classify texts for scam/fraud content using zero-shot NLI.

    Args:
        texts: List of text strings (typically transcript segments)
        threshold: Minimum score to flag as scam-related (default 0.7)

    Returns:
        List of dicts for flagged texts:
        [{"index": 0, "scam_type": "credential_phishing", "score": 0.85}, ...]
    """
    pipe = _get_pipeline()
    if pipe is None:
        return []

    if not texts:
        return []

    flagged = []

    # Process in smaller batches to manage memory
    BATCH = 16
    for batch_start in range(0, len(texts), BATCH):
        batch_texts = texts[batch_start:batch_start + BATCH]
        # Truncate
        batch_texts = [t[:512] if t else "" for t in batch_texts]

        try:
            results = pipe(
                batch_texts,
                candidate_labels=SCAM_LABELS,
                multi_label=False,
            )

            # pipe returns a single dict if input is a single string
            if isinstance(results, dict):
                results = [results]

            for i, result in enumerate(results):
                top_label = result["labels"][0]
                top_score = result["scores"][0]
                scam_type = _LABEL_TO_TYPE.get(top_label, "unknown")

                # Skip if classified as legitimate
                if scam_type == "legitimate":
                    continue

                if top_score >= threshold:
                    flagged.append({
                        "index": batch_start + i,
                        "scam_type": scam_type,
                        "score": round(top_score, 3),
                        "label": top_label,
                    })

        except Exception as e:
            if batch_start == 0:
                logger.warning(f"Zero-shot scam classification failed: {e}")
            continue

    return flagged


def is_available() -> bool:
    """Check if zero-shot classifier can be loaded."""
    return _get_pipeline() is not None
