"""Stage 3: Financial Transcription — WhisperX with persistent GPU caching.

GPU lifecycle:
  Idle:         WhisperX loaded (~3GB), ready for instant transcription
  Call arrives:  Transcribe immediately (no load delay!)
  After transcr: Unload WhisperX → emotion2vec → Qwen 2.5:3b
  After LLM:    Reload WhisperX → back to idle, ready for next call

First load takes ~130s (model + pyannote VAD from disk). Subsequent reloads
take ~15-20s (weights cached in system RAM by the OS page cache).
"""

import re
import json
import torch
import threading
from pathlib import Path
from loguru import logger

# PyTorch 2.6+ changed weights_only default to True in torch.load().
# pyannote/speechbrain checkpoints use omegaconf + typing objects that aren't
# in the safe list, and lightning_fabric passes weights_only=None (which
# PyTorch 2.8 treats as True). Since these are trusted HuggingFace models,
# we patch torch.load to treat None as False.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for all checkpoint loads.
    # lightning_fabric passes weights_only=True explicitly, and PyTorch 2.8
    # treats None as True — both break pyannote/speechbrain/FunASR checkpoints
    # that contain omegaconf objects not in the safe list.
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch 2: load_state_dict() silently drops weights when model uses meta tensors
# PyTorch 2.8 + speechbrain creates models on "meta" device for lazy init.
# Without assign=True, LSTM weights become no-ops → Silero VAD broken → 0 segments.
_original_load_state_dict = torch.nn.Module.load_state_dict
def _patched_load_state_dict(self, state_dict, *args, **kwargs):
    kwargs.setdefault("assign", True)
    return _original_load_state_dict(self, state_dict, *args, **kwargs)
torch.nn.Module.load_state_dict = _patched_load_state_dict


# ── Persistent WhisperX model cache ──
_whisperx_model = None
_whisperx_lock = threading.Lock()


def preload_whisperx():
    """Load WhisperX into GPU and keep it resident for instant transcription.

    Call this at server startup. First load ~130s, but then the model stays
    in VRAM ready for immediate use. Thread-safe.
    """
    global _whisperx_model
    with _whisperx_lock:
        if _whisperx_model is not None:
            logger.info("WhisperX already loaded — skipping preload")
            return
        import whisperx
        logger.info("Preloading WhisperX (large-v3-turbo, INT8) — first load ~130s...")
        _whisperx_model = whisperx.load_model(
            whisper_arch="large-v3-turbo", device="cuda", compute_type="int8",
        )
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"WhisperX preloaded and resident ({vram_mb:.0f} MB VRAM)")


def unload_whisperx():
    """Unload WhisperX from GPU to free VRAM for other models."""
    global _whisperx_model
    with _whisperx_lock:
        if _whisperx_model is None:
            return
        del _whisperx_model
        _whisperx_model = None
        torch.cuda.empty_cache()
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"WhisperX unloaded ({vram_mb:.0f} MB VRAM remaining)")


def reload_whisperx():
    """Reload WhisperX after other GPU models are done.

    Faster than first load (~15-20s) because model weights are still
    in OS page cache from the previous load.
    """
    global _whisperx_model
    with _whisperx_lock:
        if _whisperx_model is not None:
            logger.info("WhisperX already loaded — skipping reload")
            return
        import whisperx
        logger.info("Reloading WhisperX into GPU (fast — OS page cache)...")
        _whisperx_model = whisperx.load_model(
            whisper_arch="large-v3-turbo", device="cuda", compute_type="int8",
        )
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"WhisperX reloaded ({vram_mb:.0f} MB VRAM)")


def is_whisperx_loaded() -> bool:
    """Check if WhisperX is currently loaded in GPU."""
    return _whisperx_model is not None


# Financial term correction dictionary — data-driven, not guessed.
# Run scripts/discover_corrections.py on Earnings-21 to build the full version.
# These are seed corrections; the discovery script will expand this.
_ORG_ALIASES: dict = {}

FINANCIAL_CORRECTIONS = {
    "emmy": "EMI", "emi": "EMI",
    "kayak": "KYC", "kayc": "KYC",
    "civil": "CIBIL", "sibyl": "CIBIL", "sybil": "CIBIL",
    "hdfc": "HDFC", "icici": "ICICI", "sbi": "SBI",
    "nach": "NACH", "natch": "NACH",
    "upi": "UPI", "gst": "GST", "pan": "PAN", "tan": "TAN",
    "rbi": "RBI", "sebi": "SEBI", "irdai": "IRDAI",
    "nbfc": "NBFC", "npa": "NPA",
    "crore": "crore", "lakh": "lakh",
    "demat": "demat", "d-mat": "demat",
}

# Amount normalization patterns
AMOUNT_PATTERNS = [
    (r"(\d+)\s*thousand", lambda m: str(int(m.group(1)) * 1000)),
    (r"(\d+)\s*lakh", lambda m: str(int(m.group(1)) * 100000)),
    (r"(\d+)\s*crore", lambda m: str(int(m.group(1)) * 10000000)),
    (r"(\d+)\s*k\b", lambda m: str(int(m.group(1)) * 1000)),
]


def load_correction_dictionary(path: str = "data/models/financial_corrections.json") -> dict:
    """Load data-driven corrections if available, otherwise use seed dict."""
    global _ORG_ALIASES
    _ORG_ALIASES = _load_org_aliases()
    corrections = dict(FINANCIAL_CORRECTIONS)
    if Path(path).exists():
        with open(path) as f:
            discovered = json.load(f)
        corrections.update(discovered)
        logger.info(f"Loaded {len(discovered)} discovered corrections from {path}")
    return corrections


def transcribe_audio(
    wav_path: str,
    batch_size: int = 4,
    language: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Transcribe audio with WhisperX, apply financial corrections, flag low confidence.

    Uses the persistent cached model if available (preloaded at server startup).
    Falls back to loading a fresh model if cache is empty.

    Args:
        wav_path: Path to normalized 16kHz mono WAV
        batch_size: GPU batch size (use 4 for 6GB VRAM, not 16)
        language: Language code or None for auto-detection
        hf_token: HuggingFace token for pyannote diarization models

    Returns:
        dict with segments, low_confidence_segments, overall_confidence, language
    """
    global _whisperx_model
    import whisperx

    # Use persistent cached model if available, otherwise load fresh
    if _whisperx_model is not None:
        model = _whisperx_model
        logger.info("Using cached WhisperX model (instant)")
    else:
        logger.info("Loading WhisperX (large-v3-turbo, INT8) — no cached model")
        load_kwargs = {
            "whisper_arch": "large-v3-turbo",
            "device": "cuda",
            "compute_type": "int8",
        }
        if language:
            load_kwargs["language"] = language
        model = whisperx.load_model(**load_kwargs)
        _whisperx_model = model

    # NOTE: Do NOT restore Module.to yet — alignment and diarization also load
    # models that hit the same meta tensor issue. Restored after all GPU work.

    # Transcribe
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Capture detected language
    detected_language = result.get("language", language or "en")
    logger.info(f"Transcribed {len(result['segments'])} segments (language={detected_language})")

    # Retry with Hindi if Whisper produced only "foreign" tokens or nothing —
    # this happens when Whisper mis-detects Hindi/code-switched audio as English
    if not language:  # only retry if language wasn't explicitly set
        seg_texts = [s.get("text", "").strip().lower() for s in result.get("segments", [])]
        # Check if all segments contain only the word "foreign" (repeated any number of times)
        # WhisperX outputs "foreign" when it can't transcribe detected non-English audio
        all_foreign = bool(seg_texts) and all(
            set(t.split()) <= {"foreign", ""} or t == "" for t in seg_texts
        )
        no_segments = len(result.get("segments", [])) == 0
        logger.debug(f"Foreign check — texts={seg_texts}, all_foreign={all_foreign}, no_segments={no_segments}")
        if all_foreign or no_segments:
            logger.warning("All segments are 'foreign' — retrying with language='hi'")
            # Need a language-specific model for Hindi — unload cached, load fresh
            unload_whisperx()
            model = whisperx.load_model(
                whisper_arch="large-v3-turbo", device="cuda",
                compute_type="int8", language="hi",
            )
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = "hi"
            # Don't cache the Hindi model — it's language-specific
            del model
            torch.cuda.empty_cache()
            logger.info(f"Hindi retry: {len(result['segments'])} segments")

    # P2 FIX: Save per-segment confidence from avg_logprob BEFORE alignment
    # (alignment and diarization may discard word-level scores)
    _saved_segment_conf = {}
    _saved_word_scores = {}
    for seg in result.get("segments", []):
        logprob = seg.get("avg_logprob", -0.5)
        # Map avg_logprob (-1.0=low, 0.0=perfect) to 0-1 confidence
        conf = max(0.0, min(1.0, 1.0 + logprob))
        _saved_segment_conf[round(seg.get("start", 0), 1)] = conf
        for w in seg.get("words", []):
            score = w.get("score", w.get("conf"))
            if score is not None:
                key = (w.get("word", "").strip().lower(), round(w.get("start", 0), 2))
                _saved_word_scores[key] = score

    # Word-level alignment
    alignment_ok = False
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language, device="cuda"
        )
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, device="cuda"
        )
        del align_model
        alignment_ok = True
    except Exception as e:
        logger.warning(f"Word alignment failed (continuing without): {e}")
        # Create synthetic word entries so diarization can still assign speakers.
        # Distribute words evenly across the segment's time range.
        for seg in result.get("segments", []):
            if seg.get("words"):
                continue
            text = seg.get("text", "").strip()
            words = text.split()
            if not words:
                continue
            start = seg.get("start", 0)
            end = seg.get("end", start + 1)
            duration = end - start
            word_dur = duration / len(words)
            seg["words"] = [
                {"word": w, "start": round(start + i * word_dur, 3),
                 "end": round(start + (i + 1) * word_dur, 3), "score": 0.5}
                for i, w in enumerate(words)
            ]
        logger.info("Created synthetic word timestamps for diarization")

    # Speaker diarization
    if hf_token:
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token, device="cuda"
            )
            diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=6)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            # P1 FIX: Split segments at speaker boundaries
            result["segments"] = _split_segments_by_speaker(result.get("segments", []))
            del diarize_model
            logger.info("Speaker diarization complete")
        except Exception as e:
            logger.warning(f"Diarization failed (continuing without): {e}")
            from pipeline.degradations import report
            report("diarization", f"pyannote failed: {e}",
                   "no speaker labels for this call")
    else:
        logger.warning("No HF token — skipping diarization")
        from pipeline.degradations import report
        report(
            "diarization",
            "HF_TOKEN not set (pyannote speaker-diarization-3.1 is gated)",
            "no speaker labels: agent/customer talk share, per-speaker emotion, and "
            "agent-domination / third-party-disclosure compliance checks are unavailable",
        )

    # NOTE: WhisperX model is NOT unloaded here — it stays in GPU cache.
    # The orchestrator calls unload_whisperx() when it needs VRAM for
    # emotion2vec / Ollama, then reload_whisperx() after the pipeline finishes.

    # Apply financial term corrections
    corrections = load_correction_dictionary()
    segments = _apply_corrections(result.get("segments", []), corrections)

    # Flag low-confidence segments (P2 FIX: use saved scores if word-level lost)
    low_confidence = []
    confidences = []
    for i, seg in enumerate(segments):
        words = seg.get("words", [])
        if words:
            # Try word-level scores first
            word_scores = []
            for w in words:
                s = w.get("score", w.get("conf"))
                if s is not None:
                    word_scores.append(s)
                else:
                    # Look up from pre-alignment saved scores
                    key = (w.get("word", "").strip().lower(), round(w.get("start", 0), 2))
                    saved = _saved_word_scores.get(key)
                    if saved is not None:
                        word_scores.append(saved)
                        w["score"] = saved  # reattach for downstream use

            if word_scores:
                avg_conf = sum(word_scores) / len(word_scores)
            else:
                # Fall back to segment-level logprob confidence
                seg_start = round(seg.get("start", 0), 1)
                avg_conf = _saved_segment_conf.get(seg_start, 0.5)
        else:
            seg_start = round(seg.get("start", 0), 1)
            avg_conf = _saved_segment_conf.get(seg_start, 0.5)
        seg["confidence"] = round(avg_conf, 3)
        confidences.append(avg_conf)
        if avg_conf < 0.7:
            low_confidence.append(i)
            seg["flagged"] = True
        else:
            seg["flagged"] = False

    overall_confidence = sum(confidences) / max(len(confidences), 1)

    return {
        "segments": segments,
        "low_confidence_segments": low_confidence,
        "overall_confidence": round(overall_confidence, 3),
        "num_segments": len(segments),
        "language": detected_language,
    }


def _split_segments_by_speaker(segments: list) -> list:
    """Split segments that contain multiple speakers into sub-segments.

    After whisperx.assign_word_speakers(), each word has a 'speaker' field.
    WhisperX only assigns one speaker per SEGMENT (the majority), but the
    word-level labels reveal mid-segment speaker changes. This function
    splits those segments so each sub-segment has exactly one speaker.
    """
    new_segments = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            new_segments.append(seg)
            continue

        # Check if multiple speakers exist in this segment
        speakers_in_seg = set()
        for w in words:
            sp = w.get("speaker")
            if sp:
                speakers_in_seg.add(sp)

        if len(speakers_in_seg) <= 1:
            new_segments.append(seg)
            continue

        # Multiple speakers — split into sub-segments at speaker boundaries
        current_speaker = None
        current_words = []

        for w in words:
            word_speaker = w.get("speaker", current_speaker)

            if current_speaker is None:
                current_speaker = word_speaker

            if word_speaker != current_speaker and word_speaker is not None:
                # Speaker changed — emit current sub-segment
                if current_words:
                    new_segments.append(_words_to_segment(current_words, current_speaker, seg))
                current_words = [w]
                current_speaker = word_speaker
            else:
                current_words.append(w)

        # Emit final sub-segment
        if current_words:
            new_segments.append(_words_to_segment(current_words, current_speaker, seg))

    split_count = len(new_segments) - len(segments)
    if split_count > 0:
        logger.info(f"Diarization split: {len(segments)} segments → {len(new_segments)} (+{split_count} from speaker boundaries)")

    return new_segments


def _words_to_segment(words: list, speaker: str, original_seg: dict) -> dict:
    """Create a new segment dict from a group of words with the same speaker."""
    text = " ".join(w.get("word", "") for w in words).strip()
    start = words[0].get("start", original_seg.get("start", 0))
    end = words[-1].get("end", original_seg.get("end", 0))
    return {
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
        "words": words,
    }


# ── CONTEXTUAL CORRECTIONS ────────────────────────────────────────────────────
# Some ASR errors cannot be fixed by a word-level mapping because the wrong token
# is an ordinary English word. Whisper renders "EMI" as "me" or "m.e." — mapping
# "me" -> "EMI" outright would corrupt every sentence containing the pronoun, so
# these patterns require financial context around the token.
#
# Observed on real runs: "your me of 12,500 rupees" (English) and "आपकी m.e.
# 12,500 रुपे" (Hindi). Both are the same error and both were previously uncorrected.
CONTEXTUAL_CORRECTIONS: list[tuple[str, str]] = [
    # "m.e." / "m e" as a spelled-out EMI
    (r"\bm\.\s?e\.?(?=\s|$)", "EMI"),
    (r"\bm\s+e\s+i\b", "EMI"),
    # "me" preceded by a determiner and followed by a financial continuation
    (r"\b(your|the|my|his|her|their|aapki|aapka|apna)\s+me\b(?=\s+(of|is|was|for|due|amount|payment|instal))",
     r"\1 EMI"),
    # "me of <amount>" — "your me of 12,500 rupees"
    (r"\bme\s+of\s+(?=[\d₹$]|rs\b|rupee)", "EMI of "),
    # "monthly me" / "me payment"
    (r"\bmonthly\s+me\b", "monthly EMI"),
    (r"\bme\s+(payment|amount|due date|instalment|installment)\b", r"EMI \1"),
    # Common Indian-finance mishearings that need surrounding context
    (r"\bsee\s+bill\b", "CIBIL"),
    (r"\bpan\s+card\s+number\b", "PAN card number"),
]


def _load_org_aliases(path: str = "data/vocab/org_names.json") -> dict:
    """Institution-name corrections, e.g. Whisper hearing 'Trybank' as 'Treebank'.

    Proper nouns are the other class of error a generic dictionary cannot fix:
    the right spelling depends on which institution the deployment serves. Ships
    with the demo tenant and is extended per deployment rather than hardcoded.
    """
    aliases = {
        "treebank": "Trybank",
        "try bank": "Trybank",
        "tri bank": "Trybank",
        "trybank": "Trybank",
    }
    try:
        p = Path(path)
        if p.exists():
            aliases.update({k.lower(): v for k, v in json.loads(p.read_text()).items()})
    except Exception as e:
        logger.warning(f"Could not load org aliases from {path}: {e}")
    return aliases


def _apply_corrections(segments: list, corrections: dict) -> list:
    """Apply financial term corrections to transcript segments.

    Term corrections (EMI, KYC, CIBIL, etc.) fix actual ASR errors and are applied
    to the text. Amount normalization ("20 thousand" → 20000) and currency symbols
    ("rupees" → "₹") are stored as metadata instead of overwriting the transcript,
    preserving readability. Entity extraction captures these separately.
    """
    for seg in segments:
        text = seg.get("text", "")
        original = text

        # Term corrections only (case-insensitive) — fixes ASR errors
        for wrong, right in corrections.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)

        # Context-dependent corrections, for errors whose wrong form is a real word
        for pattern, replacement in CONTEXTUAL_CORRECTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Institution names
        for wrong, right in _ORG_ALIASES.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)

        # Extract normalized amounts as metadata (don't replace in text)
        amounts = []
        for pattern, replacement in AMOUNT_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                raw = match.group(0)
                value = int(replacement(match))
                amounts.append({"raw": raw, "value": value})
        if amounts:
            seg["normalized_amounts"] = amounts

        seg["text"] = text
        if text != original:
            seg["corrected"] = True
            seg["original_text"] = original
        else:
            seg["corrected"] = False

    return segments
