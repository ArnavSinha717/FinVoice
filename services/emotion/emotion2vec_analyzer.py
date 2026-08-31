"""Emotion2vec — Audio-based emotion recognition from speech waveforms.

Uses iic/emotion2vec_plus_large (FunASR) to detect emotions directly from audio:
  angry, disgusted, fearful, happy, neutral, sad, surprised, other

Unlike FinBERT (text-only sentiment), emotion2vec captures vocal tone, pitch
patterns, and prosody — it detects anger even in politely-worded sentences.

VRAM: ~1GB. In the 6GB VRAM lifecycle:
  Stage 3: WhisperX (~3GB) → unload
  Stage 4 (new): emotion2vec (~1GB) → unload
  Stage 4E: Ollama/Qwen3 (~5GB) → unload

Optimized for batch processing — segments are grouped into mini-batches
to reduce per-segment file I/O overhead.
"""

import os
import tempfile
import traceback
import numpy as np
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field

try:
    from funasr import AutoModel
    HAS_EMOTION2VEC = True
except ImportError:
    HAS_EMOTION2VEC = False
    logger.warning("funasr not installed — emotion2vec disabled")


EMOTION_LABELS = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "sad", "surprised", "other",
]

_model = None


class SegmentEmotion(BaseModel):
    """Emotion classification for a single transcript segment."""
    segment_id: int
    speaker: str
    emotion: str = Field(description="Primary emotion: angry, happy, sad, neutral, etc.")
    emotion_score: float = Field(ge=0, le=1, description="Confidence of primary emotion")
    all_scores: dict[str, float] = Field(description="Scores for all 8 emotions")
    low_confidence: bool = Field(default=False, description="True if emotion was below confidence threshold")


# Free VRAM (MiB) required before emotion2vec will claim the GPU. The model is
# ~1.2 GB; the margin leaves room for WhisperX and the resident Ollama model.
_EMOTION_VRAM_MARGIN_MB = int(os.getenv("FINVOICE_EMOTION_VRAM_MARGIN_MB", "2200"))


def _select_emotion_device() -> str:
    """Choose cuda or cpu for emotion2vec, preferring cuda when there is headroom."""
    forced = os.getenv("FINVOICE_EMOTION_DEVICE", "auto").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free_mb = torch.cuda.mem_get_info()[0] // (1024 ** 2)
        if free_mb >= _EMOTION_VRAM_MARGIN_MB:
            logger.info(f"emotion2vec -> cuda ({free_mb} MiB free)")
            return "cuda"
        logger.info(f"emotion2vec -> cpu (only {free_mb} MiB free, "
                    f"need {_EMOTION_VRAM_MARGIN_MB} MiB)")
        return "cpu"
    except Exception as e:
        logger.warning(f"Could not inspect VRAM, using CPU for emotion2vec: {e}")
        return "cpu"


def _get_model():
    """Load emotion2vec model (lazy, cached).

    PyTorch 2.8 breaks FunASR's model.to("cuda") due to meta tensors.
    Fix: monkey-patch Module.to() during load to handle meta tensors via to_empty().
    This lets FunASR initialize fully on GPU with correct device placement for inputs.
    """
    global _model
    if _model is not None:
        return _model

    if not HAS_EMOTION2VEC:
        return None

    import torch

    # PyTorch 2.8 fix: The global patches (app.py) cause meta tensors to persist
    # even on CPU. FunASR loads model structure first (with meta tensors), then
    # should load weights — but the patches interfere.
    #
    # Nuclear fix: after AutoModel creates the model, manually reload the checkpoint
    # weights with assign=True, forcing real tensors to replace meta placeholders.
    # Device choice. This was pinned to CPU when the pipeline used qwen3:8b (5.2 GB
    # of a 6 GB card), which left no room for anything else. That model is gone:
    # qwen2.5:3b is ~2.2 GB and WhisperX ~1.2 GB, so there is headroom. emotion2vec
    # had meanwhile become the slowest analyzer — 30-60s of a 65-88s pipeline, all
    # of it CPU inference.
    #
    # FINVOICE_EMOTION_DEVICE forces the choice; the default inspects free VRAM and
    # falls back to CPU rather than risking an OOM mid-run.
    _device = _select_emotion_device()
    logger.info(f"Loading emotion2vec_plus_large on {_device}...")
    try:
        _model = AutoModel(
            model="iic/emotion2vec_plus_large",
            trust_remote_code=True,
            device=_device,
            disable_update=True,
        )

        # Find the checkpoint file and reload weights directly
        # ModelScope has used two cache layouts; check both rather than silently
        # skipping the meta-tensor weight reload below.
        import glob as _glob
        _candidates = [
            os.path.expanduser("~/.cache/modelscope/hub/models/iic/emotion2vec_plus_large/model.pt"),
            *sorted(_glob.glob(os.path.expanduser(
                "~/.cache/modelscope/models/iic--emotion2vec_plus_large/snapshots/*/model.pt"
            ))),
        ]
        ckpt_path = next((c for c in _candidates if os.path.exists(c)), _candidates[0])
        if os.path.exists(ckpt_path):
            # Get the actual nn.Module inside FunASR's wrapper
            inner_model = None
            if hasattr(_model, 'model'):
                inner_model = _model.model
            elif hasattr(_model, 'module'):
                inner_model = _model.module

            if inner_model is not None and hasattr(inner_model, 'load_state_dict'):
                # Count meta tensors before reload
                meta_before = sum(
                    1 for p in inner_model.parameters() if p.device.type == "meta"
                )
                if meta_before > 0:
                    logger.info(f"emotion2vec: {meta_before} meta tensors detected, reloading weights...")
                    checkpoint = torch.load(ckpt_path, map_location=_device)
                    # FunASR checkpoints often have a 'model' key wrapping state_dict
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    # Strip 'd2v_model.' prefix if present (FunASR naming convention)
                    cleaned = {}
                    for k, v in state_dict.items():
                        clean_key = k.replace("d2v_model.", "") if k.startswith("d2v_model.") else k
                        cleaned[clean_key] = v

                    inner_model.load_state_dict(cleaned, strict=False, assign=True)
                    meta_after = sum(
                        1 for p in inner_model.parameters() if p.device.type == "meta"
                    )
                    logger.info(f"emotion2vec: weights reloaded, meta tensors {meta_before} → {meta_after}")

                    # Also fix buffers
                    for name, buf in list(inner_model.named_buffers()):
                        if buf.device.type == "meta":
                            parts = name.split(".")
                            mod = inner_model
                            for part in parts[:-1]:
                                mod = getattr(mod, part)
                            setattr(mod, parts[-1], torch.zeros_like(buf, device="cpu"))
        else:
            logger.warning(f"emotion2vec checkpoint not found at {ckpt_path}")

        logger.info("emotion2vec loaded on CPU (PyTorch 2.8 safe)")
    except Exception as e:
        logger.error(f"emotion2vec failed to load: {e}")
        logger.error(traceback.format_exc())
        _model = None

    return _model


def unload_model():
    """Free emotion2vec from GPU memory."""
    global _model
    if _model is not None:
        del _model
        _model = None
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("emotion2vec unloaded, VRAM freed")
        except Exception:
            pass


def _parse_emotion_result(result_item, segment_id: int, speaker: str) -> SegmentEmotion | None:
    """Parse a single emotion2vec result into a SegmentEmotion object.

    Handles dict, object, and nested list formats from different FunASR versions.
    """
    labels = None
    scores = None

    if isinstance(result_item, dict):
        labels = result_item.get("labels", result_item.get("label", EMOTION_LABELS))
        scores = result_item.get("scores", result_item.get("score", []))
        # FunASR may nest labels/scores as lists of lists
        if isinstance(labels, list) and labels and isinstance(labels[0], list):
            labels = labels[0]
        if isinstance(scores, list) and scores and isinstance(scores[0], list):
            scores = scores[0]
    elif hasattr(result_item, "labels"):
        labels = result_item.labels
        scores = result_item.scores if hasattr(result_item, "scores") else []
    else:
        return None

    if not scores or not labels:
        return None

    # Build scores dict, mapping FunASR labels to our standard labels
    score_dict = {}
    for label, score_val in zip(labels, scores):
        label_clean = label.strip("/").lower() if isinstance(label, str) else str(label)
        matched = False
        for emotion in EMOTION_LABELS:
            if emotion in label_clean:
                score_dict[emotion] = float(score_val)
                matched = True
                break
        if not matched:
            score_dict[label_clean] = float(score_val)

    if not score_dict:
        return None

    primary = max(score_dict, key=score_dict.get)
    primary_score = score_dict[primary]

    # Confidence threshold: below 0.4, the classification isn't meaningful.
    # Also remap "other"/<unk> to "neutral" — these are unclassifiable segments.
    MIN_CONFIDENCE = 0.4
    is_low_confidence = primary_score < MIN_CONFIDENCE or primary == "other"

    if is_low_confidence:
        # Fall back to "neutral" but preserve original scores for debugging
        primary = "neutral"
        primary_score = score_dict.get("neutral", primary_score)

    # emotion2vec softmax output can land a hair outside [0, 1] (e.g. 1.0000000149)
    # from float32 rounding. SegmentEmotion constrains the score to [0, 1], so an
    # unclamped value raises ValidationError and discards the whole batch — which
    # silently produced zero emotions for the entire call.
    def _clamp(x: float) -> float:
        return round(min(1.0, max(0.0, float(x))), 3)

    return SegmentEmotion(
        segment_id=segment_id,
        speaker=speaker,
        emotion=primary,
        emotion_score=_clamp(primary_score),
        all_scores={k: _clamp(v) for k, v in score_dict.items()},
        low_confidence=is_low_confidence,
    )


def analyze_emotions(
    wav_path: str,
    segments: list,
    granularity: str = "utterance",
    audio_array: np.ndarray | None = None,
    sample_rate: int = 16000,
) -> list[SegmentEmotion]:
    """Analyze emotions per segment from audio waveforms.

    Passes numpy arrays directly to FunASR — no temp file I/O.
    Falls back to temp files only if direct array input fails.

    Args:
        wav_path: Path to 16kHz mono WAV file (used as fallback)
        segments: Transcript segments with 'start', 'end', 'speaker' fields
        granularity: 'utterance' for per-segment analysis
        audio_array: Pre-loaded audio numpy array (avoids redundant disk read)
        sample_rate: Sample rate of audio_array (default 16000)

    Returns:
        List of SegmentEmotion objects
    """
    model = _get_model()
    if model is None:
        logger.warning("emotion2vec not available — skipping emotion analysis")
        from pipeline.degradations import report
        report("emotion2vec", "model unavailable (check the funasr version pin)",
               "segment_emotions, emotion_distribution, speaker_emotion_breakdown, "
               "emotion_transitions and escalation_moments are all empty")
        return []

    if audio_array is not None:
        audio_data = audio_array
        sr = sample_rate
    else:
        import soundfile as sf
        try:
            audio_data, sr = sf.read(wav_path)
        except Exception as e:
            logger.error(f"Could not read audio for emotion2vec: {e}")
            return []

    # Convert to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    total_samples = len(audio_data)

    # Pre-slice all valid segments into chunks
    valid_segments = []  # (segment_index, segment_dict, audio_chunk)
    for i, seg in enumerate(segments):
        start_time = seg.get("start", 0)
        end_time = seg.get("end", start_time + 1)

        if end_time - start_time < 0.5:
            continue

        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), total_samples)

        if end_sample <= start_sample:
            continue

        chunk = audio_data[start_sample:end_sample]
        if len(chunk) < sr * 0.5:
            continue

        valid_segments.append((i, seg, chunk))

    if not valid_segments:
        logger.warning("emotion2vec: no valid segments to analyze (all too short)")
        return []

    # Smart subsample for very long files (>100 segments)
    # Prioritize: speaker changes, first/last 5 segments per speaker, then uniform skip
    if len(valid_segments) > 100:
        original_count = len(valid_segments)
        keep_indices = set()

        # Always keep first 5 and last 5 segments
        for idx in range(min(5, len(valid_segments))):
            keep_indices.add(idx)
        for idx in range(max(0, len(valid_segments) - 5), len(valid_segments)):
            keep_indices.add(idx)

        # Keep segments near speaker changes
        prev_speaker = None
        for idx, (_, seg, _) in enumerate(valid_segments):
            speaker = seg.get("speaker", "")
            if speaker != prev_speaker and prev_speaker is not None:
                # Keep the segment before and after the change
                keep_indices.add(max(0, idx - 1))
                keep_indices.add(idx)
            prev_speaker = speaker

        # Fill remaining with uniform sampling to reach ~60 segments
        target = 60
        if len(keep_indices) < target:
            remaining = [i for i in range(len(valid_segments)) if i not in keep_indices]
            step = max(1, len(remaining) // (target - len(keep_indices)))
            for i in range(0, len(remaining), step):
                keep_indices.add(remaining[i])
                if len(keep_indices) >= target:
                    break

        valid_segments = [valid_segments[i] for i in sorted(keep_indices)]
        logger.info(f"emotion2vec: smart subsampling {original_count} → {len(valid_segments)} segments")

    logger.info(f"emotion2vec: processing {len(valid_segments)} segments (direct numpy, no temp files)")
    results = []
    BATCH_SIZE = 40  # Larger batches since no file I/O overhead

    for batch_start in range(0, len(valid_segments), BATCH_SIZE):
        batch = valid_segments[batch_start:batch_start + BATCH_SIZE]

        # Pass numpy arrays directly to FunASR (no temp file I/O)
        # Ensure float32 — emotion2vec model weights are float32, numpy default is float64
        import numpy as np
        chunks = [chunk.astype(np.float32) if chunk.dtype != np.float32 else chunk for _, _, chunk in batch]
        try:
            batch_results = model.generate(
                input=chunks,
                output_dir=None,
                granularity=granularity,
            )

            for idx, (seg_idx, seg, _) in enumerate(batch):
                if idx < len(batch_results):
                    parsed = _parse_emotion_result(
                        batch_results[idx], seg_idx, seg.get("speaker", "unknown")
                    )
                    if parsed:
                        results.append(parsed)
                    if not results and idx == 0:
                        logger.debug(f"emotion2vec raw result: {str(batch_results[idx])[:300]}")

        except Exception as batch_err:
            # Fallback: write temp files for this batch (older FunASR may not accept arrays)
            if batch_start == 0:
                logger.warning(f"Direct numpy failed ({batch_err}), falling back to temp files")
            import soundfile as sf
            temp_paths = []
            try:
                for _, _, chunk in batch:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                    sf.write(tmp_path, chunk, sr)
                    temp_paths.append(tmp_path)

                fallback_results = model.generate(
                    input=temp_paths,
                    output_dir=None,
                    granularity=granularity,
                )
                for idx, (seg_idx, seg, _) in enumerate(batch):
                    if idx < len(fallback_results):
                        parsed = _parse_emotion_result(
                            fallback_results[idx], seg_idx, seg.get("speaker", "unknown")
                        )
                        if parsed:
                            results.append(parsed)
            except Exception as e:
                logger.warning(f"emotion2vec batch {batch_start} failed entirely: {e}")
            finally:
                for tmp_path in temp_paths:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    if results:
        emotion_counts = {}
        for r in results:
            emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1
        dist = ", ".join(f"{k}={v}" for k, v in sorted(emotion_counts.items(), key=lambda x: -x[1]))
        logger.info(f"emotion2vec: {len(results)} segments analyzed — {dist}")
    else:
        logger.warning("emotion2vec: no segments could be analyzed")

    return results


def get_emotion_summary(emotions: list[SegmentEmotion]) -> dict:
    """Summarize emotions across the call.

    Uses score-weighted averaging for dominant emotion (not just count).
    30 segments of "happy" at 0.51 won't beat 20 segments of "angry" at 0.95.

    Returns:
        Dict with dominant_emotion, emotion_distribution, escalation_moments
    """
    if not emotions:
        return {
            "dominant_emotion": "neutral",
            "emotion_distribution": {},
            "escalation_moments": [],
        }

    # Score-weighted dominant: average confidence per emotion
    weighted_scores = {}
    weighted_counts = {}
    for e in emotions:
        weighted_scores.setdefault(e.emotion, 0.0)
        weighted_counts.setdefault(e.emotion, 0)
        weighted_scores[e.emotion] += e.emotion_score
        weighted_counts[e.emotion] += 1

    weighted_avg = {k: weighted_scores[k] / weighted_counts[k] for k in weighted_scores}
    dominant = max(weighted_avg, key=weighted_avg.get)

    # Distribution by count (for visualization)
    total = len(emotions)
    distribution = {k: round(v / total, 3) for k, v in weighted_counts.items()}

    # Find escalation moments (angry/fearful with high confidence)
    escalation_moments = []
    for e in emotions:
        if e.emotion in ("angry", "fearful") and e.emotion_score > 0.6:
            escalation_moments.append({
                "segment_id": e.segment_id,
                "speaker": e.speaker,
                "emotion": e.emotion,
                "score": e.emotion_score,
            })

    return {
        "dominant_emotion": dominant,
        "emotion_distribution": distribution,
        "escalation_moments": escalation_moments,
    }


def get_speaker_emotion_breakdown(emotions: list[SegmentEmotion]) -> dict:
    """Split emotions by speaker for agent vs customer separate profiles.

    Returns:
        {
            "agent": {"dominant": "neutral", "distribution": {"neutral": 0.7, ...}, "total_segments": 15},
            "customer": {"dominant": "frustrated", "distribution": {...}, "total_segments": 10},
        }
    """
    if not emotions:
        return {}

    by_speaker: dict[str, list[SegmentEmotion]] = {}
    for e in emotions:
        speaker = e.speaker.lower()
        by_speaker.setdefault(speaker, []).append(e)

    result = {}
    for speaker, speaker_emotions in by_speaker.items():
        # Score-weighted dominant per speaker
        weighted_scores = {}
        weighted_counts = {}
        for e in speaker_emotions:
            weighted_scores.setdefault(e.emotion, 0.0)
            weighted_counts.setdefault(e.emotion, 0)
            weighted_scores[e.emotion] += e.emotion_score
            weighted_counts[e.emotion] += 1

        weighted_avg = {k: weighted_scores[k] / weighted_counts[k] for k in weighted_scores}
        dominant = max(weighted_avg, key=weighted_avg.get)

        total = len(speaker_emotions)
        result[speaker] = {
            "dominant": dominant,
            "distribution": {k: round(v / total, 3) for k, v in weighted_counts.items()},
            "total_segments": total,
        }

    return result


def detect_emotion_transitions(emotions: list[SegmentEmotion]) -> list[dict]:
    """Detect significant per-speaker emotion shifts across the call.

    Tracks when a speaker's emotion changes meaningfully (e.g. neutral → angry)
    and reports the segment. Ignores low-confidence segments and requires
    the new emotion to persist for at least 2 consecutive segments.

    Returns:
        List of transitions: [
            {"speaker": "customer", "from_emotion": "neutral", "to_emotion": "angry",
             "at_segment": 22, "significance": "high"},
        ]
    """
    if not emotions:
        return []

    # Group by speaker, sorted by segment_id
    by_speaker: dict[str, list[SegmentEmotion]] = {}
    for e in emotions:
        by_speaker.setdefault(e.speaker, []).append(e)

    transitions = []

    negative_emotions = {"angry", "fearful", "sad", "disgusted"}
    positive_emotions = {"happy", "surprised"}

    for speaker, speaker_emotions in by_speaker.items():
        speaker_emotions.sort(key=lambda e: e.segment_id)

        # Skip low-confidence segments for transition detection
        confident = [e for e in speaker_emotions if not e.low_confidence]
        if len(confident) < 3:
            continue

        prev_emotion = confident[0].emotion
        prev_segment = confident[0].segment_id

        for i in range(1, len(confident)):
            curr = confident[i]
            if curr.emotion == prev_emotion:
                continue

            # Check if the new emotion persists (next segment also has it)
            persists = (
                i + 1 < len(confident)
                and confident[i + 1].emotion == curr.emotion
            )
            # Or if confidence is very high, a single segment counts
            high_conf = curr.emotion_score >= 0.7

            if not persists and not high_conf:
                continue

            # Determine significance
            to_neg = curr.emotion in negative_emotions
            from_neg = prev_emotion in negative_emotions
            to_pos = curr.emotion in positive_emotions

            if prev_emotion == "neutral" and to_neg:
                significance = "high"
            elif from_neg and to_pos:
                significance = "medium"  # Recovery
            elif to_neg and not from_neg:
                significance = "high"
            elif prev_emotion != curr.emotion:
                significance = "low"
            else:
                continue

            transitions.append({
                "speaker": speaker,
                "from_emotion": prev_emotion,
                "to_emotion": curr.emotion,
                "at_segment": curr.segment_id,
                "from_segment": prev_segment,
                "significance": significance,
            })

            prev_emotion = curr.emotion
            prev_segment = curr.segment_id

    transitions.sort(key=lambda t: t["at_segment"])

    if transitions:
        high = sum(1 for t in transitions if t["significance"] == "high")
        logger.info(f"Emotion transitions: {len(transitions)} detected ({high} high significance)")

    return transitions
