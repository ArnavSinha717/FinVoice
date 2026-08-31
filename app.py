"""FinVoice — Financial Call Intelligence Pipeline API."""

import os
import json
import uuid
import threading
from pathlib import Path
from contextlib import asynccontextmanager

# PyTorch 2.8 compatibility patches — must run before any model imports
import torch

# Patch 1: torch.load weights_only default changed to True in 2.6+
# pyannote/speechbrain/FunASR checkpoints need weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for all checkpoint loads.
    # lightning_fabric passes weights_only=True explicitly, and PyTorch 2.8
    # treats None as True — both break pyannote/speechbrain/FunASR checkpoints
    # that contain omegaconf objects not in the safe list.
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch 2: Module.to() fails on meta tensors (accelerate/transformers 4.57+ use them during loading)
# This affects ALL models: WhisperX, pyannote, FinBERT, Detoxify, emotion2vec, NER
# Applied globally so every model load in the process is covered
_original_module_to = torch.nn.Module.to
def _safe_module_to(self, *args, **kwargs):
    try:
        return _original_module_to(self, *args, **kwargs)
    except NotImplementedError:
        device = args[0] if args else kwargs.get("device", "cpu")
        return self.to_empty(device=device)
torch.nn.Module.to = _safe_module_to

# Patch 3: load_state_dict() silently drops weights when model uses meta tensors
# PyTorch 2.8 with accelerate/speechbrain creates models on "meta" device for lazy init.
# Without assign=True, load_state_dict copies real weights into meta placeholders → no-op.
# This causes Silero VAD LSTM weights to remain uninitialized → WhisperX produces 0 segments.
_original_load_state_dict = torch.nn.Module.load_state_dict
def _patched_load_state_dict(self, state_dict, *args, **kwargs):
    kwargs.setdefault("assign", True)
    return _original_load_state_dict(self, state_dict, *args, **kwargs)
torch.nn.Module.load_state_dict = _patched_load_state_dict

from dotenv import load_dotenv
load_dotenv()

import time as _time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.auth import require_api_key, log_status as log_auth_status
from pipeline import call_index
from services.llm.client import check_ollama_health
from services.asr.transcriber import preload_whisperx, is_whisperx_loaded
from pipeline.orchestrator import process_call

def _reconcile_orphaned_runs(output_dir: str = "data/processed") -> int:
    """Fail any run that was in flight when the process died.

    Pipeline runs live in daemon threads, so a restart kills them silently while
    their progress file still says "processing". The dashboard then polls that call
    forever with no error and no result. Anything still marked in-progress at
    startup cannot be running, because nothing survived the restart.
    """
    progress_dir = Path(output_dir)
    if not progress_dir.exists():
        return 0

    orphaned = 0
    for f in progress_dir.glob("*_progress.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("status") != "processing":
                continue
            call_id = data.get("call_id") or f.name.split("_")[0]
            if (progress_dir / f"{call_id}_record.json").exists():
                data["status"] = "complete"   # finished but never flushed
            else:
                data["status"] = "error"
                data["error"] = ("Pipeline was interrupted by a server restart. "
                                 "Re-upload the call to process it again.")
            f.write_text(json.dumps(data))
            orphaned += 1
        except Exception as e:
            logger.warning(f"Could not reconcile {f.name}: {e}")

    if orphaned:
        logger.warning(f"Reconciled {orphaned} interrupted pipeline run(s) at startup")
    return orphaned


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load WhisperX at server start so the first call transcribes instantly."""
    def _preload():
        try:
            preload_whisperx()
        except Exception as e:
            logger.warning(f"WhisperX preload failed (will load on first call): {e}")

    log_auth_status()
    _reconcile_orphaned_runs()
    # Background thread so the server starts responding immediately
    threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
    title="FinVoice",
    description="Financial call intelligence pipeline — raw audio to structured, ML-trainable, audit-ready data",
    version="0.2.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """Health check — GPU status, Ollama status, system info."""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024 ** 2,
            "vram_allocated_mb": int(torch.cuda.memory_allocated() / 1024 ** 2),
        }

    ollama = check_ollama_health()

    return {
        "status": "healthy",
        "gpu": gpu_info,
        "ollama": ollama,
        "whisperx_loaded": is_whisperx_loaded(),
    }


@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    call_type: str = Form("general"),
    start_time: float = Form(0),
    end_time: float = Form(0),
):
    """Upload an audio file and start the processing pipeline.

    Returns immediately with a call_id. Poll /api/calls/{call_id}/progress
    to track pipeline progress, then fetch full results from /api/results/{call_id}.
    """
    # Save uploaded file
    upload_dir = Path("data/raw_audio")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    if not safe_name or safe_name in (".", ".."):
        safe_name = "upload"
    file_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"Received {file.filename} ({len(content)} bytes), call_type={call_type}")

    # Generate call_id upfront so we can return it immediately and track this specific run
    call_id = str(uuid.uuid4())[:8]
    output_dir = "data/processed"

    # Write initial progress file BEFORE returning — guarantees frontend
    # always has something to poll, and it's for THIS specific call
    from pipeline.orchestrator import _write_progress
    _write_progress(call_id, output_dir, "0", "Queued", [], "processing")

    # Run pipeline in background thread
    def _run_pipeline():
        try:
            process_call(
                input_audio_path=str(file_path),
                call_type=call_type,
                output_dir=output_dir,
                call_id=call_id,
                start_time=start_time if start_time > 0 else None,
                end_time=end_time if end_time > 0 else None,
            )
        except Exception as e:
            logger.error(f"Pipeline failed for {call_id}: {e}")
            _write_progress(call_id, output_dir, "error", str(e), [], "error", str(e))

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    return JSONResponse(content={
        "status": "processing",
        "call_id": call_id,
        "message": f"Pipeline started for {file.filename}",
        "filename": file.filename,
    }, status_code=202)


@app.get("/api/progress")
async def get_latest_progress():
    """Get the progress of the most recent pipeline run."""
    progress_dir = Path("data/processed")
    if not progress_dir.exists():
        return JSONResponse(content={"status": "idle"})

    progress_files = sorted(progress_dir.glob("*_progress.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not progress_files:
        return JSONResponse(content={"status": "idle"})

    try:
        with open(progress_files[0]) as f:
            return JSONResponse(content=json.load(f))
    except Exception:
        return JSONResponse(content={"status": "idle"})


@app.get("/api/calls/{call_id}/progress")
async def get_call_progress(call_id: str):
    """Get the pipeline progress for a specific call."""
    progress_path = Path(f"data/processed/{call_id}_progress.json")
    if not progress_path.exists():
        return JSONResponse(content={"call_id": call_id, "status": "unknown"})

    try:
        with open(progress_path) as f:
            return JSONResponse(content=json.load(f))
    except Exception:
        return JSONResponse(content={"call_id": call_id, "status": "unknown"})


@app.get("/api/results/{call_id}")
async def get_results(call_id: str):
    """Retrieve processed call results by call ID."""

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        return JSONResponse(content=json.load(f))


@app.get("/api/calls")
async def list_calls(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List all processed calls with summary metadata."""
    import json

    results_dir = Path("data/processed")
    if not results_dir.exists():
        return {"calls": [], "total": 0}

    files = sorted(results_dir.glob("*_record.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    total = len(files)
    page = files[offset:offset + limit]

    calls = []
    for f in page:
        try:
            with open(f) as fp:
                data = json.load(fp)
            calls.append({
                "call_id": data.get("call_id"),
                "audio_file": data.get("audio_file"),
                "duration_seconds": data.get("duration_seconds"),
                "language": data.get("detected_language", data.get("language", "en")),
                "call_type": data.get("call_type"),
                "num_speakers": data.get("num_speakers"),
                "overall_risk_level": data.get("overall_risk_level"),
                "compliance_score": data.get("compliance_score"),
                "requires_human_review": data.get("requires_human_review"),
                "review_priority": data.get("review_priority"),
                "call_summary": data.get("call_summary", "")[:200],
                "pii_count": data.get("pii_count", 0),
                "tamper_risk": data.get("tamper_risk", "none"),
            })
        except Exception:
            continue

    return {"calls": calls, "total": total, "offset": offset, "limit": limit}


@app.get("/api/calls/{call_id}/transcript")
async def get_transcript(call_id: str):
    """Get full time-aligned transcript for a call."""
    import json

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        data = json.load(f)

    return {
        "call_id": call_id,
        "language": data.get("detected_language", data.get("language")),
        "num_speakers": data.get("num_speakers"),
        "overall_confidence": data.get("overall_transcript_confidence"),
        "segments": data.get("transcript_segments", []),
    }


@app.get("/api/calls/{call_id}/transcript/masked")
async def get_masked_transcript(call_id: str):
    """Get PII-masked transcript for a call."""
    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        data = json.load(f)

    segments = data.get("transcript_segments", [])
    pii_entities = data.get("pii_entities", [])

    # Build PII mask map: segment_id → list of (text, masked_text) replacements
    mask_map: dict[int, list[tuple[str, str]]] = {}
    for pii in pii_entities:
        seg_id = pii.get("segment_id")
        if seg_id is not None:
            if seg_id not in mask_map:
                mask_map[seg_id] = []
            original = pii.get("text", "")
            masked = pii.get("masked_text", f"[{pii.get('entity_type', 'PII')}]")
            if original:
                mask_map[seg_id].append((original, masked))

    # Apply masks to segments
    masked_segments = []
    for seg in segments:
        seg_copy = dict(seg)
        seg_id = seg.get("id", seg.get("segment_id"))
        text = seg_copy.get("text", "")
        if seg_id in mask_map:
            for original, masked_text in mask_map[seg_id]:
                text = text.replace(original, masked_text)
        seg_copy["text"] = text
        masked_segments.append(seg_copy)

    return {
        "call_id": call_id,
        "masked": True,
        "pii_count": len(pii_entities),
        "segments": masked_segments,
    }


@app.get("/api/calls/{call_id}/download")
async def download_call(call_id: str):
    """Download the full analysis JSON for a single call."""
    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        # Also check batch_results
        result_path = Path(f"data/processed/batch_results/{call_id}_record.json")
        if not result_path.exists():
            raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    return FileResponse(
        str(result_path),
        media_type="application/json",
        filename=f"{call_id}_analysis.json",
    )


@app.get("/api/calls/{call_id}/entities")
async def get_entities(call_id: str):
    """Get all extracted entities (financial + PII) for a call."""
    import json

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        data = json.load(f)

    return {
        "call_id": call_id,
        "financial_entities": data.get("financial_entities", []),
        "pii_entities": data.get("pii_entities", []),
        "obligations": data.get("obligations", []),
    }


@app.get("/api/calls/{call_id}/compliance")
async def get_compliance(call_id: str):
    """Get compliance report for a call."""
    import json

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        data = json.load(f)

    return {
        "call_id": call_id,
        "compliance_score": data.get("compliance_score"),
        "overall_risk_level": data.get("overall_risk_level"),
        "compliance_checks": data.get("compliance_checks", []),
        "fraud_signals": data.get("fraud_signals", []),
        "toxicity_flags": data.get("toxicity_flags", []),
        "tamper_signals": data.get("tamper_signals", []),
        "tamper_risk": data.get("tamper_risk", "none"),
        "review_reasons": data.get("review_reasons", []),
    }


@app.get("/api/calls/{call_id}/emotions")
async def get_emotions(call_id: str):
    """Get emotion analysis for a call."""
    import json

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        data = json.load(f)

    # Normalize segment emotion field names for frontend
    raw_emotions = data.get("segment_emotions", [])
    normalized_emotions = []
    for e in raw_emotions:
        normalized_emotions.append({
            "segment_id": e.get("segment_id"),
            "speaker": e.get("speaker", ""),
            "emotion": e.get("emotion", e.get("emotion_label", "neutral")),
            "score": e.get("emotion_score", e.get("score", 0)),
        })

    # Fix dominant emotion — use emotion_distribution if dominant is "neutral" but data shows otherwise
    dominant = data.get("customer_emotion_dominant", "neutral")
    dist = data.get("emotion_distribution", {})
    if dominant == "neutral" and dist:
        dominant = max(dist, key=dist.get)

    return {
        "call_id": call_id,
        "segment_emotions": normalized_emotions,
        "emotion_distribution": dist,
        "customer_sentiment_trajectory": data.get("customer_sentiment_trajectory", []),
        "agent_sentiment_trajectory": data.get("agent_sentiment_trajectory", []),
        "customer_emotion_dominant": dominant,
        "speaker_emotion_breakdown": data.get("speaker_emotion_breakdown", {}),
        "escalation_moments": data.get("escalation_moments", []),
    }


@app.post("/api/calls/{call_id}/corrections")
async def submit_corrections(call_id: str, corrections: dict):
    """Accept human corrections to transcript or entities (review UI)."""
    import json

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    # Save corrections alongside the result
    corrections_path = Path(f"data/processed/{call_id}_corrections.json")
    with open(corrections_path, "w") as f:
        json.dump({"call_id": call_id, "corrections": corrections}, f, indent=2)

    logger.info(f"Corrections saved for call {call_id}")
    return {"status": "saved", "call_id": call_id}


@app.get("/api/stats")
async def get_stats():
    """Aggregate statistics across all processed calls."""
    import json

    results_dir = Path("data/processed")
    if not results_dir.exists():
        return {"total_calls": 0}

    # Aggregate from the index when it is usable; fall back to scanning on any
    # problem, because a stale index must degrade to slow, not to wrong.
    if call_index.ensure_fresh(str(results_dir)):
        try:
            rows = call_index.query(
                "SELECT overall_risk_level, compliance_score, duration_seconds, "
                "language, requires_human_review FROM calls", (), str(results_dir))
            risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            languages: dict[str, int] = {}
            total_compliance = total_duration = pending = 0
            for r in rows:
                risk_counts[r["overall_risk_level"]] = risk_counts.get(r["overall_risk_level"], 0) + 1
                total_compliance += r["compliance_score"] or 0
                total_duration += r["duration_seconds"] or 0
                languages[r["language"]] = languages.get(r["language"], 0) + 1
                pending += 1 if r["requires_human_review"] else 0
            return {
                "total_calls": len(rows),
                "total_duration_seconds": round(total_duration, 1),
                "avg_compliance_score": round(total_compliance / max(len(rows), 1), 1),
                "risk_distribution": risk_counts,
                "languages": languages,
                "pending_reviews": pending,
            }
        except Exception as e:
            logger.warning(f"Stats via index failed, scanning instead: {e}")

    files = list(results_dir.glob("*_record.json"))
    total = len(files)

    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    total_compliance = 0
    total_duration = 0
    languages = {}

    pending_reviews = 0
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            risk = data.get("overall_risk_level", "low")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            total_compliance += data.get("compliance_score", 0)
            total_duration += data.get("duration_seconds", 0)
            lang = data.get("detected_language", data.get("language", "en"))
            languages[lang] = languages.get(lang, 0) + 1
            if data.get("requires_human_review"):
                pending_reviews += 1
        except Exception:
            continue

    return {
        "total_calls": total,
        "total_duration_seconds": round(total_duration, 1),
        "avg_compliance_score": round(total_compliance / max(total, 1), 1),
        "risk_distribution": risk_counts,
        "languages": languages,
        "pending_reviews": pending_reviews,
    }


@app.get("/api/export/{format}")
async def export_data(format: str = "csv"):
    """Export all processed call data in specified format as a downloadable file.

    Supported formats: csv, parquet, jsonl, training_intents, training_sentiment,
    training_entities, all (returns JSON manifest).
    """
    from pipeline.output_generator import (
        load_records_from_dir, export_to_csv, export_to_parquet,
        export_to_jsonl, export_training_pairs_jsonl,
        export_sentiment_pairs_jsonl, export_entity_pairs_jsonl,
        export_all,
    )

    records = load_records_from_dir("data/processed")
    if not records:
        raise HTTPException(status_code=404, detail="No processed calls found")

    output_dir = "data/exports"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    media_types = {
        "csv": "text/csv",
        "parquet": "application/octet-stream",
        "jsonl": "application/jsonlines",
        "training_intents": "application/jsonlines",
        "training_sentiment": "application/jsonlines",
        "training_entities": "application/jsonlines",
    }

    if format == "csv":
        path = export_to_csv(records, f"{output_dir}/call_summary.csv")
        filename = "call_summary.csv"
    elif format == "parquet":
        path = export_to_parquet(records, f"{output_dir}/calls.parquet")
        filename = "calls.parquet"
    elif format == "jsonl":
        path = export_to_jsonl(records, f"{output_dir}/calls.jsonl")
        filename = "calls.jsonl"
    elif format == "training_intents":
        path = export_training_pairs_jsonl(records, f"{output_dir}/training_intents.jsonl")
        filename = "training_intents.jsonl"
    elif format == "training_sentiment":
        path = export_sentiment_pairs_jsonl(records, f"{output_dir}/training_sentiment.jsonl")
        filename = "training_sentiment.jsonl"
    elif format == "training_entities":
        path = export_entity_pairs_jsonl(records, f"{output_dir}/training_entities.jsonl")
        filename = "training_entities.jsonl"
    elif format == "all":
        outputs = export_all(records, output_dir)
        return {"status": "exported", "files": outputs, "total_calls": len(records)}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Use csv, parquet, jsonl, "
                   f"training_intents, training_sentiment, training_entities, or all"
        )

    if not path or not Path(path).exists():
        raise HTTPException(status_code=500, detail="Export generation failed")

    return FileResponse(
        path,
        media_type=media_types.get(format, "application/octet-stream"),
        filename=filename,
    )


@app.get("/api/calls/{call_id}/intents")
async def get_intents(call_id: str):
    """Get intent classifications for each utterance in a call."""
    import json

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        data = json.load(f)

    # Enrich intents with utterance text from transcript segments
    intents = data.get("intents", [])
    segments = data.get("transcript_segments", [])
    seg_map = {s.get("id", i): s for i, s in enumerate(segments)}
    enriched = []
    for intent in intents:
        seg = seg_map.get(intent.get("segment_id"))
        enriched.append({
            **intent,
            "text": seg["text"] if seg else "",
        })

    return {
        "call_id": call_id,
        "intents": enriched,
    }


@app.post("/api/calls/{call_id}/review-action")
async def submit_review_action(call_id: str, body: dict):
    """Submit a review action (approve/escalate/reject) for a call."""
    action = body.get("action")
    notes = body.get("notes", "")

    if action not in ("approve", "escalate", "reject"):
        raise HTTPException(status_code=400, detail="action must be approve, escalate, or reject")

    result_path = Path(f"data/processed/{call_id}_record.json")
    if not result_path.exists():
        # Also search batch_results subdirectory
        result_path = Path(f"data/processed/batch_results/{call_id}_record.json")
        if not result_path.exists():
            raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    with open(result_path) as f:
        data = json.load(f)

    data["review_status"] = action
    data["review_notes"] = notes
    data["review_timestamp"] = _time.time()

    if action == "approve":
        data["requires_human_review"] = False
    elif action == "escalate":
        data["review_priority"] = 1

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Review action '{action}' applied to call {call_id}")
    return {"status": action, "call_id": call_id}


@app.get("/api/calls/{call_id}/audio")
async def get_audio(call_id: str):
    """Serve the audio file for playback in the frontend."""
    # Try cleaned version first, then normalized
    for pattern in [
        f"data/transcripts/{call_id}_cleaned.wav",
        f"data/transcripts/{call_id}_normalized.wav",
    ]:
        path = Path(pattern)
        if path.exists():
            return FileResponse(str(path), media_type="audio/wav", filename=f"{call_id}.wav")

    # Look up original path from record
    for search_dir in ["data/processed", "data/processed/batch_results"]:
        result_path = Path(f"{search_dir}/{call_id}_record.json")
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            original = data.get("audio_file")
            if original and Path(original).exists():
                ext = Path(original).suffix.lower()
                media_type = {
                    ".wav": "audio/wav", ".mp3": "audio/mpeg",
                    ".m4a": "audio/mp4", ".ogg": "audio/ogg",
                }.get(ext, "audio/mpeg")
                return FileResponse(original, media_type=media_type, filename=f"{call_id}{ext}")

    raise HTTPException(status_code=404, detail="Audio file not found")


@app.get("/api/review-queue")
async def get_review_queue():
    """Get all calls requiring human review, sorted by priority."""
    import json

    results_dir = Path("data/processed")
    if not results_dir.exists():
        return {"items": [], "total": 0}

    if call_index.ensure_fresh(str(results_dir)):
        try:
            rows = call_index.query(
                "SELECT * FROM calls WHERE requires_human_review = 1 "
                "ORDER BY review_priority ASC, mtime DESC", (), str(results_dir))
            items = [{
                "call_id": r["call_id"],
                "review_priority": r["review_priority"],
                "overall_risk_level": r["overall_risk_level"],
                "call_summary": r["call_summary"],
                "review_reasons": json.loads(r["review_reasons"] or "[]"),
                "call_type": r["call_type"],
                "language": r["language"],
                "duration_seconds": r["duration_seconds"],
                "compliance_score": r["compliance_score"],
                "date": r["mtime"],
            } for r in rows]
            return {"items": items, "total": len(items)}
        except Exception as e:
            logger.warning(f"Review queue via index failed, scanning instead: {e}")

    items = []
    for f in results_dir.glob("*_record.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if not data.get("requires_human_review"):
                continue
            items.append({
                "call_id": data.get("call_id"),
                "review_priority": data.get("review_priority", 0),
                "overall_risk_level": data.get("overall_risk_level", "low"),
                "call_summary": data.get("call_summary", ""),
                "review_reasons": data.get("review_reasons", []),
                "call_type": data.get("call_type"),
                "language": data.get("detected_language", data.get("language", "en")),
                "duration_seconds": data.get("duration_seconds"),
                "compliance_score": data.get("compliance_score"),
                "date": f.stat().st_mtime,
            })
        except Exception:
            continue

    # review_priority: 1 = urgent ... 5 = routine, so ascending puts urgent first
    items.sort(key=lambda x: x.get("review_priority", 5))
    return {"items": items, "total": len(items)}


def _build_local_context(call_id: str | None = None) -> str:
    """Build a concise context string from local processed records for RAG."""
    import json as _json
    results_dir = Path("data/processed")
    if not results_dir.exists():
        return ""

    records = []
    if call_id:
        p = results_dir / f"{call_id}_record.json"
        if p.exists():
            with open(p) as f:
                records.append(_json.load(f))
    else:
        for p in sorted(results_dir.glob("*_record.json"))[-20:]:
            try:
                with open(p) as f:
                    records.append(_json.load(f))
            except Exception:
                continue

    if not records:
        return ""

    parts = []
    for rec in records:
        cid = rec.get("call_id", "?")
        summary = rec.get("call_summary", "N/A")
        if summary in ("N/A", "Call processed — see extracted fields.", "[2-3 sentence summary]"):
            summary = ""
        call_lines = [
            f"[CALL {cid}] type={rec.get('call_type','general')} duration={rec.get('duration_seconds',0):.0f}s risk={rec.get('overall_risk_level','low')} compliance={rec.get('compliance_score',100)}/100 speakers={rec.get('num_speakers',0)}",
        ]
        if summary:
            call_lines.append(f"Summary: {summary[:300]}")
        # Outcomes
        outcomes = rec.get("key_outcomes", [])
        if outcomes and outcomes != ["See analysis fields"]:
            call_lines.append(f"Outcomes: {'; '.join(str(o) for o in outcomes[:4])}")
        # Financial entities — compact
        ents = rec.get("financial_entities", [])
        amounts = [e for e in ents if e.get("entity_type") in ("payment_amount", "currency_amount")]
        orgs = [e for e in ents if e.get("entity_type") == "organization"]
        people = [e for e in ents if e.get("entity_type") == "person_name"]
        if amounts:
            call_lines.append(f"Amounts: {', '.join(e.get('raw_text','') for e in amounts[:10])}")
        if orgs:
            call_lines.append(f"Organizations: {', '.join(e.get('value',e.get('raw_text','')) for e in orgs[:8])}")
        if people:
            call_lines.append(f"People: {', '.join(e.get('value',e.get('raw_text','')) for e in people[:6])}")
        # Compliance violations
        violations = [c for c in rec.get("compliance_checks", []) if not c.get("passed", True)]
        if violations:
            call_lines.append(f"Violations: {'; '.join(v.get('check_name','?') + ' [' + v.get('severity','?') + ']' for v in violations[:5])}")
        # Fraud
        fraud = rec.get("fraud_signals", [])
        if fraud:
            call_lines.append(f"Fraud: {'; '.join(s.get('signal_type','?') for s in fraud[:3])}")
        # Key transcript passages (limit to ~5 most content-rich segments)
        segs = rec.get("transcript_segments", [])
        if segs:
            # Pick segments with the most words for richer context
            ranked = sorted(segs, key=lambda s: len(s.get("text", "").split()), reverse=True)
            top = ranked[:5]
            call_lines.append("Key passages: " + " | ".join(s.get("text", "").strip()[:200] for s in top))
        parts.append("\n".join(call_lines))

    return "\n\n".join(parts)


@app.post("/api/audit/query")
async def audit_query(body: dict):
    """Answer a question across all processed calls, using the local LLM over local records.

    Body: {"question": "Show me all calls with compliance violations", "call_id": "optional"}
    """
    question = body.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field")

    call_id = body.get("call_id")
    local_context = _build_local_context(call_id)

    # Answered entirely locally: the context is assembled from processed records and
    # passed to the local LLM, so no call data leaves the machine.
    if not local_context:
        raise HTTPException(
            status_code=404,
            detail="No processed calls to answer from. Upload and process a call first.",
        )

    try:
        answer = await _local_llm_query(question, local_context)
    except Exception as e:
        logger.error(f"Local LLM query failed: {e}")
        raise HTTPException(status_code=503, detail=f"Local LLM unavailable: {e}")
    return {"question": question, "answer": answer}


async def _local_llm_query(question: str, context: str) -> str:
    """Answer a question using local Ollama with call data context."""
    import httpx

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    system = (
        "You are FinVoice AI Analyst, an expert financial call intelligence assistant. "
        "You have access to processed call recordings from a financial institution. "
        "Answer questions using ONLY the provided call data. Be specific — cite call IDs, "
        "quote transcript passages, and reference exact figures. "
        "If the data doesn't contain the answer, say what data IS available."
    )
    prompt = f"=== CALL DATA ===\n{context}\n\n=== QUESTION ===\n{question}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{ollama_url}/api/chat",
            json={
                "model": "qwen2.5:3b",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024},
            },
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "No response")


@app.post("/api/compliance/reason")
async def compliance_reason(body: dict):
    """Explain a compliance finding in natural language, using the local LLM.

    Body: {
        "transcript_excerpt": "We may have to take further steps",
        "context": "Collections call, customer missed 3 EMIs",
        "regulation": "RBI Fair Practice Code"
    }
    """
    excerpt = body.get("transcript_excerpt") or body.get("excerpt") or ""
    context = body.get("context", "")
    regulation = body.get("regulation", "RBI Fair Practice Code")

    if not excerpt:
        raise HTTPException(status_code=400, detail="Missing 'transcript_excerpt' field")

    try:
        reasoning = await _local_llm_query(
            f"Under {regulation}, assess this call excerpt for compliance. "
            f"State whether it is a violation, cite the specific principle, and keep it to 3 sentences.",
            f"Excerpt: {excerpt}\nCall context: {context or 'not provided'}",
        )
        return {"reasoning": reasoning, "regulation": regulation, "source": "local"}
    except Exception as e:
        logger.warning(f"Compliance reasoning failed: {e}")
        raise HTTPException(status_code=503, detail=f"Local LLM unavailable: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
