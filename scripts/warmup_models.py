"""Pre-download every model the pipeline needs, before a demo or first real run.

The pipeline lazy-loads models the first time each stage needs one. On a cold
machine that means Stage 3 and Stage 4 stall for tens of minutes downloading
several GB, and because progress is only reported at stage boundaries the
dashboard looks frozen the whole time.

Run this once, ahead of time, on the machine that will do the demo.

    python scripts/warmup_models.py
    python scripts/warmup_models.py --skip whisperx emotion2vec

Exit code is 0 if everything that can load did load, 1 otherwise.
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

HF_MODELS = [
    ("finbert", "ProsusAI/finbert", "text-classification"),
    ("sentiment-multilingual", "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual", "text-classification"),
    ("zero-shot-scam", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "zero-shot-classification"),
    ("indic-abuse", "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL", "text-classification"),
]


def _timed(label, fn):
    print(f"\n▶ {label} ...", flush=True)
    t0 = time.perf_counter()
    try:
        fn()
        print(f"  ✓ {label} ready ({time.perf_counter() - t0:.0f}s)", flush=True)
        return True
    except Exception as e:
        print(f"  ✗ {label} FAILED: {type(e).__name__}: {e}", flush=True)
        return False


def warm_hf(name, repo, task):
    from transformers import pipeline
    pipeline(task, model=repo, device=-1)


def warm_whisperx():
    from services.asr.transcriber import preload_whisperx, unload_whisperx
    preload_whisperx()
    unload_whisperx()


# WhisperX loads a SEPARATE wav2vec2 alignment model per language, at transcription
# time, inside Stage 3. Warming only the ASR model leaves a Hindi or Tamil call
# stalling mid-pipeline on a fresh download — observed taking 10+ minutes on a
# 79-second call, with no progress reported. Warm every language you claim to serve.
ALIGN_LANGUAGES = os.getenv("FINVOICE_ALIGN_LANGUAGES", "en,hi,ta").split(",")


def warm_alignment():
    import whisperx
    failed = []
    for lang in [l.strip() for l in ALIGN_LANGUAGES if l.strip()]:
        try:
            whisperx.load_align_model(language_code=lang, device="cpu")
            print(f"    alignment [{lang}] ok", flush=True)
        except Exception as e:
            failed.append(f"{lang}: {type(e).__name__}")
            print(f"    alignment [{lang}] FAILED: {type(e).__name__}: {e}", flush=True)
    if failed:
        raise RuntimeError("; ".join(failed))


def warm_emotion2vec():
    from services.emotion.emotion2vec_analyzer import _get_model
    if _get_model() is None:
        raise RuntimeError("emotion2vec model did not load — check the funasr version")


def warm_detoxify():
    from detoxify import Detoxify
    Detoxify("multilingual", device="cpu")


def warm_spacy():
    import spacy
    spacy.load("en_core_web_sm")


def warm_presidio():
    from analysis.pii_detection import detect_pii
    detect_pii([{"text": "My PAN is ABCDE1234F", "speaker": "customer"}])


def warm_diarization():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — diarization will be skipped at runtime")
    from whisperx.diarize import DiarizationPipeline
    DiarizationPipeline(use_auth_token=token, device="cpu")


def warm_ollama():
    from services.llm.client import check_ollama_health, preload_ollama_model
    health = check_ollama_health()
    if health.get("status") != "healthy":
        raise RuntimeError(f"Ollama not reachable: {health}")
    available = set(health.get("models", []))
    for model in ("qwen2.5:3b", "qwen3:8b"):
        if model not in available:
            raise RuntimeError(f"Missing Ollama model '{model}' — run: ollama pull {model}")
    preload_ollama_model("qwen2.5:3b", keep_alive="1m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", nargs="*", default=[], help="task names to skip")
    args = parser.parse_args()
    skip = set(args.skip)

    tasks = [("ollama", warm_ollama), ("spacy", warm_spacy), ("presidio", warm_presidio)]
    tasks += [(name, (lambda r=repo, t=task, n=name: warm_hf(n, r, t))) for name, repo, task in HF_MODELS]
    tasks += [
        ("detoxify", warm_detoxify),
        ("emotion2vec", warm_emotion2vec),
        ("whisperx", warm_whisperx),
        ("alignment", warm_alignment),
        ("diarization", warm_diarization),
    ]

    results = {}
    for name, fn in tasks:
        if name in skip:
            print(f"\n▶ {name} ... skipped")
            continue
        results[name] = _timed(name, fn)

    ok = [n for n, v in results.items() if v]
    bad = [n for n, v in results.items() if not v]

    print("\n" + "─" * 60)
    print(f"ready:   {', '.join(ok) if ok else '(none)'}")
    if bad:
        print(f"FAILED:  {', '.join(bad)}")
        print("\nThe pipeline still runs without these, but the matching features")
        print("degrade silently. Fix them before demoing.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
