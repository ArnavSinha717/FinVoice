# FinVoice

**Financial call intelligence — raw audio in, structured, audit-ready, ML-trainable data out.**

FinVoice takes a recorded bank or lender call and produces a single structured
`CallRecord`: a diarized transcript, extracted financial entities and obligations,
intent labels, an RBI compliance scorecard, fraud and audio-tamper signals, PII
detections with Indian identifier support, and per-speaker emotion trajectories.

Everything runs locally. Transcription, emotion, sentiment, NER and LLM extraction
all execute on-device against Ollama and a 6 GB GPU — no call audio leaves the machine.

---

## Pipeline

| Stage | What happens | Tools |
|-------|--------------|-------|
| **1 · Ingestion** | Format normalization to 16 kHz mono, optional trimming | FFmpeg |
| **2 · Quality & cleanup** | SNR / clipping / spectral scoring, VAD, dead-air and hold-music removal | Silero VAD, librosa |
| **3 · Transcription** | ASR with word-level timestamps, speaker diarization, financial term correction | WhisperX `large-v3-turbo` (INT8), pyannote 3.1 |
| **4 · Understanding** | Entities, intents, obligations, sentiment, emotion, compliance, fraud, tamper, PII | FinBERT, spaCy, IndicNER, emotion2vec, mDeBERTa, Presidio, Parselmouth, Qwen |
| **5 · Output** | `CallRecord` JSON, CSV/Parquet/JSONL exports, training-pair datasets, audit trail | Pydantic, PyArrow |

Stage 4 runs CPU analysis and GPU emotion extraction concurrently. WhisperX is
loaded, used, and unloaded around the LLM stages so the whole pipeline fits in 6 GB
of VRAM.

## Requirements

- **Python 3.12** and **Node.js 20+**
- **NVIDIA GPU with ~6 GB VRAM** and a working CUDA 12.x driver
- **[Ollama](https://ollama.com)** running locally
- **FFmpeg** on `PATH`

## Setup

```bash
git clone https://github.com/<you>/FinVoice.git
cd FinVoice

# 1. Python environment (~8 GB installed: PyTorch + CUDA libraries)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt

# 2. Local LLMs — qwen2.5:3b handles English, qwen3:8b handles other languages
ollama pull qwen2.5:3b
ollama pull qwen3:8b

# 3. Configuration
cp .env.example .env    # then add your HF_TOKEN — see below

# 4. Frontend
cd frontend && npm install && cd ..
```

### The HuggingFace token matters

Speaker diarization uses `pyannote/speaker-diarization-3.1`, which is gated. Without
`HF_TOKEN` set the pipeline still completes, but it logs `No HF token — skipping
diarization` and every speaker-derived feature degrades: agent/customer talk share,
per-speaker emotion breakdown, and the agent-domination and third-party-disclosure
compliance checks. Create a token, accept the terms on the
[diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) and
[segmentation](https://huggingface.co/pyannote/segmentation-3.0) model pages, then
put it in `.env`.

### Warm the models before demoing

On first run the pipeline downloads several GB of weights (WhisperX, pyannote, FinBERT,
emotion2vec, mDeBERTa, MuRIL, Detoxify). It does this *inside* a pipeline stage, and
progress is only reported at stage boundaries — so a cold machine looks frozen on
"Stage 4A" for as long as the download takes. Worse, a stalled HuggingFace connection
blocks the stage indefinitely rather than failing.

Do this once, ahead of time, on the machine that will run the demo:

```bash
python scripts/warmup_models.py
```

Set `HF_HUB_DOWNLOAD_TIMEOUT=30` in `.env` so a stalled download fails fast instead of
hanging.

## Running

```bash
# Terminal 1 — API on :8000
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 2 — dashboard on :3000
cd frontend && npm run dev
```

Open http://localhost:3000. The dashboard proxies `/api/*` to port 8000.

A fresh clone has no processed calls, so every view starts empty. To populate the
dashboard immediately with the bundled sample call:

```bash
python scripts/seed_demo.py           # add it
python scripts/seed_demo.py --clear   # remove it
```

Then upload an audio file from the dashboard, or:

```bash
curl -F "file=@call.mp3" -F "call_type=collections" http://localhost:8000/api/process
```

`/api/process` returns a `call_id` immediately and runs the pipeline in a background
thread. Poll `/api/calls/{call_id}/progress` for stage-by-stage status. Calls are
processed **one at a time** — a second upload queues behind the first rather than
competing for the same 6 GB of VRAM.

## API

`GET /api/health` reports GPU, Ollama and WhisperX status — check it first.

| Endpoint | Purpose |
|----------|---------|
| `POST /api/process` | Upload audio, start the pipeline |
| `GET /api/calls/{id}/progress` | Live stage progress |
| `GET /api/results/{id}` | Full `CallRecord` |
| `GET /api/calls` · `/api/stats` · `/api/review-queue` | Dashboard data |
| `GET /api/calls/{id}/{transcript,entities,compliance,emotions,intents}` | Per-section views |
| `GET /api/calls/{id}/transcript/masked` | PII-masked transcript |
| `GET /api/export/{csv,parquet,jsonl,training_*}` | Bulk and training-pair exports |
| `POST /api/audit/query` | Ask questions across processed calls (local LLM) |
| `POST /api/compliance/reason` | LLM explanation of a compliance finding |

## Compliance coverage

Rule-based checks against RBI **Fair Practice Code** (opening disclosure, prohibited
threat/coercion language, call-timing under Section 8(c), third-party disclosure
under 8(d), agent domination, ignored call-end requests) and **KYC Master Direction**
identity-verification requirements, plus SEC Regulation FD checks for earnings calls.

Severity is weighted by call type — a violation on a KYC or collections call escalates
faster than the same violation on a general call.

## PII detection

Microsoft Presidio with custom Indian recognizers:

- **Aadhaar** — 12 digits, validated with the **Verhoeff checksum** to suppress false positives
- **PAN** — `ABCDE1234F`
- **UPI ID** — `name@bank`
- **IFSC** — 4 letters + `0` + 6 digits
- Indian phone numbers, plus Presidio's built-in `PERSON` / `EMAIL_ADDRESS` / `DATE_TIME`

Every detection carries a masked form, so `/api/calls/{id}/transcript/masked` returns a
redacted transcript suitable for sharing.

## Languages

Whisper auto-detects the language; 21 are named for LLM prompting, with hand-written
few-shot examples for Hindi and Tamil code-switching. Per-segment language tagging
flags code-switched calls. Curated financial vocabularies ship for English, Hindi and
Tamil (`data/vocab/`).

## Evaluation

```bash
python scripts/evaluate.py                 # all tasks
python scripts/evaluate.py pii compliance  # selected
python scripts/evaluate.py --json out.json
```

Four tasks:

- **entities** — recall of the Layer-1 financial entity extractor, including
  amounts stated in words ("ten thousand rupees", "paanch hazaar rupaye").
- **intents** — accuracy of the deterministic keyword classifier, the path used
  whenever the LLM is unavailable.
- **pii** — precision/recall of the Indian identifier recognizers against a
  generated corpus. Ground truth is constructed rather than hand-labelled: Aadhaar
  numbers are built with a valid Verhoeff check digit, so the labels are exact, and
  the corpus includes Aadhaar-shaped numbers with broken check digits as hard
  negatives. Scored twice — once on cleanly written identifiers, once on the same
  identifiers rendered the way a Whisper transcript actually writes them (grouped,
  dictated digit-by-digit, lowercased). The second number is the one that describes
  production behaviour.
- **compliance** — does the rule engine fire on violations and stay quiet on clean
  calls, including that DPDP checks do not fire on unregulated call types.
- **prefilter** — what the keyword gate in front of the zero-shot NLI model costs.
  It trades recall for latency on a *fraud* detector, so the cost is measured
  rather than assumed.
- **latency** — per-stage and per-analyzer wall time from real processed calls.

The harness reports any run that completed with components disabled. An analyzer
that silently did not run produces the same empty output as one that ran and found
nothing; scoring the first as though it were the second is how a pipeline comes to
look better than it is. Every `CallRecord` carries a `degradations` list for this.

## Guardrails

Several behaviours exist because the model cannot be trusted with them:

**Numbers are handled in code.** The summary model is instructed never to convert
currency or compute totals, and `pipeline/numeric_guard.py` enforces it afterwards:
invented currency conversions are removed, figures that trace to neither an
extracted entity nor the transcript are replaced with `[unverified]`, and real
totals are computed from `financial_entities` into `derived_totals`. This is not
hypothetical — the model rendered 3,40,000 rupees as "approximately $52,808 USD",
and separately invented a remaining balance of 249,200 where the arithmetic gives
330,000. `numeric_audit` on each record says what was stripped.

**Prompt scaffolding never reaches the UI.** Small models sometimes echo the
prompt's section headings instead of filling them in. Those are detected and, if
nothing of substance remains, the summary is rebuilt from extracted data in code.

**Degradations are recorded, not hidden.** Every `CallRecord` carries a
`degradations` list naming components that did not run and what that costs. An
analyzer that silently no-ops produces the same empty output as one that ran and
found nothing, and treating the first as the second is how a pipeline comes to look
better than it is.

**Prohibited conduct is judged by meaning.** Literal phrase matching missed
"we will send our recovery people to your residence and inform your employer" —
a clear RBI Fair Practice Code violation — because the vocabulary held slightly
different wording. `analysis/prohibited_semantic.py` runs a constrained LLM judge
at temperature 0 alongside the keyword engine. Without diarization, findings are
marked speaker-unattributed rather than asserted against the agent.

## Authentication

Set `FINVOICE_API_KEY` and every endpoint except `/api/health` requires an
`X-API-Key` header. Leave it unset and the API is open, and says so loudly at
startup. Given the API serves transcripts, PII and raw call audio, enable it
anywhere the port is reachable.

## Tests

```bash
pytest tests/ -q
```

150 tests covering sentiment, intent classification, NER, emotion summarization,
Some tests download models on first run.

## Project layout

```
app.py                  FastAPI application — 25 routes
config/schemas.py       Pydantic models; CallRecord is the master output contract
pipeline/
  orchestrator.py       The 5-stage pipeline
  output_generator.py   CSV / Parquet / JSONL / training-pair exports
services/
  audio/                Normalization, quality scoring, cleanup
  asr/                  WhisperX lifecycle and transcription
  emotion/              emotion2vec analysis
  llm/                  Ollama + Instructor client
analysis/               Entities, intents, compliance, fraud, PII, tamper, sentiment
scripts/                Fine-tuning, vocabulary building, batch processing, demo seed
frontend/               Next.js 15 dashboard
eval/                   Evaluation corpus, metrics, and scoring tasks
pipeline/numeric_guard.py   Rejects model-invented figures; computes totals in code
pipeline/degradations.py    Records which analyzers did not run
pipeline/call_index.py      SQLite index behind /api/stats, /api/calls, /api/review-queue
config/auth.py              Optional API-key authentication
```

## Troubleshooting

**`address already in use` on :8000** — something else is bound to the port. Check with
`ss -tlnp | grep :8000` and stop it, or run uvicorn on another port and set
`NEXT_PUBLIC_BACKEND_URL` for the frontend.

**Transcription returns 0 segments** — usually a stale WhisperX/VAD state. `GET /api/health`
shows whether the model is loaded; restart the API to force a clean preload.

**`funasr`/`detoxify` warnings at startup** — emotion and toxicity analysis are optional
and degrade gracefully. Reinstall from `requirements-gpu.txt` to enable them.
