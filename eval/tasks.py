"""Evaluation tasks. Each returns (report_text, machine_readable_dict)."""

import os
import time
import json
import importlib
from pathlib import Path

from eval.corpus import (build, build_asr_variants, as_segments,
                         ENTITY_INTENT_CORPUS, ASR_TERM_CASES)
from eval.metrics import Score, table

ROOT = Path(__file__).resolve().parent.parent


# ── PII ───────────────────────────────────────────────────────────────────────

TRACKED = {"IN_AADHAAR", "IN_PAN", "IN_IFSC", "IN_UPI_ID", "IN_PHONE"}


def _normalise(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == "@")


def _score_samples(samples) -> tuple[dict, int, float]:
    """Run detect_pii over labelled samples and score against their labels."""
    from analysis.pii_detection import detect_pii

    # Re-index to a contiguous 0..n-1 id space so scoring never depends on the
    # detector's id convention.
    segments = as_segments(samples)
    for i, seg in enumerate(segments):
        seg["id"] = seg["segment_id"] = i
    ordered = list(samples)

    t0 = time.perf_counter()
    detected = detect_pii(segments, 0.5, "en")
    elapsed = time.perf_counter() - t0

    found: dict[int, set[tuple[str, str]]] = {}
    for e in detected:
        if e.entity_type in TRACKED:
            found.setdefault(e.segment_id, set()).add((e.entity_type, _normalise(e.text)))

    scores: dict[str, Score] = {}
    hard_fp = 0
    for idx, smp in enumerate(ordered):
        got = found.get(idx, set())
        want = {(t, _normalise(v)) for t, v in smp.expected}
        for t, v in want:
            scores.setdefault(t, Score())
            if (t, v) in got:
                scores[t].tp += 1
            else:
                scores[t].fn += 1
        for t, v in got - want:
            scores.setdefault(t, Score()).fp += 1
            if any(t == ft for ft, _ in smp.forbidden):
                hard_fp += 1
    return scores, hard_fp, elapsed


def eval_pii(per_type: int = 12) -> tuple[str, dict]:
    """Precision/recall of Indian-identifier PII detection, plus a Verhoeff ablation.

    Scored twice: once on cleanly written identifiers, once on the same identifiers
    rendered the way Whisper actually writes them (grouped, spaced, lowercased).
    The gap between the two is the gap between the synthetic corpus and real input.
    """
    samples = build(per_type=per_type)
    scores, verhoeff_fp, elapsed = _score_samples(samples)
    for t in TRACKED:
        scores.setdefault(t, Score())

    n_hard = sum(len(smp.forbidden) for smp in samples)
    report = table("PII detection — written form", scores)
    report += (
        f"\n\n  Verhoeff ablation: {n_hard} Aadhaar-shaped numbers with a broken check digit\n"
        f"    flagged as IN_AADHAAR anyway: {verhoeff_fp}/{n_hard} "
        f"({verhoeff_fp / n_hard * 100:.1f}% false-positive rate)\n"
        f"    Without checksum validation all {n_hard} would be flagged.\n"
        f"\n  {len(samples)} segments scored in {elapsed:.1f}s "
        f"({elapsed / len(samples) * 1000:.0f} ms/segment)"
    )

    asr = build_asr_variants()
    asr_scores, _, asr_elapsed = _score_samples(asr)
    report += "\n" + table("PII detection — ASR-rendered form (spaced / lowercased)", asr_scores)
    report += ("\n  Same identifiers as above, written as a Whisper transcript renders\n"
               "  them. This is the number that describes production behaviour.")

    return report, {
        "written_form": {t: {"precision": sc.precision, "recall": sc.recall,
                             "f1": sc.f1, "support": sc.support}
                         for t, sc in scores.items()},
        "asr_form": {t: {"precision": sc.precision, "recall": sc.recall,
                         "f1": sc.f1, "support": sc.support}
                     for t, sc in asr_scores.items()},
        "verhoeff_false_positives": verhoeff_fp,
        "verhoeff_hard_negatives": n_hard,
        "seconds": round(elapsed, 2),
        "segments": len(samples),
        "asr_segments": len(asr),
        "asr_seconds": round(asr_elapsed, 2),
    }


# ── COMPLIANCE ────────────────────────────────────────────────────────────────

# (call_type, agent utterances, checks expected to FAIL, checks expected to PASS)
COMPLIANCE_CASES = [
    ("collections",
     ["Give me your Aadhaar number and your date of birth right now."],
     {"dpdp_purpose_notice", "dpdp_consent_obtained", "dpdp_rights_notice", "dpdp_data_minimisation"},
     set()),
    ("kyc",
     ["To verify your identity for the purpose of KYC, do I have your consent to "
      "collect your PAN number? You may withdraw your consent at any time, and you "
      "can contact our grievance officer with any complaint."],
     set(),
     {"dpdp_purpose_notice", "dpdp_consent_obtained", "dpdp_rights_notice"}),
    ("collections",
     ["Good morning, this is Priya from Trybank Financial Services. This call is "
      "being recorded for quality purposes. I am calling about your loan account."],
     set(),
     {"caller_identification", "purpose_disclosure", "recording_consent"}),
    ("general",
     ["Give me your Aadhaar number and your date of birth right now."],
     set(), set()),  # DPDP does not apply to unregulated calls
]


def eval_compliance() -> tuple[str, dict]:
    """Does the rule engine fire on violations and stay quiet on clean calls?"""
    from analysis.compliance import run_compliance_checks

    scores = {"violation_detection": Score()}
    misses, spurious = [], []

    for call_type, utterances, want_fail, want_pass in COMPLIANCE_CASES:
        segs = [{"id": i, "segment_id": i, "text": t, "speaker": "agent",
                 "start": float(i * 10), "end": float(i * 10 + 9), "confidence": 1.0}
                for i, t in enumerate(utterances)]
        checks = run_compliance_checks(segs, call_type, "en")
        failed = {c.check_name for c in checks if not c.passed}
        passed = {c.check_name for c in checks if c.passed}

        for name in want_fail:
            if name in failed:
                scores["violation_detection"].tp += 1
            else:
                scores["violation_detection"].fn += 1
                misses.append(f"{call_type}: expected {name} to FAIL")

        for name in want_pass:
            if name in failed:
                scores["violation_detection"].fp += 1
                spurious.append(f"{call_type}: {name} failed but should pass")
            elif name not in passed:
                misses.append(f"{call_type}: {name} did not run at all")

        if call_type == "general" and any(n.startswith("dpdp_") for n in failed):
            scores["violation_detection"].fp += 1
            spurious.append("general: DPDP checks fired on an unregulated call")

    report = table("Compliance rule engine", scores)
    if misses:
        report += "\n\n  Missed:\n" + "\n".join(f"    - {m}" for m in misses)
    if spurious:
        report += "\n\n  Spurious:\n" + "\n".join(f"    - {m}" for m in spurious)
    if not misses and not spurious:
        report += "\n\n  No misses, no spurious violations."
    return report, {
        "precision": scores["violation_detection"].precision,
        "recall": scores["violation_detection"].recall,
        "misses": misses, "spurious": spurious,
    }


# ── SCAM PREFILTER ────────────────────────────────────────────────────────────

def eval_scam_prefilter(record_path: Path | None = None) -> tuple[str, dict]:
    """What the cheap keyword gate in front of the zero-shot NLI model costs.

    It trades recall for latency on a fraud detector, so the cost must be measured
    rather than assumed.
    """
    import analysis.fraud_detection as F

    if record_path is None:
        records = sorted(ROOT.glob("data/processed/*_record.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if not records:
            return "\nScam prefilter — skipped (no processed calls found)", {"skipped": True}
        record_path = records[0]

    segs = json.loads(record_path.read_text())["transcript_segments"]

    os.environ["FINVOICE_SCAM_PREFILTER"] = "1"
    importlib.reload(F)
    t0 = time.perf_counter(); on = F.detect_scam_zero_shot(segs); t_on = time.perf_counter() - t0
    admitted = sum(1 for s in segs
                   if len(s.get("text", "").strip()) > 10 and F._is_scam_candidate(s["text"]))

    os.environ["FINVOICE_SCAM_PREFILTER"] = "0"
    importlib.reload(F)
    t0 = time.perf_counter(); off = F.detect_scam_zero_shot(segs); t_off = time.perf_counter() - t0

    os.environ["FINVOICE_SCAM_PREFILTER"] = "1"
    importlib.reload(F)

    on_types = {s.signal_type for s in on}
    off_types = {s.signal_type for s in off}
    lost = off_types - on_types

    report = (
        f"\nScam prefilter (zero-shot NLI gate)\n"
        f"───────────────────────────────────\n"
        f"  corpus: {record_path.name}, {len(segs)} segments\n"
        f"  admitted by gate:   {admitted}/{len(segs)} segments\n"
        f"  with prefilter:     {t_on:6.1f}s  -> {len(on)} signals {sorted(on_types)}\n"
        f"  without prefilter:  {t_off:6.1f}s  -> {len(off)} signals {sorted(off_types)}\n"
        f"  speedup:            {t_off / max(t_on, 0.01):.1f}x\n"
        f"  signals lost:       {sorted(lost) if lost else 'none'}"
    )
    return report, {
        "admitted": admitted, "total_segments": len(segs),
        "seconds_with": round(t_on, 2), "seconds_without": round(t_off, 2),
        "speedup": round(t_off / max(t_on, 0.01), 2),
        "signals_lost": sorted(lost),
    }


# ── LATENCY ───────────────────────────────────────────────────────────────────

def eval_latency() -> tuple[str, dict]:
    """Per-stage and per-analyzer wall time from real processed calls."""
    records = sorted(ROOT.glob("data/processed/*_record.json"))
    runs = []
    for p in records:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        t = d.get("pipeline_timings") or {}
        stage = {k: v for k, v in t.items() if not k.startswith("Stage 4/")}
        if not stage or not d.get("duration_seconds"):
            continue
        runs.append((d["call_id"], d["duration_seconds"], stage,
                     {k.split("/", 1)[1]: v for k, v in t.items() if k.startswith("Stage 4/")},
                     d.get("degradations", [])))

    if not runs:
        return "\nLatency — skipped (no processed calls with timings)", {"skipped": True}

    lines = ["\nLatency", "───────"]
    payload = []
    for call_id, dur, stage, analyzers, degs in runs:
        total = sum(stage.values())
        lines.append(f"  {call_id}  {dur:6.1f}s audio -> {total:6.1f}s  "
                     f"({total / dur:.2f}x realtime)"
                     + (f"   [degraded: {', '.join(d['component'] for d in degs)}]" if degs else ""))
        for k, v in sorted(stage.items(), key=lambda x: -x[1])[:3]:
            lines.append(f"       {k:34} {v:6.1f}s")
        if analyzers:
            top = sorted(analyzers.items(), key=lambda x: -x[1])[:3]
            lines.append("       slowest analyzers: "
                         + ", ".join(f"{k}={v}s" for k, v in top))
        payload.append({"call_id": call_id, "audio_seconds": dur,
                        "pipeline_seconds": round(total, 1),
                        "realtime_factor": round(total / dur, 2),
                        "analyzers": analyzers,
                        "degradations": [d["component"] for d in degs]})

    factors = [p["realtime_factor"] for p in payload]
    lines.append(f"\n  {len(runs)} run(s), realtime factor "
                 f"min {min(factors):.2f} / median {sorted(factors)[len(factors) // 2]:.2f} / max {max(factors):.2f}")
    return "\n".join(lines), {"runs": payload}


# ── ENTITY EXTRACTION ─────────────────────────────────────────────────────────

def _num(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit() or ch == ".").rstrip(".") or ""


def eval_entities() -> tuple[str, dict]:
    """Recall of the Layer-1 (regex + spaCy) financial entity extractor.

    Scored on amount *value* rather than entity_type, because the pipeline uses
    several near-synonymous type names (payment_amount, emi_amount, currency_amount)
    and what matters downstream is whether the figure was captured at all.
    """
    from analysis.intelligence import extract_all_entities_layer1

    segments = [
        {"id": i, "segment_id": i, "text": u.text, "speaker": u.speaker,
         "start": float(i * 5), "end": float(i * 5 + 4), "confidence": 1.0}
        for i, u in enumerate(ENTITY_INTENT_CORPUS)
    ]
    t0 = time.perf_counter()
    entities = extract_all_entities_layer1(segments, "en")
    elapsed = time.perf_counter() - t0

    by_seg: dict[int, set[str]] = {}
    for e in entities:
        seg = getattr(e, "segment_id", None)
        val = _num(getattr(e, "value", "") or "") or _num(getattr(e, "raw_text", "") or "")
        if val:
            by_seg.setdefault(seg, set()).add(val.lstrip("0") or "0")

    score = Score()
    misses = []
    for i, u in enumerate(ENTITY_INTENT_CORPUS):
        got = by_seg.get(i, set())
        for etype, want in u.entities:
            want_n = _num(want).lstrip("0") or "0"
            if any(want_n == g or want_n in g for g in got):
                score.tp += 1
            else:
                score.fn += 1
                misses.append(f"{etype}={want}  in: {u.text[:64]}"
                              + (f"   [{u.note}]" if u.note else ""))

    report = table("Entity extraction — amounts and rates (Layer 1)", {"amounts": score})
    if misses:
        report += "\n\n  Missed:\n" + "\n".join(f"    - {m}" for m in misses)
    report += f"\n\n  {len(ENTITY_INTENT_CORPUS)} utterances in {elapsed:.1f}s"
    return report, {"recall": score.recall, "tp": score.tp, "fn": score.fn, "misses": misses}


# ── INTENT CLASSIFICATION ─────────────────────────────────────────────────────

def eval_intents() -> tuple[str, dict]:
    """Accuracy of the keyword-fallback intent classifier.

    The LLM path is exercised in the live pipeline; this scores the deterministic
    fallback that runs whenever the LLM is skipped, times out, or is unavailable —
    which, before the model-routing fix, was every non-English call.
    """
    from pipeline.orchestrator import _keyword_intent_fallback

    correct, total = 0, 0
    wrong = []
    t0 = time.perf_counter()
    for u in ENTITY_INTENT_CORPUS:
        if not u.intent:
            continue
        total += 1
        intent, _conf = _keyword_intent_fallback(u.text, "collections")
        got = intent.value if hasattr(intent, "value") else str(intent)
        if got == u.intent:
            correct += 1
        else:
            wrong.append(f"{u.intent:18} -> {got:18} | {u.text[:52]}")
    elapsed = time.perf_counter() - t0

    acc = correct / total if total else 0.0
    report = (f"\nIntent classification — keyword fallback\n"
              f"────────────────────────────────────────\n"
              f"  accuracy {acc:5.3f}  ({correct}/{total})   {elapsed * 1000:.0f} ms total")
    if wrong:
        report += "\n\n  Misclassified (expected -> got):\n" + "\n".join(
            f"    - {w}" for w in wrong)
    report += ("\n\n  This is the deterministic path used when the LLM is unavailable.\n"
               "  It is a floor, not the headline number.")
    return report, {"accuracy": acc, "correct": correct, "total": total, "wrong": wrong}


# ── ASR FINANCIAL-TERM CORRECTION ─────────────────────────────────────────────

def eval_asr_terms() -> tuple[str, dict]:
    """Does the correction layer fix the ASR errors it exists for, without collateral?

    Scored both ways: recall on terms that must be recovered, and a check that
    ordinary language is left alone. A correction dictionary that fixes "me" -> "EMI"
    everywhere would score perfect recall and destroy the transcript.
    """
    from services.asr.transcriber import _apply_corrections, load_correction_dictionary

    corrections = load_correction_dictionary()

    def _corrected(text: str) -> str:
        return _apply_corrections([{"text": text}], corrections)[0]["text"]

    fixed = Score()
    collateral = Score()
    missed, damaged = [], []

    for case in ASR_TERM_CASES:
        out = _corrected(case.raw)
        if case.must_not_change:
            if case.expect.lower() in out.lower():
                collateral.tp += 1
            else:
                collateral.fp += 1
                damaged.append(f"{case.raw[:56]!r} -> {out[:56]!r}")
        else:
            if case.expect.lower() in out.lower():
                fixed.tp += 1
            else:
                fixed.fn += 1
                missed.append(f"want {case.expect!r} in: {out[:64]}"
                              + (f"   [{case.note}]" if case.note else ""))

    n_pos = fixed.tp + fixed.fn
    n_neg = collateral.tp + collateral.fp
    report = (
        "\nASR financial-term correction\n"
        "─────────────────────────────\n"
        f"  terms recovered:      {fixed.tp}/{n_pos}  (recall {fixed.recall:.3f})\n"
        f"  ordinary text intact: {collateral.tp}/{n_neg}"
    )
    if missed:
        report += "\n\n  Not corrected:\n" + "\n".join(f"    - {m}" for m in missed)
    if damaged:
        report += "\n\n  COLLATERAL DAMAGE:\n" + "\n".join(f"    - {d}" for d in damaged)
    if not missed and not damaged:
        report += "\n\n  All observed ASR errors corrected; no ordinary text altered."
    return report, {
        "recall": fixed.recall, "recovered": fixed.tp, "total_terms": n_pos,
        "intact": collateral.tp, "total_negatives": n_neg,
        "missed": missed, "damaged": damaged,
    }
