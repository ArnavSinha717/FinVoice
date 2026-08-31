"""Run the FinVoice evaluation harness.

Measures the components that make quantitative claims:

  pii         precision/recall of the Indian financial identifier recognizers,
              plus an ablation showing what Aadhaar Verhoeff validation buys
  compliance  does the rule engine fire on violations and stay quiet otherwise
  prefilter   what the cheap gate in front of the zero-shot NLI model costs
  latency     per-stage and per-analyzer wall time from real processed calls

Usage:
    python scripts/evaluate.py                    # everything
    python scripts/evaluate.py pii compliance     # selected tasks
    python scripts/evaluate.py --json out.json    # also write machine-readable results

A note on honesty: several analyzers degrade silently when a model or token is
missing. Any run that produced degradations is reported here, because an absent
result is not the same as a negative result.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import tasks  # noqa: E402

TASKS = {
    "pii": tasks.eval_pii,
    "entities": tasks.eval_entities,
    "intents": tasks.eval_intents,
    "asr_terms": tasks.eval_asr_terms,
    "compliance": tasks.eval_compliance,
    "prefilter": tasks.eval_scam_prefilter,
    "latency": tasks.eval_latency,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", choices=list(TASKS) + [], default=None,
                    help="tasks to run (default: all)")
    ap.add_argument("--json", metavar="PATH", help="write machine-readable results here")
    args = ap.parse_args()

    selected = args.tasks or list(TASKS)
    results: dict[str, dict] = {}

    print("=" * 68)
    print("FinVoice evaluation")
    print("=" * 68)

    for name in selected:
        try:
            report, payload = TASKS[name]()
            print(report)
            results[name] = payload
        except Exception as e:
            print(f"\n{name} — FAILED: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    # Surface degraded runs: metrics computed over a pipeline that silently
    # skipped an analyzer describe the fallback, not the system.
    degraded = set()
    for run in results.get("latency", {}).get("runs", []):
        degraded.update(run.get("degradations", []))
    if degraded:
        print("\n" + "!" * 68)
        print("Some processed calls ran with components disabled: "
              + ", ".join(sorted(degraded)))
        print("Numbers derived from those runs measure the fallback path.")
        print("!" * 68)

    print()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
