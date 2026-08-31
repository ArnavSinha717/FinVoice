"""Seed the dashboard with the bundled sample call record.

A fresh clone has no processed calls, so every dashboard view renders empty until
a real pipeline run finishes (which needs a GPU, Ollama, and several GB of model
downloads). This copies data/sample/sample_output.json into data/processed/ so the
UI has something to show immediately.

Usage:
    python scripts/seed_demo.py            # add the sample call
    python scripts/seed_demo.py --clear    # remove it again
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "sample_output.json"
PROCESSED = ROOT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear", action="store_true", help="remove the seeded record")
    args = parser.parse_args()

    if not SAMPLE.exists():
        print(f"Sample not found: {SAMPLE}", file=sys.stderr)
        return 1

    record = json.loads(SAMPLE.read_text())
    call_id = record.get("call_id")
    if not call_id:
        print("Sample record has no call_id", file=sys.stderr)
        return 1

    target = PROCESSED / f"{call_id}_record.json"

    if args.clear:
        if target.exists():
            target.unlink()
            print(f"Removed {target.relative_to(ROOT)}")
        else:
            print("Nothing to remove.")
        return 0

    # Fields added to CallRecord after this sample was generated. Filling in the
    # defaults keeps the seeded record shape-compatible with the current schema.
    record.setdefault("has_code_switching", False)
    record.setdefault("language_distribution", {record.get("detected_language", "en"): len(record.get("transcript_segments", []))})
    record.setdefault("sentiment_context", {})
    record.setdefault("emotion_transitions", [])
    record.setdefault("escalation_moments", [])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2))
    print(f"Seeded {target.relative_to(ROOT)} (call_id={call_id})")
    print("The dashboard will now show one call. Remove it with --clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
