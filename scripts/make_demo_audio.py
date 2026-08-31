"""Generate a synthetic two-speaker collections call for demos and smoke tests.

Produces an audio file the full pipeline can chew on without needing real customer
recordings. The script deliberately contains material for most detectors: financial
amounts and dates, a payment promise, an Aadhaar request with no purpose or consent
notice (DPDP), and a coercive threat (RBI Fair Practice Code).

Requires gTTS (`pip install gTTS`) and ffmpeg on PATH. Network access needed.

Usage:
    python scripts/make_demo_audio.py [-o data/raw_audio/demo_call.mp3]
"""

import shutil
import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent

# (speaker, accent tld, line). Different tlds give distinguishable voices so
# pyannote has two speakers to separate.
SCRIPT = [
    ("agent", "co.in", "Good afternoon, am I speaking with Mister Rakesh Sharma?"),
    ("customer", "com", "Yes, speaking. Who is this?"),
    ("agent", "co.in", "Sir, this is Priya calling from Trybank Financial Services regarding your personal loan account. This call is being recorded for quality and training purposes."),
    ("customer", "com", "Okay, go ahead."),
    ("agent", "co.in", "Sir, our records show your EMI of twelve thousand five hundred rupees was due on the fifteenth of March and it has not been received. Your total outstanding is now three lakh forty thousand rupees, including a late fee of eight hundred rupees."),
    ("customer", "com", "I know, I have been having some trouble at work. I lost a client last month."),
    ("agent", "co.in", "I understand sir. Can you tell me your Aadhaar number and your date of birth?"),
    ("customer", "com", "Why do you need my Aadhaar for this? I already gave it when I took the loan."),
    ("agent", "co.in", "Just give me the number sir, it is standard procedure."),
    ("customer", "com", "Fine. Look, can I get some more time? I can pay by the end of this month."),
    ("agent", "co.in", "Sir, if you do not pay by Friday we will have to send our recovery people to your residence and inform your employer about this default."),
    ("customer", "com", "That is not necessary. Please do not call my office."),
    ("agent", "co.in", "Then make the payment sir."),
    ("customer", "com", "Okay, okay. I will pay ten thousand rupees by Friday the twenty eighth, and the balance by the tenth of next month. I promise."),
    ("agent", "co.in", "Noted sir. I will record that as a payment commitment of ten thousand rupees by Friday."),
    ("customer", "com", "Yes. And please stop calling me every day, it is very stressful."),
    ("agent", "co.in", "Understood sir. Thank you for your time. Have a good day."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default=str(ROOT / "data" / "raw_audio" / "demo_call.mp3"))
    args = parser.parse_args()

    try:
        from gtts import gTTS
    except ImportError:
        print("gTTS not installed. Run: pip install gTTS")
        return 1

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH.")
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        parts = []
        for i, (speaker, tld, line) in enumerate(SCRIPT):
            part = tmpdir / f"{i:03d}.mp3"
            gTTS(text=line, lang="en", tld=tld, slow=False).save(str(part))
            parts.append(part)
            print(f"  [{i + 1}/{len(SCRIPT)}] {speaker}")

        listing = tmpdir / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-ar", "16000", "-ac", "1", str(out)],
            check=True, capture_output=True,
        )

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
