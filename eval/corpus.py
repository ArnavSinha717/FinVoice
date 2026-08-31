"""Synthetic labelled corpus for PII evaluation.

Ground truth is *generated*, not hand-labelled: identifiers are constructed to a
known format (Aadhaar with a valid Verhoeff check digit, PAN, UPI, IFSC, phone)
and embedded at known character offsets in call-like sentences. That makes the
span labels exact and the corpus reproducible.

The hard negatives matter as much as the positives. A 12-digit number that fails
the Verhoeff check, or a PAN-shaped string with the wrong letter/digit layout, is
the case a naive regex gets wrong — so the corpus can measure what the checksum
actually buys.
"""

import random
from dataclasses import dataclass, field

# Verhoeff tables, mirrored from analysis/pii_detection.py so the corpus does not
# depend on the code under test to build its own ground truth.
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_check_digit(payload: str) -> str:
    c = 0
    for i, digit in enumerate(reversed(payload)):
        c = _D[c][_P[(i + 1) % 8][int(digit)]]
    return str(_INV[c])


def valid_aadhaar(rng: random.Random) -> str:
    """A 12-digit Aadhaar whose final digit is a correct Verhoeff check digit."""
    payload = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    return payload + _verhoeff_check_digit(payload)


def invalid_aadhaar(rng: random.Random) -> str:
    """12 digits that look like an Aadhaar but fail the Verhoeff check."""
    good = valid_aadhaar(rng)
    last = int(good[-1])
    return good[:-1] + str((last + rng.randint(1, 9)) % 10)


def valid_pan(rng: random.Random) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ("".join(rng.choice(L) for _ in range(5))
            + "".join(str(rng.randint(0, 9)) for _ in range(4))
            + rng.choice(L))


def valid_ifsc(rng: random.Random) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(L) for _ in range(4)) + "0" + "".join(str(rng.randint(0, 9)) for _ in range(6))


def valid_upi(rng: random.Random) -> str:
    names = ["rakesh", "priya", "amit", "sunita", "vikram", "meera", "arjun", "kavya"]
    banks = ["okhdfcbank", "oksbi", "okicici", "okaxis", "ybl", "paytm", "upi"]
    return f"{rng.choice(names)}{rng.randint(10, 999)}@{rng.choice(banks)}"


def valid_phone(rng: random.Random) -> str:
    return "+91" + str(rng.choice([6, 7, 8, 9])) + "".join(str(rng.randint(0, 9)) for _ in range(9))


# Sentence frames. {} is where the identifier goes; each frame reads like real
# agent/customer speech so Presidio's context features behave realistically.
_FRAMES = {
    "IN_AADHAAR": [
        "Sir, could you confirm your Aadhaar number {} for verification?",
        "My Aadhaar is {} if that helps.",
        "I have your Aadhaar on file as {}, is that correct?",
    ],
    "IN_PAN": [
        "Please share your PAN card number {} for the tax records.",
        "The PAN linked to this account is {}.",
        "I am reading your PAN as {}, please confirm.",
    ],
    "IN_IFSC": [
        "The branch IFSC code is {} for that transfer.",
        "Use IFSC {} when you set up the mandate.",
    ],
    "IN_UPI_ID": [
        "You can pay to the UPI ID {} before Friday.",
        "My UPI is {} for the refund.",
    ],
    "IN_PHONE": [
        "We will call you back on {} tomorrow morning.",
        "Is {} still your registered mobile number?",
    ],
}

# Sentences containing NO PII of the target types. Some deliberately contain
# numbers, so a length-only heuristic would trip on them.
_NEGATIVES = [
    "Your EMI of 12,500 rupees was due on the fifteenth of March.",
    "The outstanding balance is 3,40,000 rupees including a late fee of 800 rupees.",
    "I lost a client last month and things have been difficult.",
    "Please do not call my office about this matter.",
    "The interest rate on this product is 10.5 percent per annum.",
    "Your reference number for this conversation is 4429301887 for tracking.",
    "I will pay ten thousand rupees by Friday the twenty eighth.",
    "This call is being recorded for quality and training purposes.",
    "The loan tenure is 36 months starting from April.",
    "Our office is open from 9 in the morning until 6 in the evening.",
]


@dataclass
class Sample:
    segment_id: int
    text: str
    # (entity_type, exact_text) pairs that a detector SHOULD find
    expected: list[tuple[str, str]] = field(default_factory=list)
    # (entity_type, exact_text) pairs that a detector should NOT find
    forbidden: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""


def build(seed: int = 20260830, per_type: int = 12) -> list[Sample]:
    """Build the labelled corpus. Deterministic for a given seed."""
    rng = random.Random(seed)
    samples: list[Sample] = []
    sid = 0

    generators = {
        "IN_AADHAAR": valid_aadhaar,
        "IN_PAN": valid_pan,
        "IN_IFSC": valid_ifsc,
        "IN_UPI_ID": valid_upi,
        "IN_PHONE": valid_phone,
    }

    for etype, gen in generators.items():
        for i in range(per_type):
            value = gen(rng)
            frame = _FRAMES[etype][i % len(_FRAMES[etype])]
            samples.append(Sample(sid, frame.format(value), expected=[(etype, value)]))
            sid += 1

    # Hard negatives: Aadhaar-shaped numbers with a broken check digit. A regex
    # without checksum validation flags these; a correct implementation does not.
    for i in range(per_type):
        bad = invalid_aadhaar(rng)
        frame = _FRAMES["IN_AADHAAR"][i % len(_FRAMES["IN_AADHAAR"])]
        samples.append(Sample(
            sid, frame.format(bad), forbidden=[("IN_AADHAAR", bad)],
            note="12 digits, invalid Verhoeff check digit",
        ))
        sid += 1

    # Plain negatives — no identifiers of any tracked type.
    for text in _NEGATIVES:
        samples.append(Sample(sid, text, note="no PII of tracked types"))
        sid += 1

    return samples


def as_segments(samples: list[Sample]) -> list[dict]:
    """Corpus in the shape detect_pii() expects."""
    return [
        {"id": s.segment_id, "segment_id": s.segment_id, "text": s.text,
         "speaker": "agent" if s.segment_id % 2 == 0 else "customer",
         "start": float(s.segment_id * 5), "end": float(s.segment_id * 5 + 4),
         "confidence": 1.0}
        for s in samples
    ]


# ── ASR-realistic variants ────────────────────────────────────────────────────
# The pipeline's real input is Whisper output, not clean typed text. Whisper
# renders spoken identifiers with spaces, in groups, sometimes as words, and
# often lowercases letters. A recognizer that only matches the canonical written
# format will score perfectly on synthetic text and then miss most real calls.
# These cases exist to measure that gap rather than hide it.

_ASR_FRAMES = [
    "My Aadhaar number is {}.",
    "Let me read out my Aadhaar, it is {}.",
    "Sir the Aadhaar on record is {}.",
]


def _spaced_4(digits: str) -> str:
    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"


def _spaced_all(digits: str) -> str:
    return " ".join(digits)


def build_asr_variants(seed: int = 7, per_form: int = 8) -> list["Sample"]:
    """Aadhaar and PAN as an ASR system actually renders them."""
    rng = random.Random(seed)
    out: list[Sample] = []
    sid = 10_000

    for i in range(per_form):
        d = valid_aadhaar(rng)
        # Whisper's most common rendering: grouped in fours.
        out.append(Sample(sid, _ASR_FRAMES[i % len(_ASR_FRAMES)].format(_spaced_4(d)),
                          expected=[("IN_AADHAAR", _spaced_4(d))],
                          note="ASR: Aadhaar grouped in fours")); sid += 1
        # Digit-by-digit, which happens when the speaker dictates slowly.
        out.append(Sample(sid, _ASR_FRAMES[i % len(_ASR_FRAMES)].format(_spaced_all(d)),
                          expected=[("IN_AADHAAR", _spaced_all(d))],
                          note="ASR: Aadhaar digit-by-digit")); sid += 1

    for i in range(per_form):
        pan = valid_pan(rng)
        out.append(Sample(sid, f"My PAN is {pan.lower()} for the records.",
                          expected=[("IN_PAN", pan.lower())],
                          note="ASR: PAN lowercased")); sid += 1
        spaced = " ".join(pan)
        out.append(Sample(sid, f"The PAN reads {spaced}.",
                          expected=[("IN_PAN", spaced)],
                          note="ASR: PAN letter-by-letter")); sid += 1

    return out


# ── Entity and intent corpus ──────────────────────────────────────────────────
# Hand-written utterances in the style of Indian collections and KYC calls, each
# labelled with the financial entities and the intent a correct pipeline should
# produce. Kept small and explicit: the point is a reproducible regression signal
# on the extraction layer, not a benchmark-sized dataset.

@dataclass
class LabelledUtterance:
    text: str
    speaker: str
    # entity_type -> normalised value fragment that must appear
    entities: list[tuple[str, str]] = field(default_factory=list)
    intent: str | None = None
    note: str = ""


ENTITY_INTENT_CORPUS: list[LabelledUtterance] = [
    LabelledUtterance(
        "Sir, your EMI of 12,500 rupees was due on the fifteenth of March.",
        "agent", [("payment_amount", "12500")], "info_request"),
    LabelledUtterance(
        "Your total outstanding is now 3,40,000 rupees including a late fee of 800 rupees.",
        "agent", [("payment_amount", "340000"), ("payment_amount", "800")], "info_request",
        note="Indian lakh grouping"),
    LabelledUtterance(
        "I will pay ten thousand rupees by Friday the twenty eighth.",
        "customer", [("payment_amount", "10000")], "payment_promise",
        note="amount spelled out in words"),
    LabelledUtterance(
        "Main Friday tak paanch hazaar rupaye jama kar dunga.",
        "customer", [("payment_amount", "5000")], "payment_promise",
        note="romanised Hindi, hazaar = thousand"),
    LabelledUtterance(
        "The interest rate on this loan is 10.5 percent per annum.",
        "agent", [("interest_rate", "10.5")], "info_request"),
    LabelledUtterance(
        "Can you please confirm your PAN number for the records?",
        "agent", [], "info_request"),
    LabelledUtterance(
        "I do not accept this charge, it is completely wrong.",
        "customer", [], "dispute"),
    LabelledUtterance(
        "I want to speak to your manager right now.",
        "customer", [], "escalation"),
    LabelledUtterance(
        "Yes, I agree to the settlement terms you described.",
        "customer", [], "agreement"),
    LabelledUtterance(
        "No, I am not going to pay this amount at all.",
        "customer", [], "refusal"),
    LabelledUtterance(
        "Could you give me another two weeks to arrange the funds?",
        "customer", [], "request_extension"),
    LabelledUtterance(
        "Good morning, this is Priya calling from Trybank Financial Services.",
        "agent", [], "greeting"),
    LabelledUtterance(
        "The tenure on your home loan is 36 months.",
        "agent", [("tenure_months", "36")], "info_request"),
    LabelledUtterance(
        "A penalty of 1,200 rupees has been applied to your account.",
        "agent", [("payment_amount", "1200")], "info_request"),
]


# ── ASR financial-term corpus ─────────────────────────────────────────────────
# Errors observed in real WhisperX output from this project's own demo calls,
# paired with the term the transcript should contain. The correction layer exists
# for exactly these, so its value should be a measured number rather than a claim.

@dataclass
class AsrCase:
    raw: str            # what the ASR produced
    expect: str         # the term that must appear after correction
    must_not_change: bool = False
    note: str = ""


ASR_TERM_CASES: list[AsrCase] = [
    AsrCase("Sir, our records show your me of 12,500 rupees was due on the fifteenth of March.",
            "EMI", note="observed in every English demo run"),
    AsrCase("आपकी m.e. 12,500 रुपे 15 माच को ड्यू थी.", "EMI",
            note="observed in the Hindi demo run"),
    AsrCase("The monthly me payment is due on the fifth.", "EMI"),
    AsrCase("Your me amount has not been received.", "EMI"),
    AsrCase("Sir, this is Priya calling from Treebank Financial Services.", "Trybank",
            note="institution name misheard, observed in every run"),
    AsrCase("I checked your see bill score last week.", "CIBIL"),
    AsrCase("Your emmy is overdue by two months.", "EMI"),
    AsrCase("Please complete the kayak process before Friday.", "KYC"),
    AsrCase("The natch mandate has been registered.", "NACH"),
    # Negatives: ordinary uses that must survive untouched.
    AsrCase("Please call me back tomorrow morning.", "call me back",
            must_not_change=True, note="pronoun must not become EMI"),
    AsrCase("He told me about the payment yesterday.", "told me",
            must_not_change=True),
    AsrCase("Can you send me the statement?", "send me",
            must_not_change=True),
]
