"""Tier-2 detection of prohibited collections conduct, by meaning rather than string.

The tier-1 rule engine matches literal phrases from data/vocab/*.json, so anything
paraphrased escapes it. On this project's own demo call the utterance

    "if you do not pay by Friday we will have to send our recovery people to your
     residence and inform your employer about this default"

produced zero prohibited-language findings: the vocabulary holds "we will come to
your house" and "we will tell your employer", and neither is a substring of what
was said. A textbook RBI Fair Practice Code violation — coercive threat plus
third-party disclosure — passed silently through the flagship compliance feature.

Two approaches were measured on a 10-utterance set (5 violations, 5 benign):

  mDeBERTa zero-shot NLI   not separable. A benign "can you tell me your Aadhaar
                           number" scored 0.412 while a genuine legal threat scored
                           0.374, so no single threshold works.
  qwen2.5:3b as judge      9/10 with a constrained prompt and an output whitelist,
                           in 3.5s for all ten.

The LLM judge is used. Its one miss (an insult) is caught by the tier-1 keyword
list, which is why both layers run: keywords are fast, explainable and offline;
the LLM generalises. This is also the first real use of the two-tier design the
project's architecture describes.
"""

import os
import re
from loguru import logger

from config.schemas import ComplianceCheck, ComplianceViolationType

# key -> (check name, regulation, base severity, violation type)
CATEGORIES: dict[str, tuple[str, str, str, ComplianceViolationType]] = {
    "home_visit_threat": ("prohibited_home_visit_threat", "RBI_Fair_Practice_Code_Section_8",
                          "critical", ComplianceViolationType.PROHIBITED_LANGUAGE),
    "third_party_threat": ("prohibited_third_party_threat", "RBI_Fair_Practice_Code_Section_8d",
                           "critical", ComplianceViolationType.PRIVACY_VIOLATION),
    "legal_threat": ("prohibited_legal_threat", "RBI_Fair_Practice_Code_Section_8",
                     "high", ComplianceViolationType.PROHIBITED_LANGUAGE),
    "seizure_threat": ("prohibited_seizure_threat", "RBI_Fair_Practice_Code_Section_8",
                       "high", ComplianceViolationType.PROHIBITED_LANGUAGE),
    "harassment": ("prohibited_harassment", "RBI_Fair_Practice_Code_Section_8",
                   "high", ComplianceViolationType.PROHIBITED_LANGUAGE),
    "coercion": ("prohibited_coercion", "RBI_Fair_Practice_Code_Section_8",
                 "medium", ComplianceViolationType.PROHIBITED_LANGUAGE),
    "misleading": ("prohibited_misleading", "RBI_Fair_Practice_Code",
                   "high", ComplianceViolationType.MISLEADING_INFO),
}
ALLOWED_KEYS = set(CATEGORIES) | {"none"}

_PROMPT = """You audit Indian debt-collection calls against the RBI Fair Practice Code.
Classify ONE utterance spoken by the collections AGENT.

Most utterances are NOT violations. Requesting information, stating a balance,
confirming a payment plan, identifying yourself, and disclosing call recording are
all normal, permitted conduct. Answer "none" unless the utterance clearly matches
a category below.

Categories (answer with the exact key, nothing else):
  home_visit_threat   threatens to visit the home or send people to an address
  third_party_threat  threatens to tell an employer, family or neighbours about the debt
  legal_threat        threatens police, arrest, court or legal proceedings
  seizure_threat      threatens to seize, attach or confiscate property
  harassment          insults, shames or verbally abuses the customer
  coercion            says the customer has no choice or must pay immediately
  misleading          makes a false claim about the consequences of non-payment
  none                everything else

Examples:
  "Your outstanding balance is 45,000 rupees." -> none
  "Can you confirm your PAN number please?" -> none
  "I will record that you will pay by Tuesday." -> none
  "This call is being recorded for quality purposes." -> none
  "We will send our people to your house." -> home_visit_threat
  "I will call your office and tell your manager." -> third_party_threat

Answer with exactly one line and nothing else: VERDICT|<key>
"""

# Cheap gate: one LLM call per candidate segment, so skip utterances that cannot
# plausibly be threats. Deliberately broad — this sits in front of a compliance
# detector, so it should over-admit. Disable with FINVOICE_PROHIBITED_PREFILTER=0.
_RISK_CUES = re.compile(
    r"\b(pay|payment|money|due|overdue|default|owe|debt|loan|emi|"
    r"if you|unless|otherwise|or else|have to|will have to|going to|"
    r"send|visit|come to|residence|house|home|address|office|employer|"
    r"boss|family|neighbour|neighbor|relative|inform|tell|report|"
    r"legal|lawyer|court|police|arrest|warrant|case|notice|"
    r"seize|confiscate|attach|recovery|agent|shame|ashamed|stupid|useless|"
    r"immediately|today|right now|last chance|no choice|must|"
    r"consequence|serious|action|steps|trouble)\b",
    re.IGNORECASE,
)

_PREFILTER = os.getenv("FINVOICE_PROHIBITED_PREFILTER", "1") != "0"
_ENABLED = os.getenv("FINVOICE_PROHIBITED_SEMANTIC", "1") != "0"
MAX_SEGMENTS = int(os.getenv("FINVOICE_PROHIBITED_MAX_SEGMENTS", "40"))


def is_candidate(text: str) -> bool:
    return True if not _PREFILTER else bool(_RISK_CUES.search(text))


def classify_utterance(text: str, timeout: float = 30) -> str:
    """Return a category key for one agent utterance, or 'none'."""
    from services.llm.client import extract_raw

    # Greedy decoding. A compliance detector that flags a threat on one run and
    # misses it on the next is not usable as evidence; observed exactly that with
    # default sampling.
    raw = extract_raw(f"{_PROMPT}\nUTTERANCE: {text}\n", timeout=timeout, temperature=0.0)
    for line in (raw or "").splitlines():
        if "VERDICT" in line and "|" in line:
            key = line.split("|", 1)[1].strip().strip(".").lower()
            # The model has been observed inventing keys; anything off-list is 'none'
            # rather than a guess, because a fabricated violation is worse than a miss.
            return key if key in ALLOWED_KEYS else "none"
    return "none"


def detect_prohibited_semantic(segments: list, call_type: str) -> list[ComplianceCheck]:
    """Flag prohibited agent conduct that literal phrase matching misses.

    Only agent speech is judged — a customer threatening legal action is not an
    agent-conduct violation.
    """
    if not _ENABLED:
        return []

    from analysis.compliance import _agent_segments, _has_speaker_roles, _SEVERITY_BY_CALL_TYPE

    severity_map = _SEVERITY_BY_CALL_TYPE.get(call_type, _SEVERITY_BY_CALL_TYPE["general"])

    # This check asks "did the AGENT say something prohibited", so the
    # no-diarization fallback in _agent_segments() is not safe here: it returns
    # every segment, and a customer saying "please do not call my office" then gets
    # recorded as an agent third-party threat. Observed exactly that.
    #
    # Rather than skip the check entirely (which would silently disable the feature
    # on any call without an HF_TOKEN), findings are kept but explicitly marked
    # unattributed and capped at "high" severity, so a reviewer confirms the speaker
    # instead of the record asserting it.
    attributed = _has_speaker_roles(segments)
    if not attributed:
        from pipeline.degradations import report
        report("prohibited_semantic_attribution",
               "diarization produced no speaker roles",
               "prohibited-conduct findings cannot be attributed to the agent and are "
               "reported as speaker-unattributed for human confirmation",
               severity="partial")

    candidates = [
        (seg.get("id", i), seg.get("text", "").strip())
        for i, seg in enumerate(_agent_segments(segments))
        if len(seg.get("text", "").strip()) > 15 and is_candidate(seg.get("text", ""))
    ]
    if not candidates:
        return []

    if len(candidates) > MAX_SEGMENTS:
        logger.info(f"Prohibited-conduct check capped at {MAX_SEGMENTS} of "
                    f"{len(candidates)} candidate segments")
        candidates = candidates[:MAX_SEGMENTS]

    checks: list[ComplianceCheck] = []
    seen: set[str] = set()
    failures = 0

    for seg_id, text in candidates:
        try:
            key = classify_utterance(text)
        except Exception as e:
            failures += 1
            logger.warning(f"Prohibited-conduct judge failed on segment {seg_id}: {e}")
            continue

        if key == "none" or key not in CATEGORIES:
            continue
        name, regulation, raw_sev, vtype = CATEGORIES[key]
        if name in seen:
            continue
        seen.add(name)
        severity = severity_map.get(raw_sev, raw_sev)
        if not attributed:
            # Cap severity: without speaker roles we cannot say the agent said it.
            severity = "high" if severity == "critical" else severity
            evidence = (f'UNATTRIBUTED SPEAKER (diarization unavailable) — someone said: '
                        f'"{text[:180]}". Confirm the speaker before acting.')
        else:
            evidence = f'Agent said: "{text[:180]}"'
        checks.append(ComplianceCheck(
            check_name=name,
            passed=False,
            violation_type=vtype,
            evidence_text=evidence,
            segment_id=seg_id,
            regulation=regulation,
            severity=severity,
        ))

    if failures:
        from pipeline.degradations import report
        report("prohibited_semantic",
               f"LLM judge failed on {failures}/{len(candidates)} candidate segments",
               "paraphrased prohibited conduct may be missed in those segments",
               severity="partial")

    if checks:
        logger.info(f"Prohibited conduct (semantic): {len(checks)} violation(s) "
                    f"from {len(candidates)} candidate segment(s)")
    return checks
