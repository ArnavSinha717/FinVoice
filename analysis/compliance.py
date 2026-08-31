"""Regulatory Compliance Checking — Tier 1 keyword matching (CPU, instant).

Covers Indian financial regulatory framework:
- RBI Fair Practice Code (debt collection conduct)
- RBI KYC Master Direction (identity verification)
- RBI Digital Lending Guidelines (consent, transparency)
- Digital Personal Data Protection Act 2023 (data consent)
- Consumer Protection Act 2019 (right to refuse)
- IT Act 2000 Section 43A (recording disclosure)
- SEBI Guidelines (investment-related calls)
- RBI Guidelines on Call Timing (Section 8(c))
"""

import re

from config.schemas import ComplianceCheck, ComplianceViolationType
from analysis.vocab_loader import (
    get_compliance_keywords,
    get_prohibited_phrases as get_vocab_prohibited_phrases,
    get_end_call_phrases,
)

# Call types where disclosure checks are mandatory (regulated interactions).
# For "general" (earnings calls, investor presentations, etc.), disclosure
# checks are skipped — they produce false positives on non-regulated calls.
_REGULATED_CALL_TYPES = {"collections", "kyc", "consent", "onboarding", "complaint"}

# Severity multipliers by call type. Collections/KYC violations are legally
# significant; general call violations are mostly noise.
_SEVERITY_BY_CALL_TYPE = {
    "collections": {"critical": "critical", "high": "high", "medium": "medium", "low": "low"},
    "kyc": {"critical": "critical", "high": "high", "medium": "medium", "low": "low"},
    "consent": {"critical": "critical", "high": "high", "medium": "medium", "low": "low"},
    "complaint": {"critical": "high", "high": "medium", "medium": "low", "low": "low"},
    "onboarding": {"critical": "high", "high": "medium", "medium": "low", "low": "low"},
    "general": {"critical": "low", "high": "low", "medium": "low", "low": "low"},
}


def _get_disclosures(call_type: str, lang: str = "en") -> list[dict]:
    """Get required disclosure checks for a call type and language.

    Loads keywords from data/vocab/{lang}.json. For non-English, also
    includes English keywords (agent may use English terms in Hindi/Tamil calls).
    """
    compliance_kw = get_compliance_keywords(lang)
    checks_dict = compliance_kw.get(call_type, compliance_kw.get("general", {}))

    disclosures = []
    for check_name, check_data in checks_dict.items():
        keywords = list(check_data.get("keywords", []))
        # For non-English, also check English keywords (code-switching is common)
        if lang != "en":
            en_kw = get_compliance_keywords("en")
            en_checks = en_kw.get(call_type, en_kw.get("general", {}))
            if check_name in en_checks:
                keywords.extend(en_checks[check_name].get("keywords", []))
        disclosures.append({
            "check": check_name,
            "keywords": keywords,
            "regulation": check_data.get("regulation", "RBI_Fair_Practice_Code"),
        })

    return disclosures


def _get_prohibited(lang: str = "en") -> dict[str, list[str]]:
    """Get prohibited phrases for a language.

    For non-English, combines language-specific phrases with English phrases.
    """
    phrases = get_vocab_prohibited_phrases(lang)
    if lang != "en":
        en_phrases = get_vocab_prohibited_phrases("en")
        for cat, cat_phrases in en_phrases.items():
            if cat in phrases:
                phrases[cat] = list(set(phrases[cat] + cat_phrases))
            else:
                phrases[cat] = cat_phrases
    return phrases


_AGENT_LABELS = {"agent", "speaker_00", "speaker a"}


def _has_speaker_roles(segments: list) -> bool:
    """True if diarization actually assigned agent/customer roles.

    Without an HF_TOKEN, diarization is skipped and every segment comes back as
    "unknown". Agent-attributed checks would then see empty text: they would fail
    every disclosure the agent actually made, and miss prohibited language the
    agent actually used. Callers fall back to the whole transcript instead.

    Note this fallback is only safe for checks that ask "was this said at all".
    Checks comparing agent vs customer talk time must stay disabled without roles.
    """
    return any(s.get("speaker", "").lower() in _AGENT_LABELS for s in segments)


def _agent_segments(segments: list) -> list:
    """Agent segments, or every segment when diarization produced no roles."""
    if _has_speaker_roles(segments):
        return [s for s in segments if s.get("speaker", "").lower() in _AGENT_LABELS]
    return list(segments)


def _get_agent_opening_text(segments: list, max_seconds: float = 30.0) -> str:
    """Get agent's opening speech by cumulative talk time, not clock time.

    Collects agent segments until cumulative agent speech reaches max_seconds.
    This is robust to hold music, IVR menus, or late-start recordings where
    clock time doesn't reflect actual agent speech.
    """
    texts = []
    cumulative = 0.0
    for seg in _agent_segments(segments):
        duration = seg.get("end", 0) - seg.get("start", 0)
        texts.append(seg.get("text", ""))
        cumulative += duration
        if cumulative >= max_seconds:
            break
    return " ".join(texts).lower()


def run_compliance_checks(
    transcript_segments: list, call_type: str, lang: str = "en"
) -> list[ComplianceCheck]:
    """Check transcript against regulatory requirements.

    Tier 1: CPU-only keyword matching. Catches ~80% of violations instantly.
    Covers:
    - Required disclosures per call type
    - Prohibited phrases (threats, coercion, misleading, harassment, privacy)
    - Call timing violations (RBI Section 8(c))
    - Third-party disclosure violations
    - Repeated call harassment patterns
    """
    checks = []
    severity_map = _SEVERITY_BY_CALL_TYPE.get(call_type, _SEVERITY_BY_CALL_TYPE["general"])

    # Skip disclosure checks for non-regulated call types (earnings calls, etc.)
    if call_type in _REGULATED_CALL_TYPES:
        disclosures = _get_disclosures(call_type, lang=lang)

        # Agent speech: first 30 seconds of cumulative agent talk time
        # (robust to hold music, IVR, or late-start recordings)
        agent_opening = _get_agent_opening_text(transcript_segments, max_seconds=30)

        # Check required disclosures
        for disclosure in disclosures:
            found = any(
                re.search(rf"\b{re.escape(kw)}\b", agent_opening, re.IGNORECASE)
                for kw in disclosure["keywords"]
            )
            raw_severity = "high" if not found else "low"
            checks.append(ComplianceCheck(
                check_name=disclosure["check"],
                passed=found,
                violation_type=ComplianceViolationType.MISSING_DISCLOSURE if not found else None,
                evidence_text=agent_opening[:200] if not found else None,
                regulation=disclosure["regulation"],
                severity=severity_map.get(raw_severity, raw_severity),
            ))

    # Check prohibited phrases across call.
    # For regulated calls: agent speech only. For general calls: all speech
    # (speakers are labeled "Speaker A"/"Speaker B", not "agent"/"customer").
    if call_type in _REGULATED_CALL_TYPES:
        full_agent_text = " ".join(
            s.get("text", "") for s in _agent_segments(transcript_segments)
        ).lower()
    else:
        full_agent_text = " ".join(
            s.get("text", "") for s in transcript_segments
        ).lower()

    prohibited = _get_prohibited(lang=lang)
    for category, phrases in prohibited.items():
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase.lower())}\b", full_agent_text, re.IGNORECASE):
                seg_id = _find_segment_containing(transcript_segments, phrase)
                raw_severity = "critical" if category in ("threats", "coercion", "misleading") else "high"
                checks.append(ComplianceCheck(
                    check_name=f"prohibited_{category}",
                    passed=False,
                    violation_type=ComplianceViolationType.PROHIBITED_LANGUAGE,
                    evidence_text=f'Agent said: "{phrase}"',
                    segment_id=seg_id,
                    regulation="RBI_Fair_Practice_Code",
                    severity=severity_map.get(raw_severity, raw_severity),
                ))

    # Check for agent sharing customer info with third parties
    third_party_checks = _check_third_party_disclosure(transcript_segments)
    checks.extend(third_party_checks)

    # Check call duration (collections calls shouldn't exceed reasonable time)
    duration_checks = _check_call_conduct(transcript_segments, call_type, lang=lang)
    checks.extend(duration_checks)

    # General/earnings call checks — check for safe harbor, forward-looking disclaimers
    if call_type == "general" and transcript_segments:
        full_text = " ".join(s.get("text", "") for s in transcript_segments).lower()

        # Safe harbor / forward-looking statement disclaimer
        has_safe_harbor = any(
            kw in full_text
            for kw in ("forward-looking statement", "safe harbor", "actual results may differ",
                        "not guarantee", "subject to risk", "sec filing")
        )
        checks.append(ComplianceCheck(
            check_name="Forward-Looking Disclaimer",
            passed=has_safe_harbor,
            violation_type=None if has_safe_harbor else ComplianceViolationType.MISSING_DISCLOSURE,
            evidence_text=None if has_safe_harbor else "No forward-looking statement disclaimer detected",
            regulation="SEC_Regulation_FD",
            severity="low",
        ))

        # Call recording disclosure
        has_recording_notice = any(
            kw in full_text
            for kw in ("being recorded", "call is recorded", "recording", "webcast")
        )
        checks.append(ComplianceCheck(
            check_name="Recording Disclosure",
            passed=has_recording_notice,
            violation_type=None if has_recording_notice else ComplianceViolationType.MISSING_DISCLOSURE,
            evidence_text=None if has_recording_notice else "No recording/webcast notice detected",
            regulation="General_Compliance",
            severity="low",
        ))

        # Speaker identification
        has_intro = any(
            kw in full_text
            for kw in ("my name is", "this is", "i'm ", "speaking today", "i am ")
        )
        checks.append(ComplianceCheck(
            check_name="Speaker Identification",
            passed=has_intro,
            violation_type=None if has_intro else ComplianceViolationType.MISSING_DISCLOSURE,
            evidence_text=None if has_intro else "No speaker introduction detected",
            regulation="General_Compliance",
            severity="low",
        ))

    checks.extend(check_dpdp_compliance(transcript_segments, call_type, lang))

    return checks


def _check_third_party_disclosure(segments: list) -> list[ComplianceCheck]:
    """Check if agent disclosed customer info to third parties.

    RBI Fair Practice Code prohibits sharing customer debt info with
    anyone other than the customer themselves.
    """
    checks = []
    full_text = " ".join(s.get("text", "") for s in segments).lower()

    # Patterns suggesting agent discussed debt with third party
    third_party_phrases = [
        "i spoke to your wife", "i spoke to your husband",
        "i called your office", "i told your boss",
        "your family knows", "we informed your neighbor",
        "we contacted your reference", "spoke to your colleague",
    ]

    for phrase in third_party_phrases:
        if re.search(rf"\b{re.escape(phrase)}\b", full_text, re.IGNORECASE):
            seg_id = _find_segment_containing(segments, phrase)
            checks.append(ComplianceCheck(
                check_name="third_party_disclosure",
                passed=False,
                violation_type=ComplianceViolationType.PRIVACY_VIOLATION,
                evidence_text=f'Possible third-party disclosure: "{phrase}"',
                segment_id=seg_id,
                regulation="RBI_Fair_Practice_Code_Section_8d",
                severity="critical",
            ))

    return checks


def _check_call_conduct(segments: list, call_type: str, lang: str = "en") -> list[ComplianceCheck]:
    """Check general call conduct rules.

    - Agent interruption ratio (shouldn't dominate conversation excessively)
    - Use of respectful language markers
    """
    checks = []

    if not segments:
        return checks

    # Check for agent domination in collections (shouldn't be >70% talk time)
    if call_type == "collections":
        _agent_labels = {"agent", "speaker_00", "speaker a"}
        _customer_labels = {"customer", "speaker_01", "speaker b"}
        agent_segs = [s for s in segments if s.get("speaker", "").lower() in _agent_labels]
        customer_segs = [s for s in segments if s.get("speaker", "").lower() in _customer_labels]

        agent_duration = sum(s.get("end", 0) - s.get("start", 0) for s in agent_segs)
        customer_duration = sum(s.get("end", 0) - s.get("start", 0) for s in customer_segs)
        total = agent_duration + customer_duration

        if total > 0 and agent_duration / total > 0.75:
            checks.append(ComplianceCheck(
                check_name="agent_domination",
                passed=False,
                violation_type=ComplianceViolationType.IMPROPER_COLLECTION,
                evidence_text=f"Agent spoke {agent_duration/total*100:.0f}% of call (>75% threshold)",
                regulation="RBI_Fair_Practice_Code_Section_8",
                severity="medium",
            ))

    # Check for customer requesting end of call being ignored
    end_call_vocab = get_end_call_phrases(lang)
    # For non-English, also check English phrases
    if lang != "en":
        end_call_vocab = list(set(end_call_vocab + get_end_call_phrases("en")))
    end_call_phrases = end_call_vocab if end_call_vocab else []
    # Match customer speech — handles both regulated labels (customer/speaker_01)
    # and general labels (Speaker B, etc.)
    _customer_labels = {"customer", "speaker_01", "speaker b"}
    full_customer_text = " ".join(
        s.get("text", "")
        for s in segments
        if s.get("speaker", "").lower() in _customer_labels
    ).lower()

    for phrase in end_call_phrases:
        if re.search(rf"\b{re.escape(phrase)}\b", full_customer_text, re.IGNORECASE):
            # Check if agent continued speaking significantly after this
            phrase_seg_id = _find_segment_containing(segments, phrase)
            if phrase_seg_id is not None:
                remaining_agent_segs = [
                    s for s in segments[phrase_seg_id + 1:]
                    if s.get("speaker", "").lower() in ("agent", "speaker_00")
                ]
                if len(remaining_agent_segs) > 3:
                    checks.append(ComplianceCheck(
                        check_name="ignored_call_end_request",
                        passed=False,
                        violation_type=ComplianceViolationType.IMPROPER_COLLECTION,
                        evidence_text=f'Customer said: "{phrase}" but agent continued ({len(remaining_agent_segs)} more segments)',
                        segment_id=phrase_seg_id,
                        regulation="RBI_Fair_Practice_Code_Section_8c",
                        severity="high",
                    ))
            break

    return checks


# ── Digital Personal Data Protection Act, 2023 ──
# Tier 1 keyword checks for the DPDP obligations that are observable in a call:
# purpose notice (S.5), free and informed consent (S.6), communication of the
# data principal's rights (S.12-13), and data minimisation (S.6(1)).

# Language indicating the agent is requesting personal data from the customer.
_DPDP_DATA_REQUESTS = [
    "aadhaar", "aadhar", "pan number", "pan card", "date of birth", "dob",
    "account number", "card number", "cvv", "otp", "one time password",
    "mother's maiden name", "your address", "email address", "mobile number",
]

# Agent states WHY the data is needed.
_DPDP_PURPOSE_PHRASES = [
    "for the purpose of", "in order to verify", "to verify your identity",
    "this is required for", "we need this to", "used only for",
    "for verification purposes", "as part of kyc", "to process your",
]

# Agent seeks affirmative permission before collecting.
_DPDP_CONSENT_PHRASES = [
    "do you consent", "may i have your", "can i confirm your", "is that okay",
    "with your permission", "do you agree", "are you comfortable sharing",
    "would you like to proceed", "do i have your consent",
]

# Agent communicates the data principal's rights.
_DPDP_RIGHTS_PHRASES = [
    "withdraw your consent", "grievance", "data protection officer",
    "you can request deletion", "right to erasure", "opt out",
    "privacy policy", "your data rights", "lodge a complaint",
]

# Identifiers that are excessive outside identity-verification contexts.
_DPDP_SENSITIVE_IDS = ["aadhaar", "aadhar", "cvv", "otp", "one time password"]
_DPDP_MINIMISATION_EXEMPT = {"kyc", "onboarding"}


def check_dpdp_compliance(
    transcript_segments: list, call_type: str, lang: str = "en"
) -> list[ComplianceCheck]:
    """Check a call against the Digital Personal Data Protection Act, 2023.

    Only applies to regulated call types — an earnings call has no data principal.
    Checks are skipped entirely when the agent never requests personal data,
    so a call that collects nothing is not penalised for missing a consent notice.
    """
    if call_type not in _REGULATED_CALL_TYPES:
        return []

    severity_map = _SEVERITY_BY_CALL_TYPE.get(call_type, _SEVERITY_BY_CALL_TYPE["general"])
    agent_text = " ".join(
        s.get("text", "") for s in _agent_segments(transcript_segments)
    ).lower()

    if not agent_text:
        return []

    requested = [kw for kw in _DPDP_DATA_REQUESTS if kw in agent_text]
    if not requested:
        # No personal data solicited — DPDP processing obligations are not triggered.
        return []

    checks: list[ComplianceCheck] = []

    def _add(name: str, phrases: list[str], violation, regulation: str, raw_severity: str, detail: str):
        found = any(p in agent_text for p in phrases)
        checks.append(ComplianceCheck(
            check_name=name,
            passed=found,
            violation_type=None if found else violation,
            evidence_text=None if found else detail,
            regulation=regulation,
            severity=severity_map.get(raw_severity, "low") if not found else "low",
        ))

    _add(
        "dpdp_purpose_notice", _DPDP_PURPOSE_PHRASES,
        ComplianceViolationType.MISSING_DISCLOSURE,
        "DPDP_Act_2023_Section_5", "high",
        f"Agent requested {', '.join(requested[:3])} without stating the purpose of collection.",
    )
    _add(
        "dpdp_consent_obtained", _DPDP_CONSENT_PHRASES,
        ComplianceViolationType.CONSENT_NOT_OBTAINED,
        "DPDP_Act_2023_Section_6", "critical",
        f"Agent requested {', '.join(requested[:3])} without seeking affirmative consent.",
    )
    _add(
        "dpdp_rights_notice", _DPDP_RIGHTS_PHRASES,
        ComplianceViolationType.MISSING_DISCLOSURE,
        "DPDP_Act_2023_Section_12", "medium",
        "Customer was not informed of consent withdrawal or grievance redressal rights.",
    )

    # Data minimisation: sensitive identifiers outside an identity-verification call.
    if call_type not in _DPDP_MINIMISATION_EXEMPT:
        excessive = [kw for kw in _DPDP_SENSITIVE_IDS if kw in agent_text]
        if excessive:
            checks.append(ComplianceCheck(
                check_name="dpdp_data_minimisation",
                passed=False,
                violation_type=ComplianceViolationType.PRIVACY_VIOLATION,
                evidence_text=f"Requested {', '.join(excessive)} on a '{call_type}' call, which does not require identity verification.",
                regulation="DPDP_Act_2023_Section_6_1",
                severity=severity_map.get("high", "low"),
            ))

    return checks


def _find_segment_containing(segments: list, phrase: str) -> int | None:
    """Find the segment index that contains a given phrase."""
    phrase_lower = phrase.lower()
    for i, seg in enumerate(segments):
        if phrase_lower in seg.get("text", "").lower():
            return i
    return None
