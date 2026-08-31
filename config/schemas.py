"""FinVoice Pydantic schemas — structured output definitions for all pipeline stages."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── INTENT CLASSIFICATION ──

class CallIntent(str, Enum):
    AGREEMENT = "agreement"
    REFUSAL = "refusal"
    REQUEST_EXTENSION = "request_extension"
    PAYMENT_PROMISE = "payment_promise"
    COMPLAINT = "complaint"
    CONSENT_GIVEN = "consent_given"
    CONSENT_DENIED = "consent_denied"
    INFORMATION_REQUEST = "info_request"
    NEGOTIATION = "negotiation"
    ESCALATION = "escalation"
    DISPUTE = "dispute"
    ACKNOWLEDGMENT = "acknowledgment"
    GREETING = "greeting"
    # Financial / earnings call intents
    FINANCIAL_DISCLOSURE = "financial_disclosure"
    GUIDANCE_FORECAST = "guidance_forecast"
    RISK_WARNING = "risk_warning"
    QUESTION = "question"
    EXPLANATION = "explanation"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


class UtteranceIntent(BaseModel):
    """Intent classification for a single utterance."""
    segment_id: int = Field(description="Index of the transcript segment")
    speaker: str = Field(description="'agent' or 'customer'")
    intent: CallIntent
    confidence: float = Field(ge=0, le=1)
    sub_intent: Optional[str] = Field(
        None, description="More specific intent, e.g., 'partial_agreement', 'conditional_refusal'"
    )


# ── FINANCIAL ENTITY EXTRACTION ──

class CurrencyAmount(BaseModel):
    value: float
    currency: str = Field(default="INR", description="ISO currency code")
    raw_text: str = Field(description="Original text, e.g., '₹5,000' or 'five thousand rupees'")


class FinancialEntity(BaseModel):
    """A single financial entity extracted from the call."""
    entity_type: str = Field(
        description="One of: emi_amount, loan_amount, interest_rate, penalty_amount, "
        "payment_amount, account_number, reference_number, due_date, promise_date, "
        "next_call_date, product_name, tenure_months"
    )
    value: str = Field(description="Normalized value")
    raw_text: str = Field(description="Exact text from transcript")
    segment_id: int = Field(description="Which transcript segment this came from")
    start_time: float = Field(description="Timestamp in audio (seconds)")
    confidence: float = Field(ge=0, le=1)


# ── OBLIGATION & COMMITMENT DETECTION ──

class ObligationStrength(str, Enum):
    BINDING = "binding"
    CONDITIONAL = "conditional"
    PROMISE = "promise"
    VAGUE = "vague"
    DENIAL = "denial"


class Obligation(BaseModel):
    """A verbal commitment or obligation detected in the call."""
    text: str = Field(description="Exact quote from transcript")
    speaker: str = Field(description="'agent' or 'customer'")
    obligation_type: str = Field(
        description="One of: payment_promise, consent, authorization, commitment, denial, dispute"
    )
    strength: ObligationStrength
    amount: Optional[CurrencyAmount] = None
    date_referenced: Optional[str] = Field(None, description="ISO date if a date was mentioned")
    segment_id: int
    start_time: float
    legally_significant: bool = Field(
        description="True if this could have legal/compliance implications"
    )


# ── REGULATORY COMPLIANCE ──

class ComplianceViolationType(str, Enum):
    MISSING_DISCLOSURE = "missing_disclosure"
    PROHIBITED_LANGUAGE = "prohibited_language"
    CONSENT_NOT_OBTAINED = "consent_not_obtained"
    IMPROPER_COLLECTION = "improper_collection"
    PRIVACY_VIOLATION = "privacy_violation"
    MISLEADING_INFO = "misleading_information"
    CALL_TIME_VIOLATION = "call_time_violation"


class ComplianceCheck(BaseModel):
    """A single regulatory compliance check result."""
    check_name: str = Field(description="What was checked, e.g., 'opening_disclosure'")
    passed: bool
    violation_type: Optional[ComplianceViolationType] = None
    evidence_text: Optional[str] = Field(None, description="Relevant quote from transcript")
    segment_id: Optional[int] = None
    regulation: str = Field(description="Which regulation, e.g., 'RBI_Fair_Practice_Code'")
    severity: str = Field(description="'low', 'medium', 'high', 'critical'")


# ── RISK & FRAUD ──

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudSignal(BaseModel):
    signal_type: str = Field(
        description="e.g., 'voice_stress_anomaly', 'inconsistent_info', "
        "'coached_speech', 'third_party_detected'"
    )
    description: str
    segment_id: int
    confidence: float = Field(ge=0, le=1)


# ── MASTER OUTPUT: COMPLETE STRUCTURED CALL RECORD ──

class CallRecord(BaseModel):
    """Complete structured output for one call — ML-trainable and audit-ready."""

    # Metadata
    call_id: str
    audio_file: str
    duration_seconds: float
    language: str
    call_type: str = Field(
        description="'collections', 'kyc', 'onboarding', 'complaint', 'consent', 'general'"
    )

    # Audio Quality
    audio_quality_score: int = Field(ge=0, le=100)
    audio_quality_flag: str = Field(description="'TRUSTWORTHY', 'DEGRADED', 'UNRELIABLE'")
    snr_db: float
    audio_quality_components: dict = Field(
        default_factory=dict,
        description="Component scores: snr, clipping, speech_ratio, spectral (each 0-100)"
    )
    speech_percentage: float = Field(default=0.0, description="% of audio containing speech")

    # Speakers
    num_speakers: int
    agent_talk_percentage: float
    customer_talk_percentage: float

    # Full Transcript
    transcript_segments: list[dict] = Field(
        default_factory=list,
        description="Full transcript: each dict has 'start', 'end', 'text', 'speaker', 'confidence'"
    )

    # Transcript Summary
    overall_transcript_confidence: float
    num_low_confidence_segments: int

    # Intelligence Extraction
    intents: list[UtteranceIntent]
    financial_entities: list[FinancialEntity]
    obligations: list[Obligation]
    compliance_checks: list[ComplianceCheck]
    fraud_signals: list[FraudSignal]

    # PII Detection
    pii_entities: list[dict] = Field(
        default_factory=list,
        description="Detected PII: each dict has entity_type, text, masked_text, score, segment_id"
    )
    pii_count: int = Field(default=0, description="Total PII entities detected")

    # Profanity / Toxicity
    toxicity_flags: list[dict] = Field(
        default_factory=list,
        description="Toxic/profane segments: toxicity_score, categories, severity, is_agent"
    )

    # Tamper Detection
    tamper_signals: list[dict] = Field(
        default_factory=list,
        description="Audio tampering signals: spectral_discontinuity, silence_anomaly, noise_floor"
    )
    tamper_risk: str = Field(default="none", description="'none', 'low', 'medium', 'high'")

    # Audio Cleanup
    cleanup_metadata: dict = Field(
        default_factory=dict,
        description="Cleanup results: dead_air_removed, hold_music_detected, segments_stitched"
    )

    # Audio Emotions (emotion2vec)
    segment_emotions: list[dict] = Field(
        default_factory=list,
        description="Per-segment audio emotion from emotion2vec: emotion, score, all_scores"
    )
    emotion_distribution: dict = Field(
        default_factory=dict,
        description="Emotion distribution across the call"
    )

    # Language Detection
    detected_language: str = Field(default="en", description="Auto-detected primary language")
    has_code_switching: bool = Field(default=False, description="True if >10% of segments differ from primary language")
    language_distribution: dict = Field(
        default_factory=dict,
        description="Language distribution across segments: {lang_code: count}"
    )
    language_segments: list[dict] = Field(
        default_factory=list,
        description="Per-segment language detection for code-switching"
    )

    # Sentiment Trajectory
    customer_sentiment_trajectory: list[float] = Field(
        description="Sentiment score per segment for customer turns"
    )
    agent_sentiment_trajectory: list[float] = Field(
        description="Sentiment score per segment for agent turns"
    )
    customer_emotion_dominant: str = Field(description="Most frequent customer emotion")
    sentiment_context: dict = Field(
        default_factory=dict,
        description="Sentiment interpretation: {type: 'financial_negative'|'emotional_negative'|'neutral', actionable: bool}"
    )
    escalation_detected: bool

    # Risk Assessment
    overall_risk_level: RiskLevel
    compliance_score: int = Field(ge=0, le=100, description="100 = fully compliant")
    requires_human_review: bool
    review_priority: int = Field(ge=1, le=5, description="1=urgent, 5=routine")
    review_reasons: list[str]

    # Analyzers that did not run, or ran in a reduced mode. Empty means every
    # component executed. Consumers (and any evaluation harness) must check this
    # before treating an absent result as a negative result.
    degradations: list[dict] = Field(
        default_factory=list,
        description="Components that silently degraded: [{component, reason, impact, severity}]",
    )

    # Pipeline Performance
    pipeline_timings: dict = Field(
        default_factory=dict,
        description="Per-stage timing in seconds: {'Stage 1: Normalize': 2.6, ...}"
    )

    # Speaker Emotion Breakdown
    speaker_emotion_breakdown: dict = Field(
        default_factory=dict,
        description="Per-speaker emotion profile: {speaker: {dominant, distribution, total_segments}}"
    )
    emotion_transitions: list[dict] = Field(
        default_factory=list,
        description="Significant emotion shifts per speaker: [{speaker, from_emotion, to_emotion, at_segment, significance}]"
    )
    escalation_moments: list[dict] = Field(
        default_factory=list,
        description="Segments where the customer turned angry/frustrated: [{segment_id, speaker, emotion, score}]"
    )

    # Call Summary
    call_summary: str = Field(description="2-3 sentence natural language summary")
    key_outcomes: list[str] = Field(description="Bullet points of call outcomes")
    next_actions: list[str] = Field(description="Required follow-up actions")

    # Figures computed in code from extracted entities, never by the LLM.
    derived_totals: dict = Field(
        default_factory=dict,
        description="Sums/counts per entity type, computed from financial_entities",
    )
    numeric_audit: dict = Field(
        default_factory=dict,
        description="What the numeric guard removed: {removed_currency_claims, unsupported_figures}",
    )

    # Financial Insights (synthesized analysis for end users)
    financial_insights: dict = Field(
        default_factory=dict,
        description="Synthesized financial analysis: {key_metrics: [], risk_factors: [], recommendations: [], topic_sentiment: {}}"
    )
