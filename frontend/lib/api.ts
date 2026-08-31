// ===== FinVoice API Layer =====
// Real backend integration — all functions fetch from FastAPI backend via Next.js proxy.

// Resolve backend host once on client side — avoids SSR hydration mismatch
// by always using 'localhost' during SSR, then updating on client mount
function getBackendBase(): string {
  if (process.env.NEXT_PUBLIC_BACKEND_URL) return process.env.NEXT_PUBLIC_BACKEND_URL;
  if (typeof window !== 'undefined') return `http://${window.location.hostname}:8000`;
  return 'http://localhost:8000';
}
const API_BASE = getBackendBase() + '/api';
const BACKEND_DIRECT = getBackendBase();

// ===== TYPES =====

export interface FinancialInsights {
  key_metrics: { type: string; value: string; context: string; confidence?: number; segment_id?: number }[];
  risk_factors: { type: string; detail: string }[];
  recommendations: string[];
  topic_sentiment: Record<string, number>;
  call_effectiveness: {
    total_segments: number;
    total_speakers: number;
    entities_extracted: number;
    intents_classified: number;
    unknown_rate: number;
    disclosure_rate: number;
    qa_rate: number;
    total_words?: number;
    avg_words_per_segment?: number;
    speaker_balance?: number;
  };
  discussion_topics: { topic: string; mentions: number; percentage: number }[];
}

export interface AudioQualityComponents {
  snr: number;
  clipping: number;
  speech_ratio: number;
  spectral: number;
}

export interface Call {
  id: string;
  date: string;
  time: string;
  duration: string;
  durationSec: number;
  language: string;
  agent: string;
  customer: string;
  complianceScore: number;
  fraudRisk: 'low' | 'medium' | 'high';
  status: 'reviewed' | 'flagged' | 'escalated' | 'pending';
  summary: string;
  tags: string[];
  pipelineStage: number;
  audioQualityScore: number;
  audioQualityFlag: string;
  snrDb: number;
  audioQualityComponents: AudioQualityComponents;
  speechPercentage: number;
  tamperRisk: string;
  overallTranscriptConfidence: number;
  numLowConfidenceSegments: number;
  keyOutcomes: string[];
  nextActions: string[];
  financialInsights: FinancialInsights;
  numSpeakers: number;
  agentTalkPct: number;
  customerTalkPct: number;
  pipelineTimings: Record<string, number>;
}

export interface TranscriptMessage {
  time: string;
  startSec: number;
  speaker: string;
  name: string;
  text: string;
  confidence: number;
}

export interface Obligation {
  text: string;
  speaker: string;
  type: string;
  strength: string;
  amount: string | null;
  date: string | null;
  legallySignificant: boolean;
  segmentId: number;
}

export interface TamperSignal {
  signalType: string;
  description: string;
  timestamp: number;
  confidence: number;
  severity: string;
}

export interface Entity {
  type: string;
  value: string;
  context: string;
  confidence: number;
}

export interface Intent {
  utterance: string;
  intent: string;
  confidence: number;
  speaker: string;
}

export interface ComplianceRule {
  rule: string;
  passed: boolean;
  detail: string;
}

export interface FraudSignal {
  signal: string;
  risk: 'none' | 'low' | 'medium' | 'high';
  detail: string;
}

export interface Stats {
  totalCalls: number;
  avgCompliance: number;
  fraudAlerts: number;
  pendingReviews: number;
  riskDistribution: Record<string, number>;
  languages: Record<string, number>;
  totalDurationSeconds: number;
}

export interface ReviewItem {
  id: string;
  priority: number;
  riskType: string;
  summary: string;
  flags: string[];
  date: string;
  agent: string;
}

export interface PipelineStage {
  id: number;
  name: string;
  color: string;
  tools: string[];
  desc: string;
  outputs: string[];
}

export interface SpeakerEmotionProfile {
  dominant: string;
  distribution: Record<string, number>;
  total_segments: number;
}

export interface EmotionData {
  segmentEmotions: Array<{
    segment_id: number;
    speaker: string;
    emotion: string;
    score: number;
  }>;
  emotionDistribution: Record<string, number>;
  customerTrajectory: number[];
  agentTrajectory: number[];
  dominantEmotion: string;
  speakerBreakdown: Record<string, SpeakerEmotionProfile>;
  escalationMoments: Array<{ segment_id: number; speaker: string; emotion: string; score: number }>;
}

// ===== HELPERS =====

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// review_priority: 1 = urgent ... 5 = routine (see config/schemas.py CallRecord)
function deriveStatus(data: Record<string, unknown>): Call['status'] {
  if (!data.requires_human_review) return 'reviewed';
  const priority = (data.review_priority as number) ?? 5;
  if (priority <= 1) return 'escalated';
  if (priority <= 3) return 'flagged';
  return 'pending';
}

function normalizeSpeaker(raw: string): string {
  const lower = (raw || '').toLowerCase();
  if (lower.includes('agent') || lower === 'speaker_00') return 'agent';
  if (lower.includes('customer') || lower === 'speaker_01') return 'customer';
  return lower || 'other';
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function prettifyCheckName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

function confidenceToRisk(confidence: number): FraudSignal['risk'] {
  if (confidence >= 0.7) return 'high';
  if (confidence >= 0.4) return 'medium';
  if (confidence >= 0.1) return 'low';
  return 'none';
}

// ===== PIPELINE STAGES (static) =====

export const PIPELINE_STAGES: PipelineStage[] = [
  { id: 1, name: 'Ingestion', color: 'var(--s1)', tools: ['FFmpeg', 'Silero VAD'], desc: 'Audio normalization, quality scoring, dead air removal', outputs: ['Normalized WAV', 'Quality score', 'Cleaned audio'] },
  { id: 2, name: 'Transcription', color: 'var(--s2)', tools: ['WhisperX', 'pyannote'], desc: 'Financial transcription with speaker diarization', outputs: ['Time-aligned transcript', 'Speaker labels', 'Confidence scores'] },
  { id: 3, name: 'NLU & Extraction', color: 'var(--s3)', tools: ['FinBERT', 'spaCy', 'Qwen3 8B'], desc: 'Entity extraction, intent classification, sentiment analysis', outputs: ['Entities JSON', 'Intent labels', 'Sentiment scores'] },
  { id: 4, name: 'Compliance & Fraud', color: 'var(--s4)', tools: ['Rule Engine', 'emotion2vec', 'Parselmouth'], desc: 'Regulatory compliance, fraud detection, tamper analysis', outputs: ['Compliance scorecard', 'Fraud signals', 'Tamper risk'] },
  { id: 5, name: 'Output & Storage', color: 'var(--s5)', tools: ['Pydantic', 'PyArrow', 'Export API'], desc: 'Structured data export, audit trails, ML-trainable datasets', outputs: ['CallRecord JSON', 'Training datasets', 'Audit logs'] },
];

// ===== API FUNCTIONS =====

export async function getCalls(): Promise<Call[]> {
  try {
    const res = await fetch(`${API_BASE}/calls?limit=100`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.calls || []).map((c: Record<string, unknown>) => ({
      id: c.call_id as string,
      date: new Date((c as Record<string, unknown>).date as string || Date.now()).toLocaleDateString('en-IN'),
      time: new Date((c as Record<string, unknown>).date as string || Date.now()).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
      duration: formatDuration((c.duration_seconds as number) || 0),
      durationSec: (c.duration_seconds as number) || 0,
      language: (c.language as string) || 'en',
      agent: 'Agent',
      customer: 'Customer',
      complianceScore: (c.compliance_score as number) ?? 0,
      fraudRisk: ((c.overall_risk_level as string) || 'low') as Call['fraudRisk'],
      status: deriveStatus(c),
      summary: (c.call_summary as string) || '',
      tags: [(c.call_type as string) || 'general'],
      pipelineStage: 5,
      audioQualityScore: (c.audio_quality_score as number) ?? 0,
      audioQualityFlag: (c.audio_quality_flag as string) || 'UNKNOWN',
      snrDb: (c.snr_db as number) ?? 0,
      audioQualityComponents: (c.audio_quality_components as AudioQualityComponents) || { snr: 0, clipping: 0, speech_ratio: 0, spectral: 0 },
      speechPercentage: (c.speech_percentage as number) ?? 0,
      tamperRisk: (c.tamper_risk as string) || 'none',
      overallTranscriptConfidence: (c.overall_transcript_confidence as number) ?? 0,
      numLowConfidenceSegments: (c.num_low_confidence_segments as number) ?? 0,
      keyOutcomes: (c.key_outcomes as string[]) || [],
      nextActions: (c.next_actions as string[]) || [],
      financialInsights: (c.financial_insights as FinancialInsights) || { key_metrics: [], risk_factors: [], recommendations: [], topic_sentiment: {}, call_effectiveness: { total_segments: 0, total_speakers: 0, entities_extracted: 0, intents_classified: 0, unknown_rate: 0, disclosure_rate: 0, qa_rate: 0 }, discussion_topics: [] },
      numSpeakers: (c.num_speakers as number) || 2,
      agentTalkPct: (c.agent_talk_percentage as number) ?? 0,
      customerTalkPct: (c.customer_talk_percentage as number) ?? 0,
      pipelineTimings: (c.pipeline_timings as Record<string, number>) || {},
    }));
  } catch {
    return [];
  }
}

export async function getCallById(id: string): Promise<Call | null> {
  try {
    const res = await fetch(`${API_BASE}/results/${id}`);
    if (!res.ok) return null;
    const c = await res.json();
    return {
      id: c.call_id,
      date: new Date(Date.now()).toLocaleDateString('en-IN'),
      time: new Date(Date.now()).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
      duration: formatDuration(c.duration_seconds || 0),
      durationSec: c.duration_seconds || 0,
      language: c.detected_language || c.language || 'en',
      agent: 'Agent',
      customer: 'Customer',
      complianceScore: c.compliance_score ?? 0,
      fraudRisk: (c.overall_risk_level || 'low') as Call['fraudRisk'],
      status: deriveStatus(c),
      summary: c.call_summary || '',
      tags: [c.call_type || 'general'],
      pipelineStage: 5,
      audioQualityScore: c.audio_quality_score ?? 0,
      audioQualityFlag: c.audio_quality_flag || 'UNKNOWN',
      snrDb: c.snr_db ?? 0,
      audioQualityComponents: c.audio_quality_components || { snr: 0, clipping: 0, speech_ratio: 0, spectral: 0 },
      speechPercentage: c.speech_percentage ?? 0,
      tamperRisk: c.tamper_risk || 'none',
      overallTranscriptConfidence: c.overall_transcript_confidence ?? 0,
      numLowConfidenceSegments: c.num_low_confidence_segments ?? 0,
      keyOutcomes: (c.key_outcomes as string[]) || [],
      nextActions: (c.next_actions as string[]) || [],
      financialInsights: (c.financial_insights as FinancialInsights) || { key_metrics: [], risk_factors: [], recommendations: [], topic_sentiment: {}, call_effectiveness: { total_segments: 0, total_speakers: 0, entities_extracted: 0, intents_classified: 0, unknown_rate: 0, disclosure_rate: 0, qa_rate: 0 }, discussion_topics: [] },
      numSpeakers: (c.num_speakers as number) || 2,
      agentTalkPct: c.agent_talk_percentage ?? 0,
      customerTalkPct: c.customer_talk_percentage ?? 0,
      pipelineTimings: c.pipeline_timings || {},
    };
  } catch {
    return null;
  }
}

export async function getTranscript(callId: string): Promise<TranscriptMessage[]> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/transcript`);
    if (!res.ok) return [];
    const data = await res.json();
    const segments = data.segments || [];

    // Also fetch entities to inject markup
    let entities: Array<Record<string, unknown>> = [];
    try {
      const entRes = await fetch(`${API_BASE}/calls/${callId}/entities`);
      if (entRes.ok) {
        const entData = await entRes.json();
        entities = entData.financial_entities || [];
      }
    } catch { /* ignore */ }

    return segments.map((seg: Record<string, unknown>) => {
      const speaker = normalizeSpeaker(seg.speaker as string);
      let text = escapeHtml((seg.text as string) || '');

      // Inject entity markup for matching entities in this segment
      const segId = seg.segment_id ?? segments.indexOf(seg);
      const segEntities = entities.filter(e => e.segment_id === segId);
      for (const ent of segEntities) {
        const raw = escapeHtml((ent.raw_text as string) || '');
        if (raw && text.includes(raw)) {
          const etype = escapeHtml((ent.entity_type as string) || 'entity');
          text = text.replace(raw, `<entity type="${etype}">${raw}</entity>`);
        }
      }

      const realName = seg.speaker_name as string | undefined;
      const speakerLabel = realName ? realName :
        speaker === 'agent' ? 'Agent' : speaker === 'customer' ? 'Customer' :
        `Speaker ${((seg.original_speaker_id as string) || speaker).replace('SPEAKER_', '')}`;
      return {
        time: formatTimestamp((seg.start as number) || 0),
        startSec: (seg.start as number) || 0,
        speaker,
        name: speakerLabel,
        text,
        confidence: (seg.confidence as number) ?? 1.0,
      };
    });
  } catch {
    return [];
  }
}

export async function getEntities(callId: string): Promise<Entity[]> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/entities`);
    if (!res.ok) return [];
    const data = await res.json();
    const all = [
      ...(data.financial_entities || []),
      ...(data.pii_entities || [])
        .filter((p: Record<string, unknown>) => {
          if (!p.text && !p.original_text) return false;  // skip empty PII
          // Filter out garbage DATE_TIME detections (durations, phone numbers)
          if (p.entity_type === 'DATE_TIME') {
            const text = ((p.text as string) || '').toLowerCase();
            if (/minute|second|moment|hour/.test(text)) return false;
            if (/^\d[\d\-]+$/.test(text)) return false;  // phone numbers
          }
          return true;
        })
        .map((p: Record<string, unknown>) => ({
          entity_type: `PII_${(p.entity_type as string) || 'UNKNOWN'}`,
          value: (p.text as string) || (p.original_text as string) || '',
          raw_text: `${(p.masked_text as string) || '[REDACTED]'} (detected in segment ${p.segment_id ?? '?'})`,
          confidence: (p.score as number) || 0.9,
        })),
    ];
    return all.map((e: Record<string, unknown>) => ({
      type: (e.entity_type as string) || 'unknown',
      value: (e.value as string) || '',
      context: (e.raw_text as string) || (e.entity_type as string) || '',
      confidence: (e.confidence as number) ?? 0,
    }));
  } catch {
    return [];
  }
}

export async function getIntents(callId: string): Promise<Intent[]> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/intents`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.intents || []).map((i: Record<string, unknown>) => ({
      utterance: (i.text as string) || '',
      intent: (i.intent as string) || (i.label as string) || 'unknown',
      confidence: (i.confidence as number) ?? 0,
      speaker: (i.speaker as string) || 'unknown',
    }));
  } catch {
    return [];
  }
}

export async function getCompliance(callId: string): Promise<ComplianceRule[]> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/compliance`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.compliance_checks || []).map((c: Record<string, unknown>) => ({
      rule: prettifyCheckName((c.check_name as string) || ''),
      passed: c.passed as boolean,
      detail: [c.evidence_text, c.regulation].filter(Boolean).join(' — '),
    }));
  } catch {
    return [];
  }
}

export async function getFraudSignals(callId: string): Promise<FraudSignal[]> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/compliance`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.fraud_signals || []).map((f: Record<string, unknown>) => ({
      signal: (f.signal_type as string) || 'unknown',
      risk: confidenceToRisk((f.confidence as number) || 0),
      detail: (f.description as string) || '',
    }));
  } catch {
    return [];
  }
}

export async function getEmotions(callId: string): Promise<EmotionData | null> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/emotions`);
    if (!res.ok) return null;
    const data = await res.json();
    return {
      segmentEmotions: data.segment_emotions || [],
      emotionDistribution: data.emotion_distribution || {},
      customerTrajectory: data.customer_sentiment_trajectory || [],
      agentTrajectory: data.agent_sentiment_trajectory || [],
      dominantEmotion: data.customer_emotion_dominant || 'neutral',
      speakerBreakdown: data.speaker_emotion_breakdown || {},
      escalationMoments: data.escalation_moments || [],
    };
  } catch {
    return null;
  }
}

export async function getStats(): Promise<Stats> {
  const empty: Stats = { totalCalls: 0, avgCompliance: 0, fraudAlerts: 0, pendingReviews: 0, riskDistribution: {}, languages: {}, totalDurationSeconds: 0 };
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) return empty;
    const data = await res.json();
    const risk = data.risk_distribution || {};
    return {
      totalCalls: data.total_calls || 0,
      avgCompliance: data.avg_compliance_score || 0,
      fraudAlerts: (risk.high || 0) + (risk.critical || 0),
      pendingReviews: data.pending_reviews || 0,
      riskDistribution: risk,
      languages: data.languages || {},
      totalDurationSeconds: data.total_duration_seconds || 0,
    };
  } catch {
    return empty;
  }
}

export async function getReviewQueue(): Promise<ReviewItem[]> {
  try {
    const res = await fetch(`${API_BASE}/review-queue`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.items || []).map((item: Record<string, unknown>) => ({
      id: (item.call_id as string) || '',
      priority: (item.review_priority as number) ?? 5,
      riskType: ((item.overall_risk_level as string) || 'low').charAt(0).toUpperCase() + ((item.overall_risk_level as string) || 'low').slice(1),
      summary: (item.call_summary as string) || '',
      flags: (item.review_reasons as string[]) || [],
      date: item.date ? new Date((item.date as number) * 1000).toLocaleDateString('en-IN') : '',
      agent: 'Agent',
    }));
  } catch {
    return [];
  }
}

export interface PipelineProgress {
  call_id?: string;
  status: 'processing' | 'complete' | 'error' | 'idle' | 'unknown';
  current_stage?: string;
  current_stage_name?: string;
  stages_completed?: string[];
  error?: string;
  // Live timing data
  elapsed?: number;
  stage_times?: Record<string, number>;
  // After Stage 2 (quality)
  audio_quality_score?: number;
  audio_quality_flag?: string;
  audio_snr_db?: number;
  audio_speech_pct?: number;
  audio_duration?: number;
  audio_quality_components?: Record<string, number>;
  // After Stage 3 (transcription)
  transcript_segments?: number;
  transcript_language?: string;
}

export async function uploadCall(
  file: File,
  options?: { startTime?: number; endTime?: number }
): Promise<{ success: boolean; callId: string; message: string }> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('call_type', 'general');
    if (options?.startTime && options.startTime > 0) {
      formData.append('start_time', String(options.startTime));
    }
    if (options?.endTime && options.endTime > 0) {
      formData.append('end_time', String(options.endTime));
    }
    // Call backend directly to avoid Next.js proxy timeout on long pipeline runs
    const res = await fetch(`${BACKEND_DIRECT}/api/process`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.text();
      return { success: false, callId: '', message: err || 'Processing failed' };
    }
    const data = await res.json();
    return {
      success: true,
      callId: data.call_id || '',
      message: data.message || 'Pipeline started',
    };
  } catch (e) {
    return { success: false, callId: '', message: String(e) };
  }
}

export async function getProgress(callId?: string): Promise<PipelineProgress> {
  try {
    // Use call-specific endpoint when we have a call_id, generic fallback otherwise
    const url = callId
      ? `${API_BASE}/calls/${callId}/progress`
      : `${API_BASE}/progress`;
    const res = await fetch(url);
    if (!res.ok) return { status: 'idle' };
    return await res.json();
  } catch {
    return { status: 'idle' };
  }
}

export function getPipelineStages(): PipelineStage[] {
  return PIPELINE_STAGES;
}

export async function getObligations(callId: string): Promise<Obligation[]> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/entities`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.obligations || []).map((o: Record<string, unknown>) => ({
      text: (o.text as string) || '',
      speaker: (o.speaker as string) || 'unknown',
      type: (o.obligation_type as string) || (o.type as string) || '',
      strength: (o.strength as string) || 'vague',
      amount: (o.amount as string) || null,
      date: (o.date_referenced as string) || (o.date as string) || null,
      legallySignificant: (o.legally_significant as boolean) || false,
      segmentId: (o.segment_id as number) ?? 0,
    }));
  } catch {
    return [];
  }
}

export async function getTamperSignals(callId: string): Promise<{ signals: TamperSignal[]; risk: string }> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/compliance`);
    if (!res.ok) return { signals: [], risk: 'none' };
    const data = await res.json();
    const signals = (data.tamper_signals || []).map((t: Record<string, unknown>) => ({
      signalType: (t.signal_type as string) || 'unknown',
      description: (t.description as string) || '',
      timestamp: (t.timestamp as number) ?? 0,
      confidence: (t.confidence as number) ?? 0,
      severity: (t.severity as string) || 'low',
    }));
    return { signals, risk: (data.tamper_risk as string) || 'none' };
  } catch {
    return { signals: [], risk: 'none' };
  }
}

export async function submitReviewAction(
  callId: string,
  action: 'approve' | 'escalate' | 'reject',
  notes: string = '',
): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/review-action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, notes }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export function getAudioUrl(callId: string): string {
  return `${API_BASE}/calls/${callId}/audio`;
}

// ===== EXPORT FUNCTIONS =====

export type ExportFormat =
  | 'csv' | 'parquet' | 'jsonl'
  | 'training_intents' | 'training_sentiment' | 'training_entities';

const EXPORT_LABELS: Record<ExportFormat, string> = {
  csv: 'Call Summary (CSV)',
  parquet: 'Full Data (Parquet)',
  jsonl: 'Full Records (JSONL)',
  training_intents: 'Intent Training Pairs',
  training_sentiment: 'Sentiment Training Pairs',
  training_entities: 'Entity/NER Training Pairs',
};

export function getExportLabel(format: ExportFormat): string {
  return EXPORT_LABELS[format] || format;
}

export async function downloadExport(format: ExportFormat): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/export/${format}`);
    if (!res.ok) return false;

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ext = format === 'csv' ? '.csv' : format === 'parquet' ? '.parquet' : '.jsonl';
    a.href = url;
    a.download = `finvoice_${format}${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return true;
  } catch {
    return false;
  }
}

export async function downloadCallJson(callId: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/download`);
    if (!res.ok) return false;

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${callId}_analysis.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return true;
  } catch {
    return false;
  }
}

export async function downloadMaskedTranscript(callId: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/calls/${callId}/transcript/masked`);
    if (!res.ok) return false;

    const data = await res.json();
    const text = (data.segments || [])
      .map((s: Record<string, unknown>) => `[${s.speaker}] ${s.text}`)
      .join('\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${callId}_masked_transcript.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return true;
  } catch {
    return false;
  }
}

// ===== SYSTEM HEALTH =====

export interface HealthStatus {
  status: string;
  gpu: { device?: string; vram_total_mb?: number; vram_allocated_mb?: number };
  ollama: { status: string; models?: string[]; detail?: string };
  whisperx_loaded: boolean;
}

export async function getHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// ===== AI ANALYST =====
// Answered by the local LLM over locally processed records — no call data leaves
// the machine.

export async function queryAuditTrail(question: string, callId?: string): Promise<{ answer: string; error?: string }> {
  try {
    const payload: Record<string, string> = { question };
    if (callId) payload.call_id = callId;
    const res = await fetch(`${API_BASE}/audit/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.text();
      return { answer: '', error: err };
    }
    const data = await res.json();
    return { answer: data.answer || data.response || JSON.stringify(data) };
  } catch (e) {
    return { answer: '', error: String(e) };
  }
}

export async function complianceReason(excerpt: string, context: string, regulation: string): Promise<{ reasoning: string; error?: string }> {
  try {
    const res = await fetch(`${API_BASE}/compliance/reason`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ excerpt, context, regulation }),
    });
    if (!res.ok) {
      const err = await res.text();
      return { reasoning: '', error: err };
    }
    const data = await res.json();
    return { reasoning: data.reasoning || data.response || JSON.stringify(data) };
  } catch (e) {
    return { reasoning: '', error: String(e) };
  }
}
