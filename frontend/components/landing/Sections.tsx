'use client';

import Link from 'next/link';
import {
  ComplianceIcon, FraudIcon, DiarizationIcon, NERIcon, SentimentIcon, ExportIcon,
  LLMRoutingIcon, MemoryIcon, RAGIcon, EmbeddingsIcon, AuditTrailIcon,
  GPUAlphaIcon, GPUBravoIcon, CPUClusterIcon, CloudIcon,
} from './AnimatedIcons';

/* ===== FEATURES SECTION ===== */
export function Features() {
  const features = [
    { Icon: ComplianceIcon, title: 'Compliance Checking', desc: 'Automated RBI guideline verification across every call. Flags missing disclosures and consent gaps.', accent: 'var(--s4)', color: '#ef4444' },
    { Icon: FraudIcon, title: 'Fraud Detection', desc: 'Multi-signal anomaly detection for phishing, social engineering, and unauthorized transaction patterns.', accent: 'var(--s4)', color: '#ef4444' },
    { Icon: DiarizationIcon, title: 'Speaker Diarization', desc: 'Precise agent-customer separation with voice biometric matching and overlap handling.', accent: 'var(--s2)', color: '#10b981' },
    { Icon: NERIcon, title: 'Financial NER', desc: 'Custom-trained entity recognition for amounts, account numbers, dates, and banking terminology.', accent: 'var(--s3)', color: '#f59e0b' },
    { Icon: SentimentIcon, title: 'Sentiment Analysis', desc: 'Real-time emotion tracking across the call. Detects frustration, confusion, and satisfaction shifts.', accent: 'var(--s3)', color: '#f59e0b' },
    { Icon: ExportIcon, title: 'ML-Ready Export', desc: 'Structured JSON and CSV output designed for downstream ML training and analytics pipelines.', accent: 'var(--s5)', color: '#a855f7' },
  ];

  return (
    <section className="section" id="features">
      <div className="container">
        <div className="fade-up" style={{ textAlign: 'center' }}>
          <p className="section-label">Capabilities</p>
          <h2 className="section-title gradient-text">Built for Financial Audio Intelligence</h2>
        </div>
        <div className="features-grid">
          {features.map(f => (
            <div className="feature-card fade-up" key={f.title} style={{ '--accent': f.accent } as React.CSSProperties}>
              <div className="feature-icon">
                <f.Icon color={f.color} size={56} />
              </div>
              <h4>{f.title}</h4>
              <p style={{ color: 'var(--text-dim)', fontSize: '0.875rem', fontWeight: 300 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ===== INTELLIGENCE LAYER =====
   Was a "Powered by Backboard.io" section from the hackathon. That dependency is
   gone and the stack now runs entirely on-device, so the section describes what
   the pipeline actually does rather than a sponsor integration. */
export function IntelligenceLayer() {
  const cells = [
    { Icon: LLMRoutingIcon, title: 'Local LLM', desc: 'Qwen 2.5 via Ollama, on-device' },
    { Icon: MemoryIcon, title: 'Structured Records', desc: 'One typed CallRecord per call' },
    { Icon: RAGIcon, title: 'Compliance Engine', desc: 'RBI Fair Practice Code, KYC, DPDP 2023' },
    { Icon: EmbeddingsIcon, title: 'PII Redaction', desc: 'Aadhaar, PAN, UPI, IFSC with checksums' },
    { Icon: AuditTrailIcon, title: 'Audit Trail', desc: 'Every analyzer and degradation logged' },
  ];

  return (
    <section className="section intelligence-section" id="intelligence">
      <div className="container">
        <div className="fade-up" style={{ textAlign: 'center' }}>
          <p className="section-label" style={{ color: 'var(--s5)', borderColor: 'rgba(168,85,247,0.3)' }}>Runs entirely on-device</p>
          <h2 className="section-title gradient-text">Local Intelligence Layer</h2>
        </div>
        <div className="intelligence-grid fade-up">
          {cells.map((cell, i) => (
            <div key={cell.title} style={{ display: 'contents' }}>
              <div className="intelligence-cell">
                <div className="intelligence-icon">
                  <cell.Icon color="#a855f7" size={48} />
                </div>
                <h4>{cell.title}</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 300 }}>{cell.desc}</p>
              </div>
              {i < cells.length - 1 && (
                <div className="intelligence-connector"><div className="connector-line" /></div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ===== INFRASTRUCTURE SECTION ===== */
export function Infrastructure() {
  const nodes = [
    { Icon: GPUAlphaIcon, title: 'GPU \u00B7 WhisperX', desc: 'ASR with word-level alignment and speaker diarization', accent: 'var(--s3)', color: '#f59e0b', badgeClass: 'badge-warning', badgeText: 'GPU', spec: 'large-v3-turbo, INT8' },
    { Icon: GPUBravoIcon, title: 'GPU \u00B7 Emotion + LLM', desc: 'emotion2vec and Qwen 2.5 extraction, VRAM-aware scheduling', accent: 'var(--s2)', color: '#10b981', badgeClass: 'badge-success', badgeText: 'GPU', spec: 'fits in 6GB' },
    { Icon: CPUClusterIcon, title: 'CPU', desc: 'Audio preprocessing, rule engine, PII, exports, API serving', accent: 'var(--s1)', color: '#6366f1', badgeClass: 'badge-info', badgeText: 'CPU', spec: 'parallel analyzers' },
    { Icon: CloudIcon, title: 'No Cloud', desc: 'No call audio, transcript or PII leaves the machine', accent: 'var(--s5)', color: '#a855f7', badgeClass: 'badge-purple', badgeText: 'Local', spec: 'on-device' },
  ];

  return (
    <section className="section" id="infra">
      <div className="container">
        <div className="fade-up" style={{ textAlign: 'center' }}>
          <p className="section-label">Infrastructure</p>
          <h2 className="section-title gradient-text">Distributed Processing Architecture</h2>
        </div>
        <div className="infra-grid fade-up">
          {nodes.map((node, i) => (
            <div key={node.title} style={{ display: 'contents' }}>
              <div className="infra-node" style={{ '--node-accent': node.accent } as React.CSSProperties}>
                <div className="infra-node-icon">
                  <node.Icon color={node.color} size={48} />
                </div>
                <h4>{node.title}</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: 300 }}>{node.desc}</p>
                <div className="infra-specs">
                  <span className={`badge ${node.badgeClass}`}>{node.badgeText}</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{node.spec}</span>
                </div>
              </div>
              {i < nodes.length - 1 && (
                <div className="infra-connector">
                  <svg viewBox="0 0 80 4"><line x1="0" y1="2" x2="80" y2="2" stroke="var(--border-light)" strokeWidth="2" strokeDasharray="6 4" /></svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ===== CTA + FOOTER ===== */
export function CTAFooter() {
  return (
    <>
      <section className="section cta-section">
        <div className="container" style={{ textAlign: 'center' }}>
          <div className="fade-up">
            <h2 className="section-title gradient-text">Ready to process your first call?</h2>
            <p className="section-desc" style={{ margin: '0 auto var(--sp-8)', fontStyle: 'italic' }}>
              Upload an audio file and watch FinVoice transform it into structured, auditable intelligence in real time.
            </p>
            <Link href="/dashboard" className="btn btn-primary btn-lg">
              Launch Dashboard &rarr;
            </Link>
          </div>
        </div>
      </section>
      <footer className="footer">
        <div className="container footer-inner">
          <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem', letterSpacing: '0.1em', textTransform: 'uppercase' as const }}>
            FIN<span style={{ color: 'var(--orange)' }}>VOICE</span> &copy; 2025
          </span>
          <div className="footer-links">
            <a href="#" style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>GitHub</a>
            <a href="#" style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Docs</a>
            <a href="#" style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Contact</a>
          </div>
        </div>
      </footer>
    </>
  );
}
