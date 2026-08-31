'use client';

import { useState, useRef, useEffect } from 'react';
import { queryAuditTrail, getStats, getHealth, type HealthStatus } from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const QUICK_QUERIES = [
  'Show all compliance violations',
  'Which calls had fraud signals?',
  'What are the most common customer complaints?',
  'Summary of all high-risk calls',
  'Which calls had emotional escalation?',
];

export default function AiAnalyst() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [callCount, setCallCount] = useState<number | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth().then(setHealth);
    getStats().then(s => setCallCount(s.totalCalls));
  }, []);

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (query?: string) => {
    const q = query || input.trim();
    if (!q || loading) return;

    const userMsg: Message = { role: 'user', content: q, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    const result = await queryAuditTrail(q);
    const assistantMsg: Message = {
      role: 'assistant',
      content: result.error ? `Error: ${result.error}` : result.answer,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, assistantMsg]);
    setLoading(false);
  };

  return (
    <>
      {/* FAB */}
      <button className="ai-fab" onClick={() => setOpen(!open)} title="AI Analyst">
        {open ? '\u2715' : '\u2728'}
      </button>

      {/* Panel */}
      <div className={`ai-panel ${open ? 'open' : ''}`}>
        <div className="ai-panel-header">
          <div>
            <h3 style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', marginBottom: 2 }}>AI Analyst</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: health?.ollama?.status === 'healthy' ? 'var(--success)' : 'var(--danger)',
                display: 'inline-block',
              }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-dim)' }}>
                {health?.ollama?.status === 'healthy'
                  ? `Local LLM ready${callCount !== null ? ` — ${callCount} call${callCount === 1 ? '' : 's'} indexed` : ''}`
                  : 'Local LLM offline — start Ollama'}
              </span>
            </div>
          </div>
          <button onClick={() => setOpen(false)} style={{ color: 'var(--text-dim)', fontSize: '1.2rem', background: 'none', border: 'none', cursor: 'pointer' }}>
            &times;
          </button>
        </div>

        <div className="ai-panel-body" ref={bodyRef}>
          {messages.length === 0 && (
            <div style={{ padding: 'var(--sp-4)' }}>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: 'var(--sp-4)', lineHeight: 1.6 }}>
                Ask questions about your call portfolio. Answered on-device by the local LLM over your processed calls — no call data leaves this machine.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-2)' }}>
                {QUICK_QUERIES.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSubmit(q)}
                    style={{
                      fontFamily: 'var(--font-mono)', fontSize: '0.75rem', padding: 'var(--sp-2) var(--sp-3)',
                      background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                      borderRadius: 8, color: 'var(--text-dim)', cursor: 'pointer',
                      textAlign: 'left', transition: 'all 0.2s',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--s5)'; e.currentTarget.style.color = 'var(--text)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-dim)'; }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} style={{
              marginBottom: 'var(--sp-3)',
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            }}>
              <div style={{
                maxWidth: '85%', padding: 'var(--sp-2) var(--sp-3)',
                borderRadius: msg.role === 'user' ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                background: msg.role === 'user' ? 'rgba(99,102,241,0.15)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${msg.role === 'user' ? 'rgba(99,102,241,0.3)' : 'var(--border)'}`,
              }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-dim)', marginTop: 4, textAlign: 'right' }}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)', padding: 'var(--sp-2)', color: 'var(--text-dim)' }}>
              <span style={{ animation: 'pulse 1.5s ease-in-out infinite', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                Analyzing across all calls...
              </span>
            </div>
          )}
        </div>

        <div className="ai-panel-input">
          <div style={{ display: 'flex', gap: 'var(--sp-2)' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
              placeholder="Ask about your calls..."
              style={{
                flex: 1, padding: 'var(--sp-2) var(--sp-3)',
                background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)',
                borderRadius: 8, color: 'var(--text)', fontFamily: 'var(--font-mono)',
                fontSize: '0.8rem', outline: 'none',
              }}
            />
            <button
              onClick={() => handleSubmit()}
              disabled={loading || !input.trim()}
              style={{
                padding: 'var(--sp-2) var(--sp-3)', borderRadius: 8,
                background: loading || !input.trim() ? 'rgba(255,255,255,0.03)' : 'linear-gradient(135deg, var(--s5), var(--s1))',
                color: '#fff', fontFamily: 'var(--font-mono)', fontSize: '0.8rem',
                border: 'none', cursor: loading || !input.trim() ? 'default' : 'pointer',
                opacity: loading || !input.trim() ? 0.4 : 1,
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
