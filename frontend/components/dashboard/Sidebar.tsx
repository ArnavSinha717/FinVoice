'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getHealth, type HealthStatus } from '@/lib/api';

type View = 'overview' | 'upload' | 'calls' | 'review' | 'exports' | 'settings';

interface SidebarProps {
  activeView: View;
  onNavigate: (view: View) => void;
  open: boolean;
}

const NAV_ITEMS: { view: View; icon: string; label: string }[] = [
  { view: 'overview', icon: '\u25A3', label: 'Overview' },
  { view: 'upload', icon: '\u21E7', label: 'Upload' },
  { view: 'calls', icon: '\u260E', label: 'Calls' },
  { view: 'review', icon: '\u2691', label: 'Review Queue' },
  { view: 'exports', icon: '\u2913', label: 'Exports' },
  { view: 'settings', icon: '\u2699', label: 'Settings' },
];

export default function Sidebar({ activeView, onNavigate, open }: SidebarProps) {
  // Was a Backboard 'AI Memory' badge. Backboard is gone; this now reports
  // something the operator can act on — whether the local LLM is reachable.
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    getHealth().then(setHealth);
    const t = setInterval(() => getHealth().then(setHealth), 30000);
    return () => clearInterval(t);
  }, []);

  return (
    <aside className={`sidebar${open ? ' open' : ''}`}>
      <div className="sidebar-header">
        <Link href="/" className="sidebar-logo" style={{ fontFamily: 'var(--font-mono)' }}>
          Fin<span>Voice</span>
        </Link>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(item => (
          <button
            key={item.view}
            className={`sidebar-item${activeView === item.view ? ' active' : ''}`}
            onClick={() => onNavigate(item.view)}
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            <span className="sidebar-icon">{item.icon}</span>
            <span className="sidebar-label">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-status" style={{ marginBottom: 'var(--sp-2)' }}>
          <span className="status-dot" />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Pipeline active</span>
        </div>
        {health && (
          <div className="sidebar-status">
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: health.ollama?.status === 'healthy' ? 'var(--s5)' : 'var(--danger)',
              display: 'inline-block', flexShrink: 0,
              animation: health.ollama?.status === 'healthy' ? 'pulse 2s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
              {health.ollama?.status === 'healthy'
                ? `Local LLM ready${health.whisperx_loaded ? ' · ASR loaded' : ''}`
                : 'Local LLM offline'}
            </span>
          </div>
        )}
      </div>
    </aside>
  );
}

export type { View };
