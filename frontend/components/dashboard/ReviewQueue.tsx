'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { getReviewQueue, submitReviewAction, type ReviewItem } from '@/lib/api';

export default function ReviewQueue() {
  const router = useRouter();
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);

  useEffect(() => {
    getReviewQueue().then(setItems).finally(() => setLoading(false));
  }, []);

  const handleReviewAction = async (callId: string, action: 'approve' | 'escalate' | 'reject') => {
    setActionInProgress(`${callId}-${action}`);
    try {
      await submitReviewAction(callId, action);
      setItems(prev => prev.filter(item => item.id !== callId));
    } catch (err) {
      console.error(`Review action failed:`, err);
    } finally {
      setActionInProgress(null);
    }
  };

  const handleCardClick = (callId: string) => {
    router.push(`/dashboard/calls/${callId}`);
  };

  const filtered = items.filter(item => {
    if (priorityFilter === 'high' && item.priority > 2) return false;
    if (priorityFilter === 'medium' && item.priority !== 3) return false;
    if (priorityFilter === 'low' && item.priority < 4) return false;
    if (riskFilter !== 'all' && item.riskType.toLowerCase() !== riskFilter.toLowerCase()) return false;
    return true;
  });

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 'var(--sp-12)', color: 'var(--text-dim)' }}>
        <span style={{ fontFamily: 'var(--font-mono)' }}>Loading review queue...</span>
      </div>
    );
  }

  const highCount = items.filter(i => i.priority <= 2).length;
  const medCount = items.filter(i => i.priority === 3).length;
  const lowCount = items.filter(i => i.priority >= 4).length;

  return (
    <>
      {/* Summary strip */}
      {items.length > 0 && (
        <div className="call-stats-strip" style={{ marginBottom: 'var(--sp-4)' }}>
          <div className="call-stat-chip">
            <span className="call-stat-label">Total</span>
            <span className="call-stat-value">{items.length}</span>
          </div>
          <div className="call-stat-chip">
            <span className="call-stat-label">High</span>
            <span className="call-stat-value" style={{ color: highCount > 0 ? 'var(--danger)' : undefined }}>{highCount}</span>
          </div>
          <div className="call-stat-chip">
            <span className="call-stat-label">Medium</span>
            <span className="call-stat-value" style={{ color: medCount > 0 ? 'var(--warning)' : undefined }}>{medCount}</span>
          </div>
          <div className="call-stat-chip">
            <span className="call-stat-label">Low</span>
            <span className="call-stat-value">{lowCount}</span>
          </div>
        </div>
      )}

      <div className="review-filters">
        <select
          className="filter-select"
          style={{ fontFamily: 'var(--font-mono)' }}
          value={priorityFilter}
          onChange={e => setPriorityFilter(e.target.value)}
        >
          <option value="all">All Priorities</option>
          <option value="high">High (7-10)</option>
          <option value="medium">Medium (4-6)</option>
          <option value="low">Low (1-3)</option>
        </select>
        <select
          className="filter-select"
          style={{ fontFamily: 'var(--font-mono)' }}
          value={riskFilter}
          onChange={e => setRiskFilter(e.target.value)}
        >
          <option value="all">All Risk Types</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
          <option value="Critical">Critical</option>
        </select>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          {filtered.length} of {items.length} shown
        </span>
      </div>

      <div className="review-list">
        {filtered.length === 0 ? (
          <div style={{ padding: 'var(--sp-8)', textAlign: 'center', color: 'var(--text-dim)' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
              {items.length === 0 ? 'No calls pending review.' : 'No items match the selected filters.'}
            </p>
          </div>
        ) : (
          filtered.map(item => {
            const color = item.priority <= 2 ? 'var(--danger)' : item.priority === 3 ? 'var(--warning)' : 'var(--success)';
            const prioClass = item.priority <= 2 ? 'badge-danger' : item.priority === 3 ? 'badge-warning' : 'badge-success';

            return (
              <div className="review-card" key={item.id} style={{ '--review-color': color } as React.CSSProperties}>
                <div className="review-card-header" onClick={() => handleCardClick(item.id)} style={{ cursor: 'pointer' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--sp-2)' }}>
                    <h4 style={{ fontFamily: 'var(--font-mono)' }}>{item.id}</h4>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-muted)' }}>{item.riskType}</span>
                  </div>
                  <span className={`priority-score badge ${prioClass}`} style={{ fontFamily: 'var(--font-mono)' }}>P{item.priority}</span>
                </div>
                <p className="review-card-summary" onClick={() => handleCardClick(item.id)} style={{ cursor: 'pointer' }}>{item.summary}</p>
                <div className="review-flags">
                  {item.flags.map(f => (
                    <span className="review-flag" key={f} style={{ fontFamily: 'var(--font-mono)' }}>{f}</span>
                  ))}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--sp-2)', position: 'relative' }}>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{item.date} &middot; {item.agent}</span>
                  <div className="review-actions">
                    <button
                      className="review-btn approve"
                      style={{ fontFamily: 'var(--font-mono)' }}
                      disabled={actionInProgress !== null}
                      onClick={() => handleReviewAction(item.id, 'approve')}
                    >
                      {actionInProgress === `${item.id}-approve` ? '...' : 'Approve'}
                    </button>
                    <button
                      className="review-btn escalate"
                      style={{ fontFamily: 'var(--font-mono)' }}
                      disabled={actionInProgress !== null}
                      onClick={() => handleReviewAction(item.id, 'escalate')}
                    >
                      {actionInProgress === `${item.id}-escalate` ? '...' : 'Escalate'}
                    </button>
                    <button
                      className="review-btn reject"
                      style={{ fontFamily: 'var(--font-mono)' }}
                      disabled={actionInProgress !== null}
                      onClick={() => handleReviewAction(item.id, 'reject')}
                    >
                      {actionInProgress === `${item.id}-reject` ? '...' : 'Reject'}
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </>
  );
}
