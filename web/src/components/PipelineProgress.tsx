import { useEffect, useRef } from 'react'
import type { Job, PipelineEvent } from '../types'

const PHASES: { key: string; label: string }[] = [
  { key: 'probe', label: 'Probing' },
  { key: 'transcribe', label: 'Transcribing (faster-whisper)' },
  { key: 'score', label: 'Virality Scoring' },
  { key: 'framing', label: 'AI Framing & Hooks' },
  { key: 'render', label: 'Rendering & Export' },
]

// Map job.stage (9 stages) to one of the 5 phases.
function phaseIndex(stage: string): number {
  if (stage === 'ingest') return 0
  if (stage === 'transcribe') return 1
  if (stage === 'analyze' || stage === 'highlights' || stage === 'length') return 2
  if (stage === 'subtitles' || stage === 'phrases') return 3
  if (stage === 'render' || stage === 'export') return 4
  if (stage === 'done') return 5
  return 0
}

interface Props {
  job: Job
  events: PipelineEvent[]
  onCancel: () => void
  lostConnection?: boolean
  onResume?: () => void
}

export function PipelineProgress({ job, events, onCancel, lostConnection, onResume }: Props) {
  const terminal = job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'
  const phasePct = [8, 30, 55, 75, 90, 100]
  const fallbackPct = phasePct[Math.min(phaseIndex(job.stage), 5)] ?? 8
  const percent = Math.round(((job.progress && job.progress > 0) ? job.progress : fallbackPct / 100) * 100)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = logRef.current
    if (!el) return
    const id = requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
    })
    return () => cancelAnimationFrame(id)
  }, [events])

  const currentPhase = phaseIndex(job.stage)

  const originalName = (job.storage as Record<string,string> | undefined)?.original_name || job.source?.filename

  return (
    <div className="card">
      <div className="card-head">
        <h2>
          {terminal
            ? job.status === 'completed'
              ? 'Pipeline complete'
              : job.status === 'failed'
                ? 'Pipeline failed'
                : 'Cancelled'
            : 'Working on it'}
        </h2>
        <span className="faint mono" style={{ marginLeft: 8 }}>
          {originalName}
        </span>
        <div className="spacer" />
        {!terminal && (
          <button className="small danger" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      <div className="card-body stack">
        {lostConnection && !terminal ? (
          <div className="notice warn">
            Lost contact with the server, so this is the last progress it reported.
          </div>
        ) : null}
        {job.status === 'failed' && onResume ? (
          <div className="notice info">
            <span className="grow">
              {job.error ?? 'The run was interrupted.'}
            </span>
            <button className="small" onClick={onResume}>
              Finish export
            </button>
          </div>
        ) : null}

        {/* Stage + percentage above bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: '#f5c542', fontWeight: 800, fontSize: 14, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            {terminal
              ? (job.status === 'completed' ? '✓ Complete' : job.status)
              : (() => {
                  const stageNames: Record<number, string> = {
                    0: 'Reading video',
                    1: 'Transcribing audio (faster-whisper)',
                    2: 'Analyzing audio',
                    3: 'Scoring highlights',
                    4: 'Rendering clips',
                  }
                  const name = stageNames[currentPhase] ?? 'Processing'
                  return (<><span className="cf-live-bolt" aria-hidden="true">⚡</span><span>{`${name}...`}</span></>)
                })()}
          </span>
          <span style={{ color: '#f5c542', fontWeight: 800, fontSize: 18 }}>{percent}%</span>
        </div>

        {/* Gold progress bar — animated shimmer proves loading, not stuck */}
        <div style={{
          background: '#181822',
          height: '12px',
          borderRadius: '999px',
          border: '1px solid rgba(212,175,55,0.25)',
          overflow: 'hidden',
          position: 'relative',
          margin: '12px 0 16px 0',
          boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.6)',
        }}>
          <div
            className="cf-progress-fill-animated"
            style={{
              height: '100%',
              width: `${percent}%`,
              borderRadius: '999px',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <div className="cf-progress-shine" />
          </div>
        </div>

        {/* Stage pills */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
          {PHASES.map((phase, i) => {
            const done = terminal ? i < 5 : i < currentPhase
            const active = !terminal && i === currentPhase
            return (
              <span
                key={phase.key}
                style={{
                  padding: '6px 12px',
                  borderRadius: 20,
                  fontSize: 12,
                  fontWeight: 700,
                  border: '1px solid',
                  borderColor: active ? '#f5c542' : done ? 'rgba(212,175,55,0.4)' : 'rgba(255,255,255,0.1)',
                  background: active ? 'rgba(212,175,55,0.15)' : done ? 'rgba(212,175,55,0.08)' : 'rgba(255,255,255,0.03)',
                  color: active ? '#f5c542' : done ? '#d4af37' : '#78716c',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                {done ? '✓ ' : ''}{phase.label}
              </span>
            )
          })}
        </div>

        {/* Terminal log */}
        {events.length > 0 && (
          <div
            ref={logRef}
            className="cf-terminal-log"
            style={{
              background: '#08080c',
              border: '1px solid rgba(212, 175, 55, 0.22)',
              borderRadius: 12,
              padding: '14px 16px',
              fontFamily: "'Consolas', 'JetBrains Mono', monospace",
              fontSize: '12px',
              lineHeight: '1.65',
              color: '#d6d3d1',
              minHeight: '180px',
              height: '200px',
              maxHeight: '240px',
              overflowY: 'auto',
              flexShrink: 0,
              marginTop: '12px',
              boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.85)',
            }}
          >
            {events.map((event, i) => (
              <div key={`${event.ts}-${i}`} style={{ marginBottom: 2 }}>
                <span style={{ color: '#52525b' }}>
                  [{new Date(event.ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}]
                </span>{' '}
                <span style={{ color: event.level === 'error' ? '#f87171' : event.level === 'success' ? '#4ade80' : '#a8a29e' }}>
                  {event.message}
                </span>
              </div>
            ))}
          </div>
        )}

        {job.warnings.length > 0 && (
          <div className="stack" style={{ gap: 6 }}>
            {job.warnings.map((warning, i) => (
              <div key={`${warning}-${i}`} className="notice warn">⚠ {warning}</div>
            ))}
          </div>
        )}

        {job.error && (
          <div className="notice error">
            <strong>Error:</strong> {job.error}
          </div>
        )}
      </div>
    </div>
  )
}