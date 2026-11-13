import { useState } from 'react'
import { api, clipPlatforms, formatDuration } from '../api'
import type { Clip, Job } from '../types'

interface Props {
  job: Job
  clip: Clip
  selected: boolean
  onToggleSelect: () => void
  onOpen: () => void
  onToast: (message: string, kind?: 'success' | 'error') => void
}

export function ClipCard({ job, clip, selected, onToggleSelect, onOpen, onToast }: Props) {
  const [busy, setBusy] = useState(false)
  const platforms = clipPlatforms(clip)
  const primary = platforms[0] ?? 'tiktok'
  const busyState = clip.state === 'rendering' || clip.state === 'pending'

  async function download() {
    setBusy(true)
    try {
      // the URL is a plain file response, so a hidden anchor keeps the SPA state
      const anchor = document.createElement('a')
      anchor.href = api.downloadUrl(job.id, clip.id, primary)
      anchor.download = ''
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
    } catch (error) {
      onToast((error as Error).message, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <article className={`clip${selected ? ' selected' : ''}`} style={{ background: "#12121a", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "14px", padding: "12px", display: "flex", flexDirection: "column", gap: "8px" }}>
      <div
        className="thumb"
        onClick={onOpen}
        style={{ cursor: 'pointer' }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onOpen()
          }
        }}
      >
        {clip.poster ? (
          <img
            src={api.posterUrl(job.id, clip.id)}
            alt={`Clip ${clip.index + 1} thumbnail`}
            loading="lazy"
          />
        ) : (
          <div className="ph">{busyState ? 'rendering…' : 'no preview'}</div>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "10px 0 6px 0" }}>
          <span style={{ display: "inline-flex", padding: "4px 10px", borderRadius: 999, background: "rgba(212,175,55,0.15)", border: "1px solid rgba(212,175,55,0.4)", color: "#f5c542", fontSize: 12, fontWeight: 700 }}>Score {Math.round(clip.score * 100)}</span>
          {clip.speaker && <span style={{ display: "inline-flex", padding: "4px 10px", borderRadius: 999, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", color: "#e5e5e5", fontSize: 12, fontWeight: 600 }}>Speaker {clip.speaker.replace("Speaker ", "")}</span>}
          <span style={{ display: "inline-flex", padding: "4px 10px", borderRadius: 999, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", color: "#e5e5e5", fontSize: 12, fontWeight: 600 }}>Dur {formatDuration(clip.duration)}</span>
        </div>

        {clip.state !== 'ready' && (
          <div className="badges" style={{ top: 38, left: 8, right: 8 }}>
            {busyState ? (
              <span className="badge state-rendering">
                <span className="spinner" /> rendering
              </span>
            ) : clip.state === 'stale' ? (
              <span className="badge state-stale">edited — needs re-render</span>
            ) : clip.state === 'failed' ? (
              <span className="badge state-failed">render failed</span>
            ) : null}
          </div>
        )}
      </div>

      <div className="body">
        <div className="headline">{clip.title}</div>
        <div className="excerpt">{clip.hook || clip.text}</div>
        {clip.reasons.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "6px" }}>
            {clip.reasons.slice(0, 3).map((reason) => (
              <span key={reason} style={{ display: "inline-block", marginRight: "6px", marginBottom: "6px", padding: "2px 8px", borderRadius: "999px", background: "rgba(212,175,55,0.12)", border: "1px solid rgba(212,175,55,0.3)", color: "#f5c542", fontSize: "11px" }}>
                {reason}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="foot">
        <label className="check" style={{ flex: '0 0 auto' }} title="Include in bulk export">
          <input type="checkbox" checked={selected} onChange={onToggleSelect} />
        </label>
        <button className="small" onClick={onOpen}>
          Edit
        </button>
        <button className="small primary" onClick={download} disabled={busy || !platforms.length}>
          {platforms.length ? 'Download' : 'Not ready'}
        </button>
      </div>
    </article>
  )
}
