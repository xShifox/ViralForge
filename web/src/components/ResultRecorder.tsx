import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Calibration, Clip } from '../types'

/** The five ratings a clip can be given, worst to best. */
const OUTCOMES: { value: number; label: string }[] = [
  { value: 0, label: 'Flopped' },
  { value: 1, label: 'Below avg' },
  { value: 2, label: 'Average' },
  { value: 3, label: 'Did well' },
  { value: 4, label: 'Went well' },
]

interface Props {
  jobId: string
  clip: Clip
  onToast: (message: string, kind?: 'success' | 'error') => void
}

/**
 * Record how a published clip actually did.
 *
 * This is the only thing that makes the scoring honest. The weights the tool
 * ships with are a guess; without results they stay a guess, and the score
 * means only "higher than the other clips in this video". Nothing here claims
 * the score predicts anything - it records what happened so the next run can
 * weight the signals that actually separated your wins from your misses.
 *
 * Deliberately a five point rating rather than a number typed in: it is quick
 * enough to do for every clip, and only the extremes are used for fitting, so
 * a rough "average" is honest rather than a fabricated 0.62.
 */
export function ResultRecorder({ jobId, clip, onToast }: Props) {
  const [views, setViews] = useState('')
  const [saving, setSaving] = useState(false)
  const [recorded, setRecorded] = useState<{ outcome: number | null; views: number | null } | null>(null)
  const [status, setStatus] = useState<Calibration | null>(null)

  useEffect(() => {
    let alive = true
    api
      .calibration()
      .then((data) => {
        if (alive) setStatus(data)
      })
      .catch(() => undefined)
    return () => {
      alive = false
    }
  }, [jobId])

  // Reset transient UI when the clip under the recorder changes so a rating
  // recorded for one clip can't appear (or submit) on another.
  useEffect(() => {
    setViews('')
    setRecorded(null)
  }, [clip.id])

  async function record(outcome: number | null, withViews = false) {
    const parsed = withViews && views.trim() ? Number(views) : null
    if (withViews && (parsed === null || Number.isNaN(parsed))) {
      onToast('Enter a view count, or leave the box empty', 'error')
      return
    }
    if (parsed !== null && parsed < 0) {
      onToast('View count cannot be negative', 'error')
      return
    }
    setSaving(true)
    try {
      const result = await api.recordResult(jobId, clip.id, {
        outcome,
        views: parsed,
      })
      setRecorded({ outcome: result.outcome, views: result.views })
      setStatus(result.status)
      if (outcome !== null) setViews('')
      onToast('Recorded — the scoring weights learn from these', 'success')
    } catch (error) {
      onToast((error as Error).message, 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="field">
      <label>How did this clip do?</label>
      <div className="row wrap">
        {OUTCOMES.map((option) => (
          <button
            key={option.value}
            className={`small${recorded?.outcome === option.value ? ' on' : ''}`}
            disabled={saving}
            onClick={() => record(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <input
          type="number"
          min={0}
          placeholder="views (optional)"
          value={views}
          onChange={(e) => setViews(e.target.value)}
          style={{ width: 150 }}
        />
        <button className="small" disabled={saving} onClick={() => record(null, true)}>
          Save views
        </button>
      </div>
      {status && <span className="desc">{calibrationNote(status)}</span>}
    </div>
  )
}

export function calibrationNote(status: Calibration): string {
  if (status.applied) {
    const moved = Object.entries(status.shifts)
      .filter(([, scale]) => Math.abs(scale - 1) > 0.01)
      .sort((a, b) => Math.abs(b[1] - 1) - Math.abs(a[1] - 1))
      .slice(0, 3)
      .map(([name, scale]) => `${name.replace(/_/g, ' ')} ${scale.toFixed(2)}x`)
    const detail = moved.length ? ` Weighted up or down: ${moved.join(', ')}.` : ''
    return (
      `Scoring is fitted from ${status.used} of your ${status.records} recorded results.${detail}` +
      ' A higher weight means that signal was high on the clips that worked and low on the ones that did not.'
    )
  }
  if (status.records === 0) {
    return (
      'Rate a few clips as they perform and the scoring weights will be fitted to your history. ' +
      `(${status.needed} rated clips needed)`
    )
  }
  return `Using the built-in weights for now: ${status.reason}.`
}
