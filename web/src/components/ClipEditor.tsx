import { useEffect, useMemo, useRef, useState } from 'react'
import { api, clipPlatforms, formatDuration } from '../api'
import type { Clip, Job } from '../types'
import { ResultRecorder } from './ResultRecorder'

const BAR_LABELS: Record<string, string> = {
  text_hook: 'Hook',
  emotion: 'Emotion',
  humor: 'Humor',
  surprise: 'Surprise',
  energy: 'Energy',
  laughter: 'Laughter',
  pause_rhythm: 'Pacing',
  structure: 'Structure',
  topic: 'Topic',
}

interface Props {
  job: Job
  clip: Clip
  onClose: () => void
  onJobUpdate: (job: Job) => void
  onToast: (message: string, kind?: 'success' | 'error') => void
  minDuration?: number
  maxDuration?: number
}

export function ClipEditor({
  job,
  clip,
  onClose,
  onJobUpdate,
  onToast,
  minDuration = 15,
  maxDuration = 90,
}: Props) {
  const [title, setTitle] = useState(clip.title)
  const [start, setStart] = useState(clip.start)
  const [end, setEnd] = useState(clip.end)
  const [captions, setCaptions] = useState<Record<number, string>>(() =>
    Object.fromEntries(clip.cues.map((cue) => [cue.index, cue.text])),
  )
  const [saving, setSaving] = useState(false)
  const [previewBusy, setPreviewBusy] = useState(true)
  const [previewError, setPreviewError] = useState('')
  const [previewStage, setPreviewStage] = useState(2)
  const [videoDuration, setVideoDuration] = useState<number | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const previewRetryRef = useRef(0)
  const previewRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [dirty, setDirty] = useState(false)

  // Latest-values mirror for the poll merge below. The polling effect is
  // deliberately keyed on primitives only (clip?.id, clip.state) to preserve
  // the no-flicker contract, so it closes over the clip from mount time —
  // without this mirror, a title/start/end the user types DURING a
  // re-render gets overwritten by the frozen values on the next tick.
  const latestClipRef = useRef(clip)
  latestClipRef.current = clip

  // keep local state in sync when the parent swaps to a different clip.
  // Key strictly on the primitive id so job re-renders (new Clip object refs)
  // never reset local edits or the <video> playback.
  useEffect(() => {
    setTitle(clip.title)
    setStart(clip.start)
    setEnd(clip.end)
    setCaptions(Object.fromEntries(clip.cues.map((cue) => [cue.index, cue.text])))
    setDirty(false)
    setPreviewBusy(true)
    setPreviewError('')
    setPreviewStage(2)
    setVideoDuration(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clip?.id])

  // While the backend is re-rendering this clip, App.tsx pauses its own
  // /api/jobs polling + SSE (editingClip guard), so this drawer must poll
  // the job itself or it freezes at the initial PATCH snapshot ("starting 1%").
  // Dependencies are primitives only (clip?.id, clip.state) to preserve the
  // Task B no-flicker contract — never re-run on new Clip/Job object refs.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (clip.state !== 'rendering') return
    const jobId = job.id
    const clipId = clip.id
    let lastState: Clip['state'] = clip.state
    let lastProgress: number | null = clip.progress ?? null
    let lastNote: string | undefined = clip.progress_note
    let lastDuration: number = clip.duration
    const interval = setInterval(() => {
      api
        .getJob(jobId)
        .then((fresh) => {
          const freshClip = fresh.clips.find((c) => c.id === clipId)
          if (!freshClip) return
          const freshProgress = freshClip.progress ?? null
          if (
            freshClip.state === lastState &&
            freshProgress === lastProgress &&
            freshClip.progress_note === lastNote &&
            freshClip.duration === lastDuration
          ) {
            return
          }
          lastState = freshClip.state
          lastProgress = freshProgress
          lastNote = freshClip.progress_note
          lastDuration = freshClip.duration
          // Merge safely while rendering: forward the fresh job but preserve
          // the current clip's editable fields so a local title/start/end/
          // captions edit is never overwritten mid-render — only
          // state/progress/duration (plus outputs/error needed for the
          // ready state) move forward. Local component state is separate
          // and guarded by the [clip?.id] effect above, so this is
          // belt-and-braces for the prop values used by hasChanges/
          // duration/downloads. Read from latestClipRef (not the effect
          // closure) so edits typed while rendering survive the merge.
          // Once the fresh clip leaves `rendering`, do NOT preserve stale
          // start/end — allow fresh start/end/duration to flow through or
          // the inputs show edited values while the file is the new one
          // (Start/End/Length mismatch after Save & re-render).
          const isRendering = freshClip.state === 'rendering'
          const currentClip = latestClipRef.current
          const merged: Job = {
            ...fresh,
            clips: fresh.clips.map((c) =>
              c.id === clipId && isRendering
                ? {
                    ...c,
                    title: currentClip.title,
                    start: currentClip.start,
                    end: currentClip.end,
                    text: currentClip.text,
                    hook: currentClip.hook,
                    cues: currentClip.cues,
                    score: currentClip.score,
                    breakdown: currentClip.breakdown,
                    reasons: currentClip.reasons,
                    speaker: currentClip.speaker,
                  }
                : c,
            ),
          }
          onJobUpdate(merged)
        })
        .catch(() => undefined)
    }, 1500)
    return () => clearInterval(interval)
  }, [clip?.id, clip.state])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const duration = Math.max(0.1, end - start)
  // Hook prepend is disabled, so the stored clip.duration can be stale
  // (+2s hook tail). Length must match the actual file: use the loaded
  // <video> metadata duration when available, falling back to the
  // Start/End window before metadata loads.

  // A re-render is several distinct jobs, so the number alone would be
  // misleading. Name the phase as well as the percentage.
  const percent = Math.round(clip.progress ?? 0)
  const renderStatus =
    percent >= 100
      ? 'Finishing…'
      : `${clip.progress_note || 'starting'} · ${percent}%`
  const sourceDuration = job.source?.duration ?? 0
  const platforms = clipPlatforms(clip)
  const tooShort = duration < minDuration - 0.05
  const tooLong = duration > maxDuration + 0.05

  const previewPath = api.previewUrl(job.id, clip.id, 1)
  // Stage machine for persistent MEDIA_ERR_SRC_NOT_SUPPORTED (code 4).
  // Default is stage 2 (format=webm, VP9/Opus): WebView2 often lacks the H.264
  // patent codec, so MP4 original (stage 0) would fail after a slow download.
  // WebM is pre-warmed and Range-served, so stage 2 renders instantly with
  // no transcode wait. compat=1 (stage 1) is now the same royalty-free WebM
  // transcode (libvpx-vp9 + libopus, video/webm) - never an H.264 MP4 -
  // so it cannot code-4 for the same reason. Fallback order on code 4:
  // 2 (webm) -> 0 (MP4 original) -> 1 (compat WebM).
  // 0 = MP4 original (?v=1), 1 = compat=1 WebM, 2 = format=webm VP9.
  // previewPath already carries ?v=1, so later params append with &.
  // &r=${clip.duration}_${clip.state} makes the URL unique per render
  // generation — without it the browser HTTP cache serves the old file
  // (fixed ?v=1) after a re-render and playback stalls at the old length.
  const videoSrc =
    (window.location.origin && window.location.origin !== 'null'
      ? window.location.origin + previewPath
      : previewPath) + (previewStage === 1 ? '&compat=1' : previewStage === 2 ? '&format=webm' : '') + `&r=${clip.duration}_${clip.state}`

  const clearPreviewRetryTimer = () => {
    if (previewRetryTimerRef.current) {
      clearTimeout(previewRetryTimerRef.current)
      previewRetryTimerRef.current = null
    }
  }

  const handlePreviewError = () => {
    const code = videoRef.current?.error?.code
    // Code 4 (MEDIA_ERR_SRC_NOT_SUPPORTED) can mean the browser cannot play
    // the H.264 MP4 at all (even the old compat Main transcode was still H.264).
    // Default is webm (stage 2); fall back 2 -> 0 (MP4 original) ->
    // 1 (compat=1 WebM), retrying load() at each stage. The [videoSrc]
    // effect also reloads when the new src lands; the timed load() is a
    // backstop so the new src gets a chance to play even if the effect raced
    // the error. Stage 1 is terminal (no further fallback).
    clearPreviewRetryTimer()
    if (code === 4 && previewStage !== 1) {
      previewRetryRef.current += 1
      setPreviewBusy(true)
      setPreviewStage(previewStage === 2 ? 0 : 1)
      previewRetryTimerRef.current = setTimeout(() => {
        previewRetryTimerRef.current = null
        videoRef.current?.load()
      }, 400)
      return
    }
    setPreviewBusy(false)
    setPreviewError(`Preview load failed (code ${code ?? 'unknown'}) — the exported MP4 may be missing or unsupported by this browser.`)
  }

  const handlePreviewCanPlay = () => {
    previewRetryRef.current = 0
    clearPreviewRetryTimer()
    setPreviewBusy(false)
    setPreviewError('')
    // Video must stay paused at 0:00 until the user presses play.
    // No autoplay call here — preload="metadata" + controls is enough.
  }

  const lastRenderTokenRef = useRef<string>('')
  useEffect(() => {
    previewRetryRef.current = 0
    clearPreviewRetryTimer()
    setPreviewBusy(true)
    setPreviewError('')
    // New render generation (new &r= token baked into videoSrc) restarts the
    // fallback chain at stage 2 (webm default) — WebView2 often cannot play
    // H.264 at all, so webm is the instant path; MP4 fallbacks only try on
    // a code-4 error. Stage-driven src changes share the same token, so
    // they don't reset.
    const renderToken = `${clip.duration}_${clip.state}`
    if (lastRenderTokenRef.current !== renderToken) {
      lastRenderTokenRef.current = renderToken
      setPreviewStage(2)
    }
    // On ready transition (new render generation => new videoSrc) reset
    // playback to 0:00 and pause — never autoplay. The currentTime reset
    // is attempted before load() so a stalled old playback position
    // (e.g. stuck at 0:06 on the cached file) cannot carry over.
    if (clip.state === 'ready') {
      const el = videoRef.current
      if (el) {
        try {
          el.pause()
        } catch {
          /* noop */
        }
        try {
          el.currentTime = 0
        } catch {
          /* noop */
        }
      }
    }
    videoRef.current?.load()
    return () => {
      clearPreviewRetryTimer()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clip?.id, videoSrc, clip.state])

  const changedCaptions = useMemo(() => {
    const diff: Record<number, string> = {}
    for (const cue of clip.cues) {
      const next = captions[cue.index]
      if (next !== undefined && next !== cue.text) diff[cue.index] = next
    }
    return diff
  }, [captions, clip.cues])

  const hasChanges =
    dirty ||
    title !== clip.title ||
    Math.abs(start - clip.start) > 0.05 ||
    Math.abs(end - clip.end) > 0.05 ||
    Object.keys(changedCaptions).length > 0

  const breakdown = clip.breakdown
  const bars = Object.entries(BAR_LABELS).map(([key, label]) => ({
    key,
    label,
    value: (breakdown as unknown as Record<string, number>)[key] ?? 0,
  }))

  async function save(rerender: boolean) {
    setSaving(true)
    try {
      const updated = await api.patchClip(job.id, clip.id, {
        title: title.trim() || clip.title,
        start: Number(start.toFixed(2)),
        end: Number(end.toFixed(2)),
        captionText: Object.keys(changedCaptions).length ? changedCaptions : undefined,
        rerender,
      })
      onJobUpdate(updated)
      setDirty(false)
      if (rerender) {
        onToast('Re-rendering clip…')
        // The render happens on the server and can take a minute or more for
        // every platform variant. The app keeps polling while this clip reports
        // as rendering, so there is no point guessing a delay here; a single
        // timed refresh lands mid-render and leaves the drawer stuck.
      } else {
        onToast('Saved. Re-render to apply the changes to the video.', 'success')
      }
    } catch (error) {
      onToast((error as Error).message, 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="drawer-backdrop" onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(6, 6, 10, 0.88)', zIndex: 9998, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'none', padding: '20px', boxSizing: 'border-box' }}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Edit clip ${clip.index + 1}`} onClick={(e) => e.stopPropagation()} style={{ width: '1120px', maxWidth: '94vw', height: '86vh', maxHeight: '860px', background: '#0e0e14', border: '1px solid rgba(212, 175, 55, 0.35)', borderRadius: '20px', boxShadow: '0 30px 90px rgba(0, 0, 0, 0.95)', display: 'flex', flexDirection: 'column', overflow: 'hidden', color: '#f5f5f4' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 24px', background: '#14141d', borderBottom: '1px solid rgba(255, 255, 255, 0.08)', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '18px', fontWeight: 800, color: '#fff' }}>Clip {clip.index + 1}</span>
            <span style={{ padding: '3px 10px', borderRadius: '999px', background: 'rgba(34, 197, 94, 0.15)', border: '1px solid rgba(34, 197, 94, 0.45)', color: '#4ade80', fontSize: '11px', fontWeight: 700, textTransform: 'uppercase' }}>● {clip.state}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {clip.state === 'ready' && (
              <a href={api.downloadUrl(job.id, clip.id, platforms[0] ?? 'tiktok')} target="_blank" rel="noreferrer" style={{ padding: '8px 14px', borderRadius: '10px', background: 'rgba(212, 175, 55, 0.15)', border: '1px solid #d4af37', color: '#f5c542', fontSize: '12.5px', fontWeight: 700, textDecoration: 'none' }}>
                Open full exported clip ({formatDuration(clip.duration)}) ↗
              </a>
            )}
            <button autoFocus onClick={onClose} aria-label="Close clip editor" style={{ width: '34px', height: '34px', borderRadius: '10px', background: 'rgba(255, 255, 255, 0.06)', border: '1px solid rgba(255, 255, 255, 0.15)', color: '#fff', fontSize: '16px', cursor: 'pointer' }}>✕</button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '320px minmax(0, 1fr)', gap: '24px', padding: '20px 24px', flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', height: '100%', overflowY: 'auto', alignItems: 'center', paddingRight: '4px' }}>
            <div style={{ width: '282px', height: '500px', minHeight: '500px', background: '#050508', borderRadius: '14px', border: '1px solid rgba(212, 175, 55, 0.35)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, boxShadow: '0 12px 32px rgba(0, 0, 0, 0.8)' }}>
              <video
                ref={videoRef}
                key={clip.id}
                src={videoSrc}
                poster={api.posterUrl(job.id, clip.id)}
                controls
                preload="metadata"
                playsInline
                aria-label={`Preview of clip ${clip.index + 1}`}
                onCanPlay={handlePreviewCanPlay}
                onError={handlePreviewError}
                onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration)}
                style={{ width: '282px', height: '500px', minWidth: '282px', minHeight: '500px', objectFit: 'contain', background: '#000', display: 'block' }}
              />
            </div>
            {previewBusy && clip.state === 'ready' && <div className="faint" style={{ fontSize: 11 }}>Loading preview…</div>}
            {previewError && <div style={{ fontSize: 11, color: '#f87171', maxWidth: '282px', textAlign: 'center', overflowWrap: 'anywhere' }}>{previewError}</div>}

            <div style={{ width: '100%', background: '#14141d', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '12px', padding: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr 70px', gap: '10px', alignItems: 'end', boxSizing: 'border-box', flexShrink: 0 }}>
              <div>
                <div style={{ fontSize: 11, color: '#a8a29e', marginBottom: 4 }}>Start (s)</div>
                <input type="number" step={0.1} min={0} max={sourceDuration} value={start} onChange={(e) => setStart(Number(e.target.value))} style={{ width: '100%', padding: '8px 10px', borderRadius: '8px', background: '#09090f', border: '1px solid rgba(255,255,255,0.12)', color: '#fff', boxSizing: 'border-box' }} />
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#a8a29e', marginBottom: 4 }}>End (s)</div>
                <input type="number" step={0.1} min={start + 1} max={sourceDuration} value={end} onChange={(e) => setEnd(Number(e.target.value))} style={{ width: '100%', padding: '8px 10px', borderRadius: '8px', background: '#09090f', border: '1px solid rgba(255,255,255,0.12)', color: '#fff', boxSizing: 'border-box' }} />
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#a8a29e', marginBottom: 4 }}>Length</div>
                <div className="mono" style={{ padding: '10px 0', textAlign: 'center', color: tooShort || tooLong ? '#f87171' : '#f5c542', fontWeight: 700 }}>{formatDuration(videoDuration ?? duration)}</div>
              </div>
            </div>
            {(tooShort || tooLong) && <div className="desc" style={{ color: '#f87171', fontSize: 12, width: '100%' }}>{tooShort ? duration.toFixed(1) + 's is below the ' + minDuration + 's minimum.' : duration.toFixed(1) + 's is above the ' + maxDuration + 's maximum.'}</div>}

            <div style={{ display: 'flex', gap: '10px', width: '100%' }}>
              <button onClick={() => save(false)} disabled={saving || !hasChanges} style={{ flex: 1, padding: '12px', borderRadius: '10px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', color: '#fff', fontWeight: 700, cursor: 'pointer' }}>Save</button>
              <button onClick={() => save(true)} disabled={saving || clip.state === 'rendering' || tooShort || tooLong} style={{ flex: 1, padding: '12px', borderRadius: '10px', background: 'linear-gradient(135deg, #d4af37, #f5c542)', border: '1px solid #d4af37', color: '#08080a', fontWeight: 800, cursor: 'pointer' }}>{saving ? 'Saving…' : clip.state === 'rendering' ? 'Rendering…' : 'Save & re-render'}</button>
            </div>
            {clip.state === 'rendering' && <div style={{ fontSize: 12, color: '#a8a29e' }}>{renderStatus}</div>}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', overflowY: 'auto', paddingRight: '8px', minWidth: 0 }}>
            <div style={{ background: '#14141f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px', padding: '16px', flexShrink: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <label style={{ fontWeight: 700, fontSize: 14 }}>Top phrase</label>
                <span className="desc">{title.length}/46 characters render best</span>
              </div>
              <input type="text" value={title} maxLength={120} onChange={(e) => setTitle(e.target.value)} onBlur={() => setDirty(true)} style={{ width: '100%', padding: '10px 14px', borderRadius: '10px', background: '#09090f', border: '1px solid rgba(212,175,55,0.35)', color: '#fff', fontSize: '14px', fontWeight: 700, marginTop: '8px', boxSizing: 'border-box' }} />
            </div>

            <div style={{ background: '#14141f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px', padding: '16px', flexShrink: 0 }}>
              <label style={{ fontWeight: 700, fontSize: 14 }}>Captions ({clip.cues.length})</label>
              <span className="desc" style={{ display: 'block', marginTop: 4 }}>Edit any line — timings are preserved and the active word still highlights.</span>
              <div style={{ maxHeight: '250px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px', paddingRight: '6px' }}>
                {clip.cues.map((cue) => {
                  const value = captions[cue.index] ?? cue.text
                  const edited = value !== cue.text
                  return (
                    <div key={cue.index} style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: '10px', alignItems: 'center' }}>
                      <span className="mono" style={{ fontSize: 11, color: '#f5c542', background: 'rgba(212,175,55,0.1)', border: '1px solid rgba(212,175,55,0.25)', padding: '6px 8px', borderRadius: '8px', textAlign: 'center' }}>{cue.start.toFixed(1)}s</span>
                      <input type="text" value={value} onChange={(e) => setCaptions((prev) => ({ ...prev, [cue.index]: e.target.value }))} style={{ width: '100%', padding: '8px 12px', borderRadius: '8px', background: '#09090f', border: edited ? '1px solid #f5c542' : '1px solid rgba(255,255,255,0.1)', color: '#e7e5e4', boxSizing: 'border-box' }} />
                    </div>
                  )
                })}
              </div>
            </div>

            <div style={{ background: '#14141f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px', padding: '16px', flexShrink: 0 }}>
              <label style={{ fontWeight: 700, fontSize: 14 }}>Why this clip was chosen</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '10px', marginBottom: '14px' }}>
                {clip.reasons.map((reason) => (
                  <span key={reason} style={{ display: 'inline-block', marginRight: '8px', marginBottom: '8px', padding: '4px 10px', borderRadius: '999px', background: 'rgba(212,175,55,0.12)', border: '1px solid rgba(212,175,55,0.35)', color: '#f5c542', fontSize: '11.5px' }}>{reason}</span>
                ))}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '14px' }}>
                {bars.map((bar) => (
                  <div key={bar.key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', borderRadius: '8px', background: '#09090f', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <span style={{ color: '#a8a29e', fontSize: '12px' }}>{bar.label}</span>
                    <strong style={{ color: '#f5c542', fontSize: '12.5px', marginLeft: '8px' }}>{bar.value.toFixed(2)}</strong>
                  </div>
                ))}
              </div>
              <dl style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', margin: 0 }}>
                <div style={{ background: '#09090f', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <dt style={{ color: '#a8a29e', fontSize: 11 }}>Rank score</dt>
                  <dd style={{ color: '#f5c542', fontWeight: 800, margin: 0 }}>{Math.round(clip.score * 100)} / 100</dd>
                </div>
                <div style={{ background: '#09090f', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <dt style={{ color: '#a8a29e', fontSize: 11 }}>Source range</dt>
                  <dd style={{ color: '#fff', fontWeight: 700, margin: 0 }}>{formatDuration(clip.start)} – {formatDuration(clip.end)}</dd>
                </div>
                <div style={{ background: '#09090f', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <dt style={{ color: '#a8a29e', fontSize: 11 }}>Speaker</dt>
                  <dd style={{ color: '#fff', fontWeight: 700, margin: 0 }}>{clip.speaker ?? '—'}</dd>
                </div>
              </dl>
            </div>

            <div style={{ background: '#14141f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '14px', padding: '16px', flexShrink: 0 }}>
              <label style={{ fontWeight: 700, fontSize: 14 }}>Downloads</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '10px' }}>
                {platforms.length === 0 && <span className="faint">Nothing exported yet.</span>}
                {platforms.map((platform) => (
                  <a key={platform} href={api.downloadUrl(job.id, clip.id, platform)} download style={{ padding: '8px 14px', borderRadius: '8px', background: 'rgba(212,175,55,0.14)', border: '1px solid #d4af37', color: '#f5c542', textDecoration: 'none', fontSize: '12px', fontWeight: 700 }}>↓ {platform}</a>
                ))}
                <a href={api.subtitlesUrl(job.id, clip.id, 'srt')} download style={{ padding: '8px 14px', borderRadius: '8px', background: 'rgba(212,175,55,0.14)', border: '1px solid #d4af37', color: '#f5c542', textDecoration: 'none', fontSize: '12px', fontWeight: 700 }}>↓ .SRT</a>
                <a href={api.subtitlesUrl(job.id, clip.id, 'vtt')} download style={{ padding: '8px 14px', borderRadius: '8px', background: 'rgba(212,175,55,0.14)', border: '1px solid #d4af37', color: '#f5c542', textDecoration: 'none', fontSize: '12px', fontWeight: 700 }}>↓ .VTT</a>
              </div>
            </div>

            <ResultRecorder key={clip.id} jobId={job.id} clip={clip} onToast={onToast} />
          </div>
        </div>
      </aside>
    </div>
  )
}
