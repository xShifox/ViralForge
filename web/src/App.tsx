import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, clipPlatforms, formatDuration, subscribeToJob, type UploadOptions } from './api'
import { ClipCard } from './components/ClipCard'
import { ClipEditor } from './components/ClipEditor'
import { Dropzone } from './components/Dropzone'
import { FolderPicker } from './components/FolderPicker'
import { StartGate } from './components/StartGate'
import { OptionsPanel, type AudioPatch } from './components/OptionsPanel'
import { PipelineProgress } from './components/PipelineProgress'
import type { Clip, Health, Job, PipelineEvent, Settings } from './types'

// Stable-polling budgets. The jobs poll used to tick every 3s and
// re-rendered the whole 2-col grid (dropzone, boosters, clip grid,
// options panel) on every tick, reset the terminal log's scrollTop via
// the parent re-render, and re-mounted the clip modal <video> — the
// visible flicker/scatter. 5s for jobs and 30s for health/gpu keeps
// progress fresh without the repaint storm.
const JOBS_POLL_INTERVAL_MS = 5000
const HEALTH_POLL_INTERVAL_MS = 30000
// Cap the terminal log so a long run cannot grow an unbounded array
// that re-renders (and re-measures layout) on every SSE frame.
const MAX_EVENTS = 200

// Memoized so App re-renders that do not touch the progress primitives
// (stage/status/progress) or the batched log skip the whole card.
const MemoPipelineProgress = memo(PipelineProgress)

const DEFAULT_OPTIONS: UploadOptions = {
  maxClips: 6,
  minDuration: 15,
  maxDuration: 90,
  platforms: ['tiktok'],
  useLlm: false,
  layout: 'crop',
  watermarkText: '',
  logoPath: '',
  showProgressBar: true,
  storage: ['local'],
  trackTalker: false,
  startMiddleHook: false,
  audioDucking: false,
  outputFolder: '',
  videoEncoder: 'auto',
  speechDevice: 'auto',
  uppercaseCaptions: false,
  highlightSpokenWord: false,
  fadeCaptions: false,
  captionSize: 100,
  headlineSize: 100,
  musicBed: '',
  duckingDepth: -12,
  cropMode: 'auto',
  cropX: 50,
  cropY: 50,
  cropZoom: 100,
}

interface Toast {
  id: number
  message: string
  kind: 'info' | 'success' | 'error'
}

interface PendingFile {
  file: File
  folder: string
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [settings, setSettings] = useState<Settings | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [pendingFile, setPendingFile] = useState<PendingFile | null>(null)
  const [editingClip, setEditing] = useState<Clip | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [resumable, setResumable] = useState<Job | null>(null)
  const [folderPrompt, setFolderPrompt] = useState(false)
  const [running, setRunning] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadPercent, setUploadPercent] = useState<number | null>(null)
  const [events, setEvents] = useState<PipelineEvent[]>([])
  const [lostConnection, setLostConnection] = useState(false)
  const [fatalError, setFatalError] = useState<string | null>(null)
  const [activePreset, setActivePreset] = useState<'viral' | 'podcast' | 'pip' | null>(null)

  const nextToastId = useRef(1)

  // Toast helper — defined before the effects below that reference it.
  const onToast = useCallback((message: string, kind: 'info' | 'success' | 'error' = 'info') => {
    const id = nextToastId.current++
    setToasts((prev) => [...prev, { id, message, kind }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 5000)
  }, [])

  // Latest-clip mirror so the SSE handlers below never close over stale
  // props. Declared (and synced in the render body) BEFORE the subscription
  // effects: effects run in order on mount, so a ref declared after them is
  // still null for the first burst of events when the editor opens.
  const editingClipRef = useRef<Clip | null>(null)
  editingClipRef.current = editingClip

  // Batched event log. SSE can fire many frames per second; calling
  // setEvents + setJob per frame re-rendered the grid per frame and
  // reset the log scroll. Events now buffer in a ref and flush once
  // per animation frame, capped at MAX_EVENTS so the log (and its
  // 200px box) stays stable. Progress keeps only the latest patch.
  const pendingEventsRef = useRef<PipelineEvent[]>([])
  const pendingPatchRef = useRef<{ progress: number | null; stage: string } | null>(null)
  const flushRafRef = useRef<number | null>(null)

  const flushPendingEvents = useCallback(() => {
    flushRafRef.current = null
    const batch = pendingEventsRef.current
    const patch = pendingPatchRef.current
    pendingEventsRef.current = []
    pendingPatchRef.current = null
    if (batch.length) {
      setEvents((prev) => {
        const next = prev.concat(batch)
        return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next
      })
    }
    if (patch) {
      if (patch.progress !== null && patch.progress !== undefined) {
        setJob((prev) => (prev ? { ...prev, progress: patch.progress as number, stage: patch.stage as Job['stage'] } : prev))
      } else {
        setJob((prev) => (prev ? { ...prev, stage: patch.stage as Job['stage'] } : prev))
      }
      // `running` gates the gold bar + log visibility; a terminal stage
      // must flip it off in the same flush or the bar lingers a frame.
      setRunning(patch.stage !== 'done')
    }
  }, [])

  const scheduleEventFlush = useCallback(() => {
    if (flushRafRef.current !== null) return
    flushRafRef.current = window.requestAnimationFrame(flushPendingEvents)
  }, [flushPendingEvents])

  // Cancel any queued flush on unmount so a late rAF never setStates
  // after unmount. Buffers are cleared per job (see subscription cleanup).
  useEffect(() => {
    return () => {
      if (flushRafRef.current !== null) {
        window.cancelAnimationFrame(flushRafRef.current)
        flushRafRef.current = null
      }
    }
  }, [])

  // Initial data load
  useEffect(() => {
    let cancelled = false

    api
      .health()
      .then((data) => {
        if (!cancelled) setHealth(data)
      })
      .catch((error: Error) => {
        if (!cancelled) onToast(`Could not reach the server: ${error.message}`, 'error')
      })

    api
      .settings()
      .then((data) => {
        if (!cancelled) setSettings(data)
      })
      .catch(() => undefined)

    // Rehydrate a still-running job from a previous session so the
    // "A job is still running" banner actually has something to show.
    api
      .listJobs()
      .then(({ jobs }) => {
        if (cancelled) return
        const live = jobs.find((j) => j.status === 'running' || j.status === 'queued')
        if (live) setResumable(live)
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [onToast])

  // Health / GPU context — every 30s, NOT on the 5s job cadence.
  // Health carries the GPU / encoder labels in the header + options
  // panel; polling it every job tick re-rendered the whole grid for a
  // label that changes in minutes. Single interval, in-flight guard,
  // cleared on unmount (no overlapping timers). Silent on failure —
  // the initial load already surfaced unreachable-server toasts.
  // Paused while the clip modal is open so a health re-render can never
  // remount the modal <video>.
  useEffect(() => {
    if (editingClip) return
    let cancelled = false
    let inFlight = false
    const interval = setInterval(() => {
      if (cancelled || inFlight) return
      inFlight = true
      api
        .health()
        .then((data) => {
          if (cancelled) return
          // Bail when nothing meaningful changed so an identical
          // 30s tick does not re-render the grid at all.
          setHealth((prev) => {
            if (
              prev &&
              prev.ok === data.ok &&
              prev.gpu?.name === data.gpu?.name &&
              prev.encoder?.active === data.encoder?.active &&
              prev.asr?.device === data.asr?.device
            ) {
              return prev
            }
            return data
          })
        })
        .catch(() => undefined)
        .finally(() => {
          inFlight = false
        })
    }, HEALTH_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editingClip?.id])

  // Job subscription
  useEffect(() => {
    if (!job) return

    let cancelled = false

    const unsubscribe = subscribeToJob(
      job.id,
      0,
      (event: PipelineEvent) => {
        if (cancelled) return
        // While the clip editor modal is open, don't touch `job` — a second
        // SSE consumer would remount the modal and reset <video> + flash.
        if (editingClipRef.current) return

        // Buffer — do NOT setState per frame. The batch flushes once per
        // rAF below, capped at 200, so a burst of SSE frames costs one
        // render, not N. Latest progress wins for the job patch.
        pendingEventsRef.current.push(event)
        if (pendingEventsRef.current.length > MAX_EVENTS * 2) {
          pendingEventsRef.current = pendingEventsRef.current.slice(-MAX_EVENTS)
        }
        pendingPatchRef.current = { progress: event.progress ?? null, stage: event.stage }
        scheduleEventFlush()
      },
      () => {
        // The stream is finished. Re-read the job from the server so the
        // clips gallery gets the full rendered job, not the empty starting
        // object returned by the upload call.
        if (!cancelled && !editingClipRef.current) {
          api.getJob(job.id)
            .then((fresh) => {
              setJob(fresh)
              // An SSE drop while the job is still running must NOT flip the
              // UI to the bare spinner — keep `running` so the gold bar + log
              // stay visible and the fallback poll keeps updating them.
              const stillRunning = fresh.status === 'running' || fresh.status === 'queued'
              setRunning(stillRunning)
              if (!stillRunning) onToast('Job finished — clips are ready', 'success')
            })
            .catch((error: Error) => {
              // A silent catch here left the UI on the gold spinner forever.
              setLostConnection(true)
              onToast(`Stream ended but the job could not be refreshed: ${error.message}`, 'error')
            })
        }
      },
      (connected) => {
        setLostConnection(!connected)
      }
    )

    return () => {
      cancelled = true
      // Drop any unflushed batch for the job that just left — it belongs
      // to the old stream and must not append to the next job's log.
      pendingEventsRef.current = []
      pendingPatchRef.current = null
      if (flushRafRef.current !== null) {
        window.cancelAnimationFrame(flushRafRef.current)
        flushRafRef.current = null
      }
      unsubscribe()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id])

  // Fallback poll: 5s (not 3s), single interval, in-flight guard, clean
  // cleanup, stops for good on terminal. Paused while the Clip Modal is
  // open — prevents App re-renders that remounted the modal <video> and
  // caused WebView2 to flash/scatter.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!job || editingClip) return
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled' || job.stage === 'done') return
    let stopped = false
    let failures = 0
    let inFlight = false
    let interval: ReturnType<typeof setInterval> | null = null
    interval = setInterval(() => {
      // No overlapping requests: a slow response must not stack with
      // the next tick and double-setJob.
      if (stopped || inFlight) return
      inFlight = true
      api
        .getJob(job.id)
        .then((fresh) => {
          if (stopped) return
          failures = 0
          // Bail when the poll returned what we already have — an
          // identical tick must not re-render the grid or the modal.
          setJob((prev) => {
            if (
              prev &&
              prev.id === fresh.id &&
              prev.status === fresh.status &&
              prev.stage === fresh.stage &&
              prev.progress === fresh.progress &&
              prev.message === fresh.message &&
              prev.error === fresh.error &&
              prev.clips.length === fresh.clips.length &&
              prev.warnings.length === fresh.warnings.length
            ) {
              return prev
            }
            return fresh
          })
          // Keep the progress card in sync even if SSE died: the gold bar and
          // terminal log are gated on `running`, so a dead EventSource used to
          // leave the UI on the bare "Processing video…" spinner forever.
          setRunning(fresh.status === 'running' || fresh.status === 'queued')
          // Terminal: stop polling for good so no further setJob re-renders.
          if (fresh.status === 'completed' || fresh.status === 'failed' || fresh.status === 'cancelled' || fresh.stage === 'done') {
            stopped = true
            if (interval) clearInterval(interval)
            interval = null
          }
        })
        .catch((error: Error) => {
          // A job record that never answers must surface, not poll silently
          // forever. After ~25 s of failures, bail out loudly.
          failures += 1
          if (failures >= 5) {
            stopped = true
            if (interval) clearInterval(interval)
            interval = null
            setLostConnection(true)
            onToast(`Lost contact with the server for this job: ${error.message}`, 'error')
          }
        })
        .finally(() => {
          inFlight = false
        })
    }, JOBS_POLL_INTERVAL_MS)
    return () => {
      stopped = true
      if (interval) clearInterval(interval)
      interval = null
    }
  }, [job?.id, editingClip?.id])

  // Editing clip resolved from job so re-renders update independently of modal state
  const editingClipResolved = editingClip
    ? job?.clips.find((c) => c.id === editingClip.id) ?? editingClip
    : null

  // Debounced settings persistence. Every slider tick previously fired a
  // GET+PUT (or PUT) pair immediately — on a drag that is dozens of
  // concurrent writes racing each other, and an earlier value can land last
  // and silently win. Coalesce to one write after 400 ms of quiet.
  const persistTimer = useRef<number | null>(null)
  const persistPatch = useRef<Record<string, unknown>>({})
  const runPersist = useCallback(async () => {
    persistTimer.current = null
    const patch = persistPatch.current
    persistPatch.current = {}
    try {
      const updated = await api.saveSettings(patch)
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSettings(updated)
    } catch (error) {
      onToast((error as Error).message, 'error')
    }
  }, [onToast])

  const schedulePersist = useCallback((patch: Record<string, unknown>) => {
    // Cheap deep merge for nested section dicts (detect/render/brand).
    for (const [key, value] of Object.entries(patch)) {
      const current = persistPatch.current[key]
      persistPatch.current[key] =
        value !== null && typeof value === 'object' && !Array.isArray(value) &&
        current !== null && typeof current === 'object' && !Array.isArray(current)
          ? { ...(current as Record<string, unknown>), ...(value as Record<string, unknown>) }
          : value
    }
    if (persistTimer.current !== null) window.clearTimeout(persistTimer.current)
    persistTimer.current = window.setTimeout(runPersist, 400)
  }, [runPersist])

  // Flush any pending debounced write — called before uploads so the job
  // always picks up the settings the user is looking at.
  const flushPersist = useCallback(() => {
    if (persistTimer.current !== null) {
      window.clearTimeout(persistTimer.current)
      persistTimer.current = null
    }
    if (Object.keys(persistPatch.current).length) {
      const patch = persistPatch.current
      persistPatch.current = {}
      return api.saveSettings(patch)
        .then((updated) => { setSettings(updated); return updated })
        .catch(() => undefined)
    }
    return Promise.resolve()
  }, [])

  const saveAudio = useCallback(async (patch: AudioPatch) => {
    // The backend merges partial patches field-by-field, so a bare PUT of
    // the changed fields is enough — the old read-modify-write GET+PUT pair
    // raced with handleOptionsChange's own PUT and could resurrect stale
    // values.
    schedulePersist({ render: { ...patch } })
  }, [schedulePersist])

  const handleOptionsChange = (newOptions: UploadOptions) => {
    // Update local state synchronously so sliders move instantly.
    setOptions(newOptions)

    // Caption + media-engine choices live in the render section; the rest ride
    // on the detect section. Map camelCase frontend keys to snake_case backend
    // keys so the partial patch actually lands.
    const renderPatch: Record<string, unknown> = {}
    if (newOptions.musicBed !== undefined) renderPatch.music_path = newOptions.musicBed
    if (newOptions.duckingDepth !== undefined) renderPatch.duck_db = newOptions.duckingDepth
    if (newOptions.sfxFolder) renderPatch.sfx_folder = newOptions.sfxFolder
    if (newOptions.brollFolder) renderPatch.broll_folder = newOptions.brollFolder
    if (newOptions.fadeCaptions !== undefined) renderPatch.fade_captions = newOptions.fadeCaptions
    if (newOptions.uppercaseCaptions !== undefined) renderPatch.subtitle_uppercase = newOptions.uppercaseCaptions
    if (newOptions.highlightSpokenWord !== undefined) renderPatch.word_highlight = newOptions.highlightSpokenWord
    if (newOptions.captionSize !== undefined) renderPatch.subtitle_scale = newOptions.captionSize
    if (newOptions.headlineSize !== undefined) renderPatch.title_scale = newOptions.headlineSize
    if (newOptions.layout !== undefined) renderPatch.layout = newOptions.layout
    if (newOptions.audioDucking !== undefined) renderPatch.duck_music = newOptions.audioDucking
    if (newOptions.cropMode !== undefined) renderPatch.crop_mode = newOptions.cropMode
    if (newOptions.cropX !== undefined) renderPatch.crop_x = newOptions.cropX
    if (newOptions.cropY !== undefined) renderPatch.crop_y = newOptions.cropY
    if (newOptions.cropZoom !== undefined) renderPatch.crop_zoom = newOptions.cropZoom

    const detectPatch: Record<string, unknown> = {
      max_clips: newOptions.maxClips,
      min_duration: newOptions.minDuration,
      max_duration: newOptions.maxDuration,
      use_llm: newOptions.useLlm,
      track_talker: newOptions.trackTalker,
      start_middle_hook: newOptions.startMiddleHook,
    }

    const brandPatch: Record<string, unknown> = {
      watermark_text: newOptions.watermarkText,
      logo_path: newOptions.logoPath,
      show_progress_bar: newOptions.showProgressBar,
    }

    schedulePersist({ detect: detectPatch, render: renderPatch, brand: brandPatch, platforms: newOptions.platforms })
  }

  const [options, setOptions] = useState<UploadOptions>(DEFAULT_OPTIONS)
  const hydratedRef = useRef(false)

  // Hydrate options from settings ONCE on initial load. Do NOT overwrite on
  // subsequent settings changes (which would reset in-flight user edits).
  useEffect(() => {
    if (!settings || hydratedRef.current) return
    hydratedRef.current = true
    const d = (settings.detect ?? {}) as Record<string, unknown>
    const r = (settings.render ?? {}) as Record<string, unknown>
    const b = (settings.brand ?? {}) as Record<string, unknown>
    setOptions((prev) => ({
      ...prev,
      maxClips: Number(d.max_clips ?? prev.maxClips),
      minDuration: Number(d.min_duration ?? prev.minDuration),
      maxDuration: Number(d.max_duration ?? prev.maxDuration),
      platforms: Array.isArray(settings.platforms) ? settings.platforms : prev.platforms,
      useLlm: Boolean(d.use_llm ?? prev.useLlm),
      trackTalker: Boolean(d.track_talker ?? prev.trackTalker),
      startMiddleHook: Boolean(d.start_middle_hook ?? prev.startMiddleHook),
      layout: (r.layout as UploadOptions['layout']) ?? prev.layout,
      watermarkText: String(b.watermark_text ?? prev.watermarkText),
      logoPath: String(b.logo_path ?? prev.logoPath),
      showProgressBar: Boolean(b.show_progress_bar ?? prev.showProgressBar),
      // Audio Ducking always starts OFF — do not hydrate from persisted duck_music.
      audioDucking: false,
      duckingDepth: Number(r.duck_db ?? prev.duckingDepth),
      musicBed: String(r.music_path ?? prev.musicBed),
      sfxFolder: String(r.sfx_folder ?? prev.sfxFolder),
      brollFolder: String(r.broll_folder ?? prev.brollFolder),
      uppercaseCaptions: Boolean(r.subtitle_uppercase ?? prev.uppercaseCaptions),
      // Karaoke Glow always starts OFF on app start — do not hydrate from
      // persisted word_highlight (manual toggle still saves via handleOptionsChange).
      highlightSpokenWord: false,
      fadeCaptions: Boolean(r.fade_captions ?? prev.fadeCaptions),
      captionSize: Number(r.subtitle_scale ?? prev.captionSize),
      headlineSize: Number(r.title_scale ?? prev.headlineSize),
      cropMode: ((r.crop_mode as UploadOptions['cropMode']) ?? prev.cropMode) || 'auto',
      cropX: Number(r.crop_x ?? prev.cropX ?? 50),
      cropY: Number(r.crop_y ?? prev.cropY ?? 50),
      cropZoom: Number(r.crop_zoom ?? prev.cropZoom ?? 100),
      outputFolder: settings.output_folder ?? prev.outputFolder,
    }))
  }, [settings])

  const platformLabels = useMemo(() => {
    if (!health) return {}
    return Object.fromEntries(
      Object.entries(health.platforms).map(([key, platform]) => [
        key,
        platform.label ?? key,
      ])
    )
  }, [health])

  const readyClips = useMemo(() => {
    return (job?.clips ?? []).filter((clip) => clip.state === 'ready')
  }, [job])

  const selectedCount = selected.size

  const startJob = useCallback(async (file: File, options: UploadOptions, folder: string) => {
    // Persist any in-flight edits first so the job reads the settings the
    // user is actually looking at (debounced PUT would otherwise lag).
    await flushPersist()
    setFatalError(null)
    setRunning(true)
    setLostConnection(false)
    setUploadPercent(0)
    setEvents([])
    pendingEventsRef.current = []
    pendingPatchRef.current = null
    if (flushRafRef.current !== null) {
      window.cancelAnimationFrame(flushRafRef.current)
      flushRafRef.current = null
    }
    setSelected(new Set())
    setUploading(true)
    try {
      const created = await api.upload(
        file,
        { ...options, outputFolder: folder },
        (percent) => setUploadPercent(percent),
      )
      setJob(created)
      setUploadPercent(null)
      onToast('Upload complete — the pipeline is running', 'success')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed'
      setFatalError(message)
      onToast(message, 'error')
      setJob(null)
      setRunning(false)
      setUploadPercent(null)
    } finally {
      setUploading(false)
    }
  }, [flushPersist, onToast])

  const jobId = job?.id
  const cancel = useCallback(() => {
    if (!jobId) return
    const id = jobId
    // Optimistic: show Cancelled instantly instead of waiting for two
    // sequential HTTP round-trips while the backend worker winds down.
    setJob((prev) => (prev ? { ...prev, status: 'cancelled', message: 'Cancelled' } : prev))
    setRunning(false)
    onToast('Cancelling…', 'info')
    // Confirm with the server in the background (also flips the backend
    // record to cancelled immediately so polling stays consistent).
    api.cancelJob(id)
      .then(() => api.getJob(id))
      .then((fresh) => {
        setJob(fresh)
        setRunning(false)
        onToast('Job cancelled', 'info')
      })
      .catch((error: Error) => onToast(error.message, 'error'))
  }, [jobId, onToast])

  const resume = useCallback(() => {
    if (!jobId) return
    api.resumeJob(jobId)
      .then((fresh) => {
        setJob(fresh)
        setRunning(true)
        setLostConnection(false)
        setUploadPercent(null)
        onToast('Export resumed', 'info')
      })
      .catch((error: Error) => onToast(error.message, 'error'))
  }, [jobId, onToast])

  const cancelUpload = useCallback(() => {
    setPendingFile(null)
    setFatalError(null)
  }, [])

  const chooseFolder = useCallback((folder: string) => {
    setPendingFile((prev) => (prev ? { ...prev, folder } : prev))
  }, [])

  const confirmStart = useCallback(() => {
    if (running || job?.status === 'running' || job?.status === 'queued') return
    const pending = pendingFile
    if (!pending) return
    setPendingFile(null)
    // Use the LIVE options, not a snapshot taken at drop time: StartGate
    // shows the live options on screen, and running the snapshot would make
    // the job obey settings the user no longer sees.
    startJob(pending.file, options, pending.folder)
  }, [pendingFile, startJob, running, job?.status, options])

  const startWithFile = useCallback((file: File) => {
    if (running || job?.status === 'running' || job?.status === 'queued') return
    setFatalError(null)
    setPendingFile({
      file,
      folder: settings?.output_folder || '',
    })
  }, [running, job?.status, settings])

  const applyPreset = useCallback((preset: 'viral' | 'podcast' | 'pip') => {
    setActivePreset((prev) => (prev === preset ? null : preset))
    if (activePreset === preset) return
    const patches: Record<'viral' | 'podcast' | 'pip', Partial<UploadOptions>> = {
      viral: { startMiddleHook: true, highlightSpokenWord: true, maxDuration: 60 },
      podcast: { layout: 'split', minDuration: 60, maxDuration: 90, maxClips: 12 },
      pip: { layout: 'pip', cropMode: 'auto' },
    }
    handleOptionsChange({ ...options, ...patches[preset] })
    onToast(`Preset applied: ${preset === 'viral' ? 'Viral Hook' : preset === 'podcast' ? 'Podcast Split' : 'Cinema PiP'}`, 'success')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options, activePreset, onToast])

  const handleClear = useCallback(() => {
    // Best-effort: stop any live job on the server so a stale worker
    // cannot resurrect it on the next poll. Clears the resumable banner
    // job too, not just the one currently on screen.
    const live = job ?? resumable
    if (live) {
      if (running || live.status === 'running' || live.status === 'queued') {
        api.cancelJob(live.id).catch(() => undefined)
      }
      // Delete the job workspace from data/jobs so old renders never fill
      // the disk. Best-effort: the backend refuses while the worker still
      // owns a running job, so a delete racing a cancel simply no-ops.
      api.deleteJob(live.id).catch(() => undefined)
    }
    // Drop any pending debounced settings write — it carries pre-clear
    // values that would overwrite the defaults persisted below.
    if (persistTimer.current !== null) {
      window.clearTimeout(persistTimer.current)
      persistTimer.current = null
    }
    persistPatch.current = {}
    setJob(null)
    setEvents([])
    pendingEventsRef.current = []
    pendingPatchRef.current = null
    if (flushRafRef.current !== null) {
      window.cancelAnimationFrame(flushRafRef.current)
      flushRafRef.current = null
    }
    setSelected(new Set())
    setEditing(null)
    setRunning(false)
    setUploadPercent(null)
    setPendingFile(null)
    setFolderPrompt(false)
    setFatalError(null)
    setToasts([])
    setResumable(null)
    setLostConnection(false)
    setActivePreset(null)
    setOptions({ ...DEFAULT_OPTIONS })
    // Persist the defaults through the existing saveSettings path so the
    // next job starts clean even after a reload (best-effort).
    const defaults = DEFAULT_OPTIONS
    api.saveSettings({
      detect: {
        max_clips: defaults.maxClips,
        min_duration: defaults.minDuration,
        max_duration: defaults.maxDuration,
        use_llm: defaults.useLlm,
        track_talker: defaults.trackTalker,
        start_middle_hook: defaults.startMiddleHook,
      },
      render: {
        music_path: defaults.musicBed,
        duck_db: defaults.duckingDepth,
        sfx_folder: defaults.sfxFolder ?? '',
        broll_folder: defaults.brollFolder ?? '',
        fade_captions: defaults.fadeCaptions,
        subtitle_uppercase: defaults.uppercaseCaptions,
        word_highlight: defaults.highlightSpokenWord,
        subtitle_scale: defaults.captionSize,
        title_scale: defaults.headlineSize,
        layout: defaults.layout,
        duck_music: defaults.audioDucking,
        crop_mode: defaults.cropMode,
        crop_x: defaults.cropX,
        crop_y: defaults.cropY,
        crop_zoom: defaults.cropZoom,
      },
      brand: {
        watermark_text: defaults.watermarkText,
        logo_path: defaults.logoPath,
        show_progress_bar: defaults.showProgressBar,
      },
      platforms: defaults.platforms,
    })
      .then((updated) => setSettings(updated))
      .catch(() => undefined)
    onToast('Cleared — back to defaults', 'success')
  }, [job, resumable, running, onToast])

  // Stable primitives for the progress card. PipelineProgress only reads
  // stage/status/progress (+ error/warnings/source/storage for its
  // banners), so its `job` prop keeps the PREVIOUS identity while those
  // primitives are unchanged. A health tick, slider drag, toast, or
  // identical 5s poll then skips the card entirely via memo().
  const pipelineJob = useMemo(() => job, [
    job?.id,
    job?.status,
    job?.stage,
    job?.progress,
    job?.message,
    job?.error,
    job?.warnings,
    job?.source,
    job?.storage,
  ])

  const pipelineProps = useMemo(() => {
    if (!pipelineJob) return null
    return {
      job: pipelineJob,
      events,
      onCancel: cancel,
      lostConnection,
      onResume: resume,
    }
  }, [pipelineJob, events, cancel, lostConnection, resume])

  return (
    <div className="premium-app">
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', padding: '0 24px', height: '60px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #f5c542 0%, #d4af37 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#08080a',
            fontWeight: 900,
            fontSize: '18px',
            boxShadow: '0 0 14px rgba(212, 175, 55, 0.45)'
          }}>
            ▶
          </div>
          <span style={{
            fontSize: '20px',
            fontWeight: 800,
            letterSpacing: '0.5px',
            color: '#f5c542',
            textShadow: '0 0 16px rgba(212, 175, 55, 0.35)'
          }}>
            ViralForge
          </span>
        </div>
        
        <div className="cf-preset-dock">
          <span style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '1px', color: '#78716c', padding: '0 10px', textTransform: 'uppercase' }}>PRESETS</span>
          <button
            type="button"
            className={`cf-preset-pill ${activePreset === 'viral' ? 'active' : ''}`}
            aria-pressed={activePreset === 'viral'}
            onClick={() => applyPreset('viral')}
          >
            Viral Hook
          </button>
          <button
            type="button"
            className={`cf-preset-pill ${activePreset === 'podcast' ? 'active' : ''}`}
            aria-pressed={activePreset === 'podcast'}
            onClick={() => applyPreset('podcast')}
          >
            Podcast Split
          </button>
          <button
            type="button"
            className={`cf-preset-pill ${activePreset === 'pip' ? 'active' : ''}`}
            aria-pressed={activePreset === 'pip'}
            onClick={() => applyPreset('pip')}
          >
            Cinema PiP
          </button>
        </div>
        
        <div style={{ width: '1px', height: '24px', background: 'rgba(255, 255, 255, 0.12)', margin: '0 14px' }}></div>
        
        <div className="cf-system-status">
          {health && (
            <>
              <div className="cf-system-ready">
                <div className="cf-pulse-dot"></div>
                System Ready
              </div>
              {job && (
                <div className="cf-system-processing">
                  <span className="cf-pulse-dot processing"></span>
                  Processing
                </div>
              )}
            </>
          )}
          <button
            type="button"
            className="cf-gold-btn sm"
            onClick={handleClear}
            title="Reset the screen to defaults"
            style={{
              background: 'transparent',
              color: '#f5c542',
              border: '1px solid rgba(212, 175, 55, 0.45)',
              borderRadius: 999,
              margin: 0,
              padding: '6px 14px',
            }}
          >
            Clear
          </button>
        </div>
      </header>

      <main className="cf-main-grid" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 380px', gap: '24px', height: 'calc(100vh - 60px)', padding: '20px 28px', boxSizing: 'border-box', overflow: 'hidden', alignItems: 'start' }}>
        <section className="cf-left-col" style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', minHeight: 0, overflowY: 'auto', overflowX: 'hidden', paddingRight: '10px', paddingBottom: (running || job?.status === 'running' || job?.status === 'queued') ? '12px' : '120px', boxSizing: 'border-box', overscrollBehavior: 'contain', scrollBehavior: 'auto', scrollbarGutter: 'stable' }}>
          {resumable && !job && (
            <div className="notice">
              <div>
                <strong>A job is still running</strong>
                <div className="faint" style={{ fontSize: 12 }}>
                  {resumable.source?.filename ?? 'A video'} · {resumable.stage} ·{' '}
                  {Math.round((resumable.progress ?? 0) * 100)}%
                </div>
              </div>
              <div className="row">
                <button
                  className="small"
                  onClick={() => {
                    setJob(resumable)
                    setRunning(true)
                    setLostConnection(false)
                    setUploadPercent(null)
                    setEvents([])
                    setResumable(null)
                  }}
                >
                  Show it
                </button>
                <button className="small ghost" onClick={() => setResumable(null)}>
                  Dismiss
                </button>
              </div>
            </div>
          )}

          {fatalError && (
            <div className="cf-error-banner" role="alert">
              <div className="cf-error-copy">
                <strong>Something went wrong</strong>
                <span>{fatalError}</span>
              </div>
              <button className="cf-error-dismiss" onClick={() => setFatalError(null)}>Dismiss</button>
            </div>
          )}

          {pendingFile ? (
            <StartGate
              file={pendingFile.file}
              folder={pendingFile.folder}
              options={options}
              platformLabels={platformLabels}
              onChangeFolder={() => setFolderPrompt(true)}
              onStart={confirmStart}
              onDiscard={cancelUpload}
            />
          ) : (
            <>
              <Dropzone
                uploading={uploading}
                busy={running || job?.status === 'running' || job?.status === 'queued'}
                uploadPercent={uploadPercent}
                onFile={startWithFile}
                options={options}
              />
              <div className={`cf-boosters-grid${running || job?.status === 'running' || job?.status === 'queued' ? ' compact' : ''}`}>
                <button
                  type="button"
                  className={`cf-booster-card ${options.trackTalker ? 'active' : ''}`}
                  aria-pressed={options.trackTalker}
                  onClick={() => handleOptionsChange({...options, trackTalker: !options.trackTalker})}
                >
                  <span className="cf-booster-badge">{options.trackTalker ? 'ON' : 'OFF'}</span>
                  <span className="cf-booster-icon" aria-hidden="true">🎯</span>
                  <span className="cf-booster-label">TrackTalker™</span>
                  <span className="cf-booster-sub">AI face &amp; speaker tracking camera</span>
                </button>
                <button
                  type="button"
                  className={`cf-booster-card ${options.startMiddleHook ? 'active' : ''}`}
                  aria-pressed={options.startMiddleHook}
                  onClick={() => handleOptionsChange({...options, startMiddleHook: !options.startMiddleHook})}
                >
                  <span className="cf-booster-badge">{options.startMiddleHook ? 'ON' : 'OFF'}</span>
                  <span className="cf-booster-icon" aria-hidden="true">🪝</span>
                  <span className="cf-booster-label">Start Middle Hook™</span>
                  <span className="cf-booster-sub">Prepend best middle moment as opener</span>
                </button>
                <button
                  type="button"
                  className={`cf-booster-card ${options.audioDucking ? 'active' : ''}`}
                  aria-pressed={options.audioDucking}
                  onClick={() => handleOptionsChange({...options, audioDucking: !options.audioDucking})}
                >
                  <span className="cf-booster-badge">{options.audioDucking ? 'ON' : 'OFF'}</span>
                  <span className="cf-booster-icon" aria-hidden="true">🔊</span>
                  <span className="cf-booster-label">Audio Ducking</span>
                  <span className="cf-booster-sub">Lower music during speech — needs music bed</span>
                </button>
                <button
                  type="button"
                  className={`cf-booster-card ${options.highlightSpokenWord ? 'active' : ''}`}
                  aria-pressed={options.highlightSpokenWord}
                  onClick={() => handleOptionsChange({...options, highlightSpokenWord: !options.highlightSpokenWord})}
                >
                  <span className="cf-booster-badge">{options.highlightSpokenWord ? 'ON' : 'OFF'}</span>
                  <span className="cf-booster-icon" aria-hidden="true">✨</span>
                  <span className="cf-booster-label">Karaoke Glow</span>
                  <span className="cf-booster-sub">Word-by-word highlighted captions</span>
                </button>
              </div>
            </>
          )}
          {folderPrompt && (
            <FolderPicker
              initial={pendingFile?.folder || settings?.output_folder || ''}
              onPick={(folder) => {
                chooseFolder(folder)
                setFolderPrompt(false)
              }}
              onCancel={() => {
                cancelUpload()
                setFolderPrompt(false)
              }}
            />
          )}

          {job && pipelineProps && (running || job.status === 'running' || job.status === 'queued') && (
            <div
              className="cf-pipeline-pin"
              style={{
                position: 'sticky',
                bottom: 0,
                zIndex: 5,
                flexShrink: 0,
                marginTop: '4px',
                paddingTop: '10px',
                paddingBottom: '2px',
                background: '#0d0d14',
                borderTop: '1px solid rgba(212, 175, 55, 0.18)',
                borderRadius: '16px 16px 0 0',
                boxShadow: '0 -12px 24px rgba(0,0,0,0.55)',
              }}
            >
              <MemoPipelineProgress {...pipelineProps} />
            </div>
          )}

          {job && !running && job.status !== 'running' && job.status !== 'queued' && job.clips.length === 0 && (
            <section className="card">
              <div className="card-head">
                <h2>Result</h2>
              </div>
              <div className="card-body">
                {job.status === 'cancelled' ? (
                  <div className="notice info">This job was cancelled.</div>
                ) : (
                  <div className="empty">
                    <div className="big">?</div>
                    <div>No clips were found in this video.</div>
                    {job.error && (
                      <div className="notice error" style={{ marginTop: 14, textAlign: 'left' }}>
                        {job.error}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>
          )}

          {job && job.clips.length > 0 && (
            <section className="card" style={{ overflow: 'visible', height: 'auto', maxHeight: 'none', flexShrink: 0 }}>
              <div className="card-head">
                <h2>
                  {readyClips.length} clip{readyClips.length === 1 ? '' : 's'} ready
                </h2>
                {job.source && (
                  <span className="faint mono">
                    {(job.storage as Record<string,string> | undefined)?.original_name || job.source.filename} · {formatDuration(job.source.duration)} ·{' '}
                    {job.source.width}×{job.source.height}
                  </span>
                )}
                <div className="spacer" />
                <span className="faint">
                  {selectedCount}/{job.clips.length} selected
                </span>
                <button
                  className="small"
                  onClick={() =>
                    setSelected(
                      selectedCount === job.clips.length
                        ? new Set()
                        : new Set(job.clips.map((c) => c.id)),
                    )
                  }
                >
                  {selectedCount === job.clips.length ? 'Clear' : 'Select all'}
                </button>
                <a className="pill" href={api.exportZipUrl(job.id)} download style={{ background: 'rgba(212,175,55,0.14)', border: '1px solid rgba(212,175,55,0.45)', color: '#f5c542', padding: '5px 12px', borderRadius: '8px', textDecoration: 'none', fontWeight: 600, fontSize: '12.5px', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                  ↓ bulk export
                </a>
              </div>

              <div className="card-body">
                {options.platforms.map((platform) => {
                  const withPlatform = job.clips.filter((c) => clipPlatforms(c).includes(platform))
                  if (!withPlatform.length) return null
                  return (
                    <a
                      key={platform}
                      style={{ background: 'rgba(212,175,55,0.14)', border: '1px solid rgba(212,175,55,0.45)', color: '#f5c542', padding: '5px 12px', borderRadius: '8px', textDecoration: 'none', fontWeight: 600, fontSize: '12.5px', display: 'inline-flex', alignItems: 'center', gap: '6px', marginRight: 8, marginBottom: 12 }}
                      href={api.platformZipUrl(job.id, platform)}
                      onClick={async (event) => {
                        event.preventDefault()
                        try {
                          const blob = await api.platformZip(job.id, platform)
                          const url = URL.createObjectURL(blob)
                          const anchor = document.createElement('a')
                          anchor.href = url
                          anchor.download = `${job.source?.filename ?? 'clipforge'}_${platform}.zip`
                          anchor.click()
                          URL.revokeObjectURL(url)
                        } catch (error) {
                          onToast((error as Error).message, 'error')
                        }
                      }}
                    >
                      ↓ {platform} ({withPlatform.length})
                    </a>
                  )
                })}

                <div className="premium-clip-grid" style={{ width: '100%', overflow: 'visible', padding: '8px 0 24px 0', minHeight: 0, boxSizing: 'border-box' }}>
                  {job.clips.map((clip) => (
                    <ClipCard
                      key={clip.id}
                      job={job}
                      clip={clip}
                      selected={selected.has(clip.id)}
                      onToggleSelect={() =>
                        setSelected((prev) => {
                          const next = new Set(prev)
                          if (next.has(clip.id)) next.delete(clip.id)
                          else next.add(clip.id)
                          return next
                        })
                      }
                      onOpen={() => setEditing(clip)}
                      onToast={onToast}
                    />
                  ))}
                </div>
              </div>
            </section>
          )}
        </section>

        <aside className="cf-right-col" style={{ width: '380px', maxHeight: '100%', overflowY: 'auto', overflowX: 'hidden' }}>
          <OptionsPanel
            options={options}
            onChange={handleOptionsChange}
            onAudioChange={saveAudio}
            health={health}
          />
        </aside>
      </main>

      {editingClipResolved && job && (
        <ClipEditor
          job={job}
          clip={editingClipResolved}
          onClose={() => setEditing(null)}
          onJobUpdate={setJob}
          onToast={onToast}
          minDuration={15}
          maxDuration={90}
        />
      )}

      {/* Toasts: fixed viewport anchor — they never push the 2-col grid
          or the modal. Opacity-only entrance, no transform animation. */}
      <div aria-live="polite" style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 100, display: 'flex', flexDirection: 'column', gap: 12, pointerEvents: 'none' }}>
        {toasts.map((t) => (
          <div
            key={t.id}
            role={t.kind === 'error' ? 'alert' : 'status'}
            style={{
              padding: '12px 16px',
              borderRadius: 10,
              boxShadow: '0 8px 28px rgba(0,0,0,0.6)',
              color: '#f5f5f4',
              fontSize: 13,
              maxWidth: 340,
              pointerEvents: 'auto',
              border: '1px solid rgba(212,175,55,0.35)',
              background:
                t.kind === 'success' ? '#15803d' :
                t.kind === 'error' ? '#991b1b' : '#1d4ed8',
            }}
          >
            {t.message}
          </div>
        ))}
      </div>
    </div>
  )
}
