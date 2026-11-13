import type {
  Calibration,
  Clip,
  GpuMemory,
  Health,
  Job,
  Settings,
  SubtitleCue,
} from './types'

const BASE = '/api'

class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* keep the status text */
    }
    throw new ApiError(detail, response.status)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export interface UploadOptions {
  maxClips: number
  minDuration: number
  maxDuration: number
  platforms: string[]
  exportFormats?: string[]
  useLlm: boolean
  layout: 'crop' | 'fit' | 'split' | 'pip'
  watermarkText: string
  logoPath: string
  showProgressBar: boolean
  storage: string[]
  trackTalker: boolean
  startMiddleHook: boolean
  audioDucking: boolean
  /** Where the finished clips are moved to. Asked for on every job. */
  outputFolder?: string
  // Caption style options
  uppercaseCaptions?: boolean
  highlightSpokenWord?: boolean
  fadeCaptions?: boolean
  captionSize?: number
  headlineSize?: number
  // Audio options
  musicBed?: string
  duckingDepth?: number
  // Media engine folders (sound effects / related visuals)
  sfxFolder?: string
  brollFolder?: string
  // Video encoder options
  videoEncoder?: 'auto' | 'gpu' | 'cpu'
  // Speech recognition options
  speechDevice?: 'auto' | 'gpu' | 'cpu'
  // Manual crop: focal point (0-100) + zoom (100-200). 'none' = no crop, full frame.
  cropMode?: 'auto' | 'manual' | 'none'
  cropX?: number
  cropY?: number
  cropZoom?: number
}

/** The native folder picker, when the app is running in its own window.
 *
 * A web page cannot do this: a directory input reports a faked `C:\fakepath`
 * and browsers withhold the real location on purpose. pywebview exposes a small
 * Python API to the page, and that is the only way to get a real path back.
 * Returns null when there is no bridge, which is the browser case. */
interface DesktopBridge {
  pick_folder(start?: string): Promise<string>
  reveal_folder(path: string): Promise<boolean>
}

export async function chooseAudioFile(start?: string): Promise<string | null> {
  const bridge = (window as unknown as {
    pywebview?: { api?: { pick_file?: (start?: string, pattern?: string) => Promise<string> } }
  }).pywebview?.api
  if (!bridge || typeof bridge.pick_file !== 'function') return null
  try {
    return (await bridge.pick_file(start ?? '', '*.mp3;*.wav;*.m4a;*.aac;*.flac')) || null
  } catch {
    return null
  }
}

export function desktopApi(): DesktopBridge | null {
  const bridge = (window as unknown as {
    pywebview?: { api?: Partial<DesktopBridge> }
  }).pywebview?.api
  if (bridge && typeof bridge.pick_folder === 'function') {
    return bridge as DesktopBridge
  }
  return null
}

export async function chooseFolder(start: string): Promise<string | null> {
  const bridge = desktopApi()
  if (!bridge) return null
  try {
    return (await bridge.pick_folder(start)) || null
  } catch {
    return null
  }
}

export async function revealFolder(path: string): Promise<boolean> {
  const bridge = desktopApi()
  if (!bridge) return false
  try {
    return await bridge.reveal_folder(path)
  } catch {
    return false
  }
}

/** One folder in the chooser's listing. */
export interface FolderEntry {
  name: string
  path: string
  /** A drive letter, a known place like Videos, or an ordinary folder. */
  kind: 'drive' | 'place' | 'folder'
  hint: string
}

export interface FolderListing {
  /** Empty at the top level, where the drives and the known places are. */
  path: string
  /** Where "up" goes, or null at the root of a drive. */
  parent: string | null
  folders: FolderEntry[]
}

/** Ask the server what folders are inside a directory.
 *
 *  This is what makes browsing possible in a plain browser. The native dialog
 *  only exists in the desktop window, and a directory input will not give up
 *  the real path, so the page lists folders itself. */
export async function listFolders(path: string): Promise<FolderListing> {
  return request<FolderListing>(
    `/system/folders?path=${encodeURIComponent(path)}`)
}

/** A current reading of the card's memory. Polled on its own schedule. */
export async function readGpuMemory(): Promise<GpuMemory> {
  return request<GpuMemory>('/system/gpu')
}

export const api = {
  health: () => request<Health>('/system/health'),

  /** POST /system/benchmark - re-probes the GPU/encoder chain for real and
   * reports what it found. The "Refresh Hardware Benchmark" button uses this
   * (not health, which is served from the same cache). */
  benchmark: () =>
    request<{
      ok: boolean
      elapsed_ms: number
      gpu: { name: string; vendor: string; vram_mb: number; available: boolean; compute_label: string }
      encoder: { active: string; type: string; available: boolean; params: Record<string, unknown> }
      cuda_available: boolean
    }>('/system/benchmark', { method: 'POST' }),

  settings: () => request<Settings>('/system/settings'),

  saveSettings: (patch: Record<string, unknown>) =>
    request<Settings>('/system/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }),

  listJobs: () => request<{ jobs: Job[] }>('/jobs'),
  calibration: () => request<Calibration>('/calibration'),
  recordResult: (jobId: string, clipId: string, body: { outcome: number | null; views: number | null }) =>
    request<{ recorded: boolean; outcome: number | null; views: number | null; status: Calibration }>(
      `/jobs/${jobId}/clips/${clipId}/result`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    ),
  clearCalibration: () => request<{ removed: number; message: string }>('/calibration', { method: 'DELETE' }),

  getJob: (id: string) => request<Job>(`/jobs/${id}`),

  cancelJob: (id: string) => request<{ ok: boolean }>(`/jobs/${id}/cancel`, { method: 'POST' }),
  /** Re-run only the export stage, reusing the renders already on disk. */
  resumeJob: (id: string) => request<Job>(`/jobs/${id}/resume`, { method: 'POST' }),

  deleteJob: (id: string) => request<{ ok: boolean }>(`/jobs/${id}`, { method: 'DELETE' }),

  upload: (file: File, options: UploadOptions, onProgress?: (percent: number) => void) =>
    new Promise<Job>((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      form.append('max_clips', String(options.maxClips))
      form.append('min_duration', String(options.minDuration))
      form.append('max_duration', String(options.maxDuration))
      form.append('platforms', options.platforms.join(','))
      form.append('use_llm', String(options.useLlm))
      form.append('layout', options.layout)
      form.append('watermark_text', options.watermarkText)
      form.append('logo_path', options.logoPath)
      form.append('show_progress_bar', String(options.showProgressBar))
      form.append('storage', options.storage.join(','))
      if (options.trackTalker) form.append('track_talker', 'true')
      if (options.startMiddleHook) form.append('start_middle_hook', 'true')
      if (options.outputFolder) form.append('output_folder', options.outputFolder)
      if (options.exportFormats !== undefined)
        form.append('export_formats', Array.isArray(options.exportFormats) ? options.exportFormats.join(',') : options.exportFormats)
      if (options.videoEncoder !== undefined) form.append('video_encoder', options.videoEncoder)
      if (options.speechDevice !== undefined) form.append('speech_device', options.speechDevice)
      if (options.cropMode !== undefined) form.append('crop_mode', options.cropMode)
      if (options.cropX !== undefined) form.append('crop_x', String(options.cropX))
      if (options.cropY !== undefined) form.append('crop_y', String(options.cropY))
      if (options.cropZoom !== undefined) form.append('crop_zoom', String(options.cropZoom))

      // Backend also accepts fade_captions / sfx_folder / broll_folder per job
      // (create_job Form fields) — they must ride along with the upload, not
      // only via the best-effort settings PUT, or the first job after a toggle
      // silently runs with the persisted (stale) values.
      if (options.fadeCaptions !== undefined) form.append('fade_captions', String(options.fadeCaptions))
      if (options.sfxFolder) form.append('sfx_folder', options.sfxFolder)
      if (options.brollFolder) form.append('broll_folder', options.brollFolder)
      // XHR rather than fetch: upload progress is not observable with fetch
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/jobs`)
      let lastPercent = -1
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          const percent = Math.round((event.loaded / event.total) * 100)
          // 1% granularity: this callback hands back to React state, so
          // forwarding every progress tick (dozens/sec on fast local I/O)
          // re-renders the whole tree for a bar that moves once a percent.
          if (percent !== lastPercent) {
            lastPercent = percent
            onProgress(percent)
          }
        }
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as Job)
          } catch {
            reject(new ApiError('server returned an unreadable response', xhr.status))
          }
        } else {
          let detail = `upload failed (${xhr.status})`
          try {
            const body = JSON.parse(xhr.responseText)
            if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
          } catch {
            /* ignore */
          }
          reject(new ApiError(detail, xhr.status))
        }
      }
      xhr.onerror = () => reject(new ApiError('network error during upload', 0))
      xhr.send(form)
    }),

  patchClip: (
    jobId: string,
    clipId: string,
    patch: {
      title?: string
      start?: number
      end?: number
      captionText?: Record<number, string>
      rerender?: boolean
    },
  ) =>
    request<Job>(`/jobs/${jobId}/clips/${clipId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: patch.title,
        start: patch.start,
        end: patch.end,
        caption_text: patch.captionText,
        rerender: patch.rerender ?? false,
      }),
    }),

  exportZipUrl: (jobId: string) => `${BASE}/jobs/${jobId}/export.zip`,
  platformZipUrl: (jobId: string, platform: string) =>
    `${BASE}/jobs/${jobId}/export.zip?platform=${encodeURIComponent(platform)}`,
  downloadUrl: (jobId: string, clipId: string, platform: string) =>
    `${BASE}/jobs/${jobId}/clips/${clipId}/download?platform=${encodeURIComponent(platform)}`,
  posterUrl: (jobId: string, clipId: string) => `${BASE}/jobs/${jobId}/poster/${clipId}`,
  // The version is part of the URL on purpose. The preview is served with a
  // five minute cache, so a re-render that leaves the URL unchanged is invisible:
  // the browser replays the old file and the editor keeps showing the clip as it
  // was before the edit.
  previewUrl: (jobId: string, clipId: string, version?: number) =>
    `${BASE}/jobs/${jobId}/clips/${clipId}/preview${version ? `?v=${version}` : ''}`,
  subtitlesUrl: (jobId: string, clipId: string, fmt: 'srt' | 'vtt') =>
    `${BASE}/jobs/${jobId}/subtitles/${clipId}?fmt=${fmt}`,
  sourcePreviewUrl: (jobId: string) => `${BASE}/jobs/${jobId}/source-preview`,
  logoUrl: (name: string) => `${BASE}/assets/logo/${encodeURIComponent(name)}`,

  uploadLogo: async (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<{ path: string; url: string; name: string }>('/assets/logo', {
      method: 'POST',
      body: form,
    })
  },

  storageStatus: () =>
    request<{ backends: { key: string; label: string; available: boolean }[]; local_root: string }>(
      '/storage',
    ),

  storageTest: (backend: string) =>
    request<{ ok: boolean; message?: string }>(`/storage/test/${backend}`, {
      method: 'POST',
    }),

  // Blob download: request() hard-codes response.json(), which would throw
  // on a binary zip. Use fetch directly and surface JSON error bodies the
  // same way request() does.
  platformZip: async (jobId: string, platform: string): Promise<Blob> => {
    const response = await fetch(`${BASE}/jobs/${jobId}/export.zip`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ platform }),
    })
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`
      try {
        const body = await response.json()
        if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch {
        /* keep the status text */
      }
      throw new ApiError(detail, response.status)
    }
    return response.blob()
  },
}

/**
 * Subscribe to a job's event stream. Returns an unsubscribe function.
 *
 * `onStatus` reports whether the stream is currently healthy. An EventSource
 * does not stay closed on a network failure, it drops back to CONNECTING and
 * retries on its own, so without this the UI cannot tell "still working" from
 * "the server is gone" and sits on the last progress it saw for ever.
 */
export function subscribeToJob(
  jobId: string,
  after: number,
  onEvent: (event: import('./types').PipelineEvent) => void,
  onDone: () => void,
  onStatus?: (connected: boolean) => void,
): () => void {
  const source = new EventSource(`${BASE}/jobs/${jobId}/events?after=${after}`)
  let finished = false
  const finish = () => {
    if (finished) return
    finished = true
    source.close()
    onDone()
  }
  source.onopen = () => onStatus?.(true)
  source.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data) as import('./types').PipelineEvent
      onEvent(event)
      if (event.stage === 'done' && (event.message === 'completed' || event.message === 'failed' || event.message === 'cancelled')) {
        finish()
      }
    } catch {
      /* ignore malformed frames */
    }
  }
  source.onerror = () => {
    // A finished job arrives as a `done` event, not as a closed socket, so an
    // error here means the stream is broken rather than finished. Report it and
    // let the browser keep retrying.
    if (source.readyState === EventSource.CLOSED) {
      finish()
      return
    }
    onStatus?.(false)
  }
  return () => {
    finished = true
    source.close()
  }
}

export function describeCueTime(cue: SubtitleCue): string {
  return `${cue.start.toFixed(1)}s – ${cue.end.toFixed(1)}s`
}

export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`
}

export function formatDuration(secondsOrMs: number): string {
  const totalSecs = secondsOrMs > 100000 ? secondsOrMs / 1000 : secondsOrMs
  const total = Math.max(0, Math.round(totalSecs))
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  return `${minutes}:${String(rest).padStart(2, '0')}`
}

export function clipPlatforms(clip: Clip): string[] {
  return Object.keys(clip.outputs).filter(
    (key) => !key.startsWith('url:') && key !== 'source' && key !== 'poster',
  )
}
