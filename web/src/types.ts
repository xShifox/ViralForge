export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export type StageName =
  | 'ingest' | 'transcribe' | 'analyze' | 'highlights' | 'length'
  | 'subtitles' | 'phrases' | 'render' | 'export' | 'done'

export interface ScoreBreakdown {
  text_hook: number
  emotion: number
  humor: number
  surprise: number
  energy: number
  laughter: number
  pause_rhythm: number
  structure: number
  topic: number
  llm: number | null
}

export interface SubtitleWord {
  start: number
  end: number
  text: string
}

export interface SubtitleCue {
  index: number
  start: number
  end: number
  text: string
  words: SubtitleWord[]
  speaker: string | null
  emphasis: boolean
}

export interface Clip {
  id: string
  index: number
  start: number
  end: number
  duration: number
  text: string
  title: string
  hook: string
  speaker: string | null
  score: number
  breakdown: ScoreBreakdown
  reasons: string[]
  cues: SubtitleCue[]
  layout: string
  state: 'pending' | 'rendering' | 'ready' | 'failed' | 'stale'
  poster: string
  outputs: Record<string, string>
  error: string | null
  llm_comment: string
  /** 0-100 while re-rendering, null otherwise. */
  progress?: number | null
  /** Which part of the re-render is running, e.g. "exporting 2/3". */
  progress_note?: string
}

export interface MediaInfo {
  filename: string
  duration: number
  width: number
  height: number
  fps: number
  video_codec: string
  audio_codec: string
  size_bytes: number
  has_audio: boolean
}

export interface Job {
  id: string
  status: JobStatus
  stage: StageName
  progress: number
  message: string
  created_at: number
  updated_at: number
  elapsed: number
  source: MediaInfo | null
  clips: Clip[]
  error: string | null
  warnings: string[]
  storage: Record<string, string>
}

export interface GpuMemory {
  /** The card's total video memory, MB. Zero means no card was found. */
  total_mb: number
  /** Used by everything on the machine. Exact, from the driver. */
  used_mb: number
  free_mb: number
  /** Memory the driver attributes to no process at all - on a laptop this is
   *  the display adapter's fixed reservation, and it is why used plus free is
   *  less than the total. Reported rather than absorbed into another band. */
  reserved_mb: number
  /** How busy the card is, 0-100. */
  utilization_pct: number
  /** This process's share of `used_mb`, measured as the rise above the level
   *  the card sat at while ViralForge held nothing. Null when there is no
   *  baseline to measure against - Windows does not report per-process video
   *  memory, so this is a measurement and not a reading, and the interface says
   *  so rather than showing a confident zero. */
  process_mb: number | null
  process_known: boolean
  baseline_mb: number | null
}

export interface Health {
  ok: boolean
  ffmpeg: boolean
  ffmpeg_version: string
  asr: {
    available: boolean
    device: string
    cuda_available: boolean
  }
  llm: { enabled: boolean; base_url: string; model: string; hint: string }
  /** What the graphics card can do, and how hard it is being driven.
   *  Changes speed only - the exported file is the same on every machine. */
  gpu: {
    name: string
    vendor: string
    vram_mb: number
    available: boolean
    compute_label: string
  }
  encoder: {
    setting: string
    active: string
    type: string
    available: boolean
    params: Record<string, string>
  }
  platforms: Record<string, {
    label: string
    width: number
    height: number
    fps: number
    max_seconds: number
    note: string
  }>
  limits: {
    min_duration: number
    max_duration: number
    hard_min_duration: number
    hard_max_duration: number
    slider_min: number
    slider_max: number
  }
  queue: { pending: number; active: number; completed: number }
  data_dir: string
}

export interface Settings {
  asr: {
    model: string
    device: string
    language: string
    beam_size: number
    vad_filter: boolean
    word_timestamps: boolean
  }
  detect: {
    max_clips: number
    min_duration: number
    max_duration: number
    target_duration: number
    score_threshold: number
    diversity: number
    use_llm: boolean
    track_talker?: boolean
    start_middle_hook?: boolean
    weights: Record<string, number>
  }
  render: {
    layout: 'crop' | 'fit' | 'split' | 'pip'
    subtitle_scale: number
    subtitle_uppercase: boolean
    word_highlight: boolean
    title_scale: number
    loudnorm: boolean
    preset: string
    encoder: 'auto' | 'gpu' | 'cpu' | 'nvenc' | 'software'
  }
  brand: {
    enabled: boolean
    watermark_text: string
    logo_path: string
    logo_anchor: string
    show_progress_bar: boolean
  }
  platforms: string[]
  max_upload_mb: number
  /** Last folder the clips were moved to. Offered as the starting point of the
   *  picker; the user is still asked on every job. */
  output_folder: string
}

export interface Calibration {
  records: number
  needed: number
  applied: boolean
  reason: string
  used: number
  /** Signal name to the multiple of its built-in weight the fit settled on. */
  shifts: Record<string, number>
  weights: Record<string, number>
  base_weights: Record<string, number>
  outcomes: {
    clip_id: string
    job_id: string
    outcome: number | null
    views: number | null
    score: number
    label: string
  }[]
}

export interface PipelineEvent {
  ts: number
  stage: string
  level: 'info' | 'warn' | 'error' | 'success'
  message: string
  progress: number | null
}
