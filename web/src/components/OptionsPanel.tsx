import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { api, chooseAudioFile, chooseFolder, desktopApi, revealFolder, type UploadOptions } from '../api'
import type { Health } from '../types'
import { GoldSwitch } from './GoldSwitch'

export interface AudioPatch {
  music_path?: string
  duck_music?: boolean
  duck_db?: number
  music_volume_db?: number
  sfx_folder?: string
  broll_folder?: string
}

interface Props {
  options: UploadOptions
  onChange: (options: UploadOptions) => void
  health: Health | null
  /** Persist audio settings to the backend so the next render uses them. */
  onAudioChange?: (patch: AudioPatch) => void
}

const LAYOUTS: { key: 'crop' | 'fit' | 'split' | 'pip'; label: string }[] = [
  { key: 'crop', label: 'Center crop' },
  { key: 'fit', label: 'Fit (no crop)' },
  { key: 'split', label: 'Split' },
  { key: 'pip', label: 'PiP' },
]

export function OptionsPanel({
  options,
  onChange,
  onAudioChange,
  health,
}: Props) {
  const [activeTab, setActiveTab] = useState<'quick' | 'visual' | 'audio' | 'system'>('quick')
  const logoInput = useRef<HTMLInputElement>(null)
  const [localRoot, setLocalRoot] = useState<string | null>(null)
  const [machine, setMachine] = useState<Health | null>(null)
  const [hwBusy, setHwBusy] = useState(false)
  const [hwLabel, setHwLabel] = useState<string | null>(null)
  const hwTimer = useRef<number | null>(null)

  useEffect(() => {
    api.storageStatus()
      .then((status) => setLocalRoot(status.local_root))
      .catch(() => setLocalRoot(null))
    return () => {
      if (hwTimer.current !== null) window.clearTimeout(hwTimer.current)
    }
  }, [])

  const machineToShow = machine ?? health
  const outputPath = options.outputFolder?.trim() ? options.outputFolder : (localRoot ?? '')

  // The upload builder sends exactly what the UI shows; default the encoder
  // and ASR device explicitly instead of leaving them undefined.
  const videoEncoder = String(options.videoEncoder ?? 'auto').toLowerCase()
  const speechDevice = String(options.speechDevice ?? 'auto').toLowerCase()

  const bridge = desktopApi()

  const sliderFillStyle = (val: number, min: number, max: number): CSSProperties => {
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100))
    return {
      background: `linear-gradient(90deg, #b48608 0%, #f5c542 50%, #fffbeb 100%) 0% 0% / ${pct}% 100% no-repeat, #1a1a24`,
      height: '8px',
      borderRadius: '999px',
      WebkitAppearance: 'none',
      appearance: 'none',
      width: '100%',
      cursor: 'pointer',
      outline: 'none',
      transition: 'background-size 0.1s ease-out',
      boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.7), 0 0 8px rgba(212,175,55,0.15)',
    }
  }

  const changeOutputFolder = async () => {
    if (bridge) {
      // Native picker. chooseFolder returns null on cancel — that is fine.
      const picked = await chooseFolder(outputPath)
      if (picked) onChange({ ...options, outputFolder: picked })
      return
    }
    // No native bridge (browser): fall back to a plain prompt.
    const entered = window.prompt('Output folder', outputPath)
    if (entered !== null && entered.trim()) {
      onChange({ ...options, outputFolder: entered.trim() })
    }
  }

  const openOutputFolder = async () => {
    if (!outputPath) return
    await revealFolder(outputPath)
  }

  const refreshHardware = async () => {
    setHwBusy(true)
    try {
      const bench = await api.benchmark()
      const gpu = bench.gpu?.name ?? ''
      const gpuShort = (gpu.match(/RTX\s*\d+|GTX\s*\d+|Arc\s+\w+/i)?.[0] ?? 'GPU').replace(/\s+/g, ' ').trim()
      const encoder = String(bench.encoder?.active ?? '').toUpperCase()
      const encLabel = encoder.includes('NVENC') ? 'NVENC'
        : encoder.includes('AMF') ? 'AMF'
        : encoder.includes('QSV') ? 'QSV'
        : encoder.includes('MF') ? 'MF'
        : 'CPU'
      setMachine(await api.health())
      setHwLabel(`✓ Hardware Verified (${gpuShort} · ${encLabel})`)
    } catch {
      setHwLabel('✗ Hardware check failed')
    } finally {
      setHwBusy(false)
      if (hwTimer.current !== null) window.clearTimeout(hwTimer.current)
      hwTimer.current = window.setTimeout(() => setHwLabel(null), 2000)
    }
  }

  const browseMusicBed = async () => {
    const picked = await chooseAudioFile(options.musicBed)
    if (picked) {
      onChange({ ...options, musicBed: picked })
      onAudioChange?.({ music_path: picked })
    }
  }

  const browseSfx = async () => {
    const picked = await chooseFolder(options.sfxFolder || '')
    if (picked) {
      onChange({ ...options, sfxFolder: picked })
      onAudioChange?.({ sfx_folder: picked })
    }
  }

  const browseBroll = async () => {
    const picked = await chooseFolder(options.brollFolder || '')
    if (picked) {
      onChange({ ...options, brollFolder: picked })
      onAudioChange?.({ broll_folder: picked })
    }
  }

  const limits = health?.limits ?? {
    min_duration: 15,
    max_duration: 90,
    hard_min_duration: 15,
    hard_max_duration: 90,
    slider_min: 15,
    slider_max: 60,
  }



  const exportFormats = [
    { key: 'tiktok', label: 'TikTok' },
    { key: 'shorts', label: 'YouTube Shorts' },
    { key: 'reels', label: 'Instagram Reels' },
    { key: 'square', label: 'Square 1:1' },
    { key: 'landscape', label: 'Landscape 16:9' },
  ]

  return (
    <div className="cf-options-panel">
      {/* Tab Bar */}
      <div className="cf-tab-bar">
        <button
          className={`cf-tab-btn ${activeTab === 'quick' ? 'active' : ''}`}
          onClick={() => setActiveTab('quick')}
        >
          ⚡ Quick Setup
        </button>
        <button
          className={`cf-tab-btn ${activeTab === 'visual' ? 'active' : ''}`}
          onClick={() => setActiveTab('visual')}
        >
          🎬 Framing & Style
        </button>
        <button
          className={`cf-tab-btn ${activeTab === 'audio' ? 'active' : ''}`}
          onClick={() => setActiveTab('audio')}
        >
          🎵 Audio & B-Roll
        </button>
        <button
          className={`cf-tab-btn ${activeTab === 'system' ? 'active' : ''}`}
          onClick={() => setActiveTab('system')}
        >
          ⚙️ Storage & GPU
        </button>
      </div>

      {/* Quick Setup Tab */}
      {activeTab === 'quick' && (
        <div className="cf-tab-content cf-tab-panel-anim">
          <GoldSwitch
            checked={options.startMiddleHook}
            onChange={(checked) => onChange({ ...options, startMiddleHook: checked })}
            label="Start Middle Hook"
            hint="Prepend best middle moment (25%–75%) as opener"
          />
          
          <GoldSwitch
            checked={options.trackTalker}
            onChange={(checked) => onChange({ ...options, trackTalker: checked })}
            label="Track talker"
            hint="AI face & speaker tracking camera"
          />

          <div className="cf-row">
            <div className="cf-form-group">
              <label className="cf-label">Clips to find</label>
              <input
                type="number"
                min="1"
                max="30"
                value={options.maxClips ?? 6}
                onChange={(e) => onChange({ ...options, maxClips: parseInt(e.target.value) })}
                className="cf-input"
              />
            </div>
          </div>

          <div className="cf-form-group">
            <label className="cf-label">Minimum clip length</label>
            <input
              type="range"
              min={limits.slider_min}
              max={limits.slider_max}
              value={options.minDuration ?? limits.slider_min}
              onChange={(e) => {
                const min = parseInt(e.target.value)
                const max = Math.max(options.maxDuration ?? limits.slider_max, min)
                onChange({ ...options, minDuration: min, maxDuration: max })
              }}
              className="cf-range"
              style={sliderFillStyle(options.minDuration ?? limits.slider_min, limits.slider_min, limits.slider_max)}
            />
            <span className="cf-range-value">{options.minDuration ?? limits.slider_min}s</span>
          </div>

          <div className="cf-form-group">
            <label className="cf-label">Maximum clip length</label>
            <input
              type="range"
              min={15}
              max={90}
              step={5}
              value={options.maxDuration ?? 90}
              onChange={(e) => {
                const max = parseInt(e.target.value)
                const min = Math.min(options.minDuration ?? limits.slider_min, max)
                onChange({ ...options, maxDuration: max, minDuration: min })
              }}
              className="cf-range"
              style={sliderFillStyle(options.maxDuration ?? 90, 15, 90)}
            />
            <span className="cf-range-value">{options.maxDuration ?? 90}s</span>
          </div>

          <div className="cf-form-group">
            <label className="cf-label">Export formats</label>
            <div className="cf-pills">
              {exportFormats.map((format) => {
                const currentPlatforms: string[] = Array.isArray(options.platforms)
                  ? options.platforms
                  : Array.isArray(options.exportFormats)
                  ? options.exportFormats
                  : ['tiktok']
                const isActive = currentPlatforms.includes(format.key)
                return (
                  <button
                    key={format.key}
                    className={`cf-pill ${isActive ? 'active' : ''}`}
                    onClick={() => {
                      if (isActive && currentPlatforms.length <= 1) return
                      const next = isActive
                        ? currentPlatforms.filter((f) => f !== format.key)
                        : [...currentPlatforms, format.key]
                      onChange({ ...options, platforms: next, exportFormats: next })
                    }}
                  >
                    {format.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* Visual Tab */}
      {activeTab === 'visual' && (
        <div className="cf-tab-content cf-tab-panel-anim">
          <div className="cf-form-group">
            <label className="cf-label">Layout</label>
            <div className="cf-radio-group">
              {LAYOUTS.map((layout) => (
                <button
                  key={layout.key}
                  className={`cf-radio ${options.layout === layout.key ? 'active' : ''}`}
                  onClick={() => {
                    if (layout.key === 'crop' && options.cropMode === 'none') {
                      onChange({ ...options, layout: layout.key, cropMode: 'auto' })
                    } else {
                      onChange({ ...options, layout: layout.key })
                    }
                  }}
                >
                  {layout.label}
                </button>
              ))}
            </div>
            <label className="cf-label" style={{ marginTop: 8 }}>Crop mode</label>
            <div className="cf-radio-group" style={{ marginTop: 8 }}>
              {(
                [
                  { key: 'auto', label: 'Auto crop' },
                  { key: 'manual', label: 'Manual crop' },
                  { key: 'none', label: 'No crop (save original)' },
                ] as const
              ).map((mode) => (
                <button
                  key={mode.key}
                  className={`cf-radio ${options.cropMode === mode.key ? 'active' : ''}`}
                  onClick={() => {
                    if (mode.key === 'none') {
                      onChange({ ...options, cropMode: 'none', layout: 'fit' })
                    } else if (mode.key === 'manual') {
                      onChange({ ...options, cropMode: 'manual', layout: 'crop' })
                    } else {
                      onChange({ ...options, cropMode: 'auto' })
                    }
                  }}
                >
                  {mode.label}
                </button>
              ))}
            </div>
            {options.cropMode === 'none' && (
              <span className="cf-range-value">No crop — full frame shown, nothing cut off</span>
            )}
            {options.cropMode === 'manual' && (
              <span className="cf-range-value">Manual uses Center crop above — move focal/zoom to see difference. At 50/50/100% it matches Auto.</span>
            )}
            {options.layout === 'crop' && options.cropMode === 'manual' && (
              <>
                <div className="cf-form-group">
                  <label className="cf-label">Focal point X (left - right)</label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={options.cropX ?? 50}
                    onChange={(e) => onChange({ ...options, cropX: parseInt(e.target.value) })}
                    className="cf-range"
                    style={sliderFillStyle(options.cropX ?? 50, 0, 100)}
                  />
                  <span className="cf-range-value">{options.cropX ?? 50}%</span>
                </div>
                <div className="cf-form-group">
                  <label className="cf-label">Focal point Y (top - bottom)</label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={options.cropY ?? 50}
                    onChange={(e) => onChange({ ...options, cropY: parseInt(e.target.value) })}
                    className="cf-range"
                    style={sliderFillStyle(options.cropY ?? 50, 0, 100)}
                  />
                  <span className="cf-range-value">{options.cropY ?? 50}%</span>
                </div>
                <div className="cf-form-group">
                  <label className="cf-label">Zoom</label>
                  <input
                    type="range"
                    min="100"
                    max="200"
                    value={options.cropZoom ?? 100}
                    onChange={(e) => onChange({ ...options, cropZoom: parseInt(e.target.value) })}
                    className="cf-range"
                    style={sliderFillStyle(options.cropZoom ?? 100, 100, 200)}
                  />
                  <span className="cf-range-value">{options.cropZoom ?? 100}%</span>
                </div>
              </>
            )}
          </div>

          {/* Caption Style Section */}
          <div className="cf-form-group">
            <label className="cf-label">Caption Style</label>
            <GoldSwitch
              checked={options.uppercaseCaptions || false}
              onChange={(checked) => onChange({ ...options, uppercaseCaptions: checked })}
              label="Uppercase captions"
            />
            <GoldSwitch
              checked={options.highlightSpokenWord || false}
              onChange={(checked) => onChange({ ...options, highlightSpokenWord: checked })}
              label="Highlight the spoken word"
            />
            <GoldSwitch
              checked={options.fadeCaptions || false}
              onChange={(checked) => onChange({ ...options, fadeCaptions: checked })}
              label="Fade captions in as they are spoken"
            />
            <div className="cf-form-group">
              <label className="cf-label">Caption size</label>
              <input
                type="range"
                min="50"
                max="150"
                value={options.captionSize ?? 100}
                onChange={(e) => onChange({ ...options, captionSize: parseInt(e.target.value) })}
                className="cf-range"
                style={sliderFillStyle(options.captionSize ?? 100, 50, 150)}
              />
              <span className="cf-range-value">{options.captionSize ?? 100}%</span>
            </div>
            <div className="cf-form-group">
              <label className="cf-label">Headline size</label>
              <input
                type="range"
                min="50"
                max="150"
                value={options.headlineSize ?? 100}
                onChange={(e) => onChange({ ...options, headlineSize: parseInt(e.target.value) })}
                className="cf-range"
                style={sliderFillStyle(options.headlineSize ?? 100, 50, 150)}
              />
              <span className="cf-range-value">{options.headlineSize ?? 100}%</span>
            </div>
          </div>

          {/* Branding Section */}
          <div className="cf-form-group">
            <label className="cf-label">Branding</label>
            <div className="cf-switch-card">
              <div className="cf-switch-label">
                <span className="cf-switch-title">Watermark / handle</span>
              </div>
              <input
                type="text"
                placeholder="@yourchannel"
                value={options.watermarkText}
                onChange={(e) => onChange({ ...options, watermarkText: e.target.value })}
                className="cf-input"
              />
            </div>
            <div className="cf-switch-card">
              <div className="cf-switch-label">
                <span className="cf-switch-title">Brand Logo (PNG)</span>
              </div>
              <input
                ref={logoInput}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                hidden
                onChange={async (e) => {
                  const file = e.target.files?.[0]
                  e.target.value = ''
                  if (!file) return
                  try {
                    const result = await api.uploadLogo(file)
                    onChange({ ...options, logoPath: result.path })
                  } catch {
                    // No toast channel here; the logo simply doesn't change.
                  }
                }}
              />
              <button
                type="button"
                className="cf-button"
                onClick={() => logoInput.current?.click()}
              >
                Upload logo
              </button>
            </div>
            {options.logoPath && (
              <span className="cf-range-value" title={options.logoPath}>
                {options.logoPath}
              </span>
            )}
            <GoldSwitch
              checked={options.showProgressBar}
              onChange={(checked) => onChange({ ...options, showProgressBar: checked })}
              label="Show progress bar"
            />
          </div>
        </div>
      )}

      {/* Audio Tab */}
      {activeTab === 'audio' && (
        <div className="cf-tab-content cf-tab-panel-anim">
          <div className="cf-form-group">
            <label className="cf-label">Music bed</label>
            <div className="cf-switch-card">
              <input
                type="text"
                placeholder="No music"
                value={options.musicBed || ''}
                onChange={(e) => {
                  onChange({ ...options, musicBed: e.target.value })
                  onAudioChange?.({ music_path: e.target.value })
                }}
                className="cf-input"
              />
              <button className="cf-button" onClick={browseMusicBed}>
                Browse
              </button>
            </div>
            {options.musicBed && (
              <span className="cf-range-value">Played under every clip, ducked under speech</span>
            )}
            {!options.musicBed && (
              <span className="cf-range-value">No music set — ducking has no effect until a bed is chosen</span>
            )}
          </div>

          <GoldSwitch
            checked={options.audioDucking}
            onChange={(checked) => {
              onChange({ ...options, audioDucking: checked })
              onAudioChange?.({ duck_music: checked })
            }}
            label="Duck music while speaking"
            hint="Auto-lower music volume during speech — needs a music bed above (no bed = no effect)"
          />

          <div className="cf-form-group">
            <label className="cf-label">Ducking depth</label>
            <input
              type="range"
              min="-24"
              max="0"
              value={options.duckingDepth ?? -12}
              onChange={(e) => {
                const v = parseInt(e.target.value)
                onChange({ ...options, duckingDepth: v })
                onAudioChange?.({ duck_db: v })
              }}
              className="cf-range"
              style={sliderFillStyle(options.duckingDepth ?? -12, -24, 0)}
            />
            <span className="cf-range-value">{options.duckingDepth ?? -12} dB</span>
          </div>

          <div className="cf-form-group">
            <div className="cf-switch-card">
              <span className="cf-switch-title">Sound effects</span>
              <button className="cf-button" onClick={browseSfx}>
                Browse SFX Folder
              </button>
            </div>
            {options.sfxFolder && (
              <span className="cf-range-value" title={options.sfxFolder}>
                {options.sfxFolder}
              </span>
            )}
          </div>

          <div className="cf-form-group">
            <div className="cf-switch-card">
              <span className="cf-switch-title">Related visuals (B-Roll)</span>
              <button className="cf-button" onClick={browseBroll}>
                Browse B-Roll Folder
              </button>
            </div>
            {options.brollFolder && (
              <span className="cf-range-value" title={options.brollFolder}>
                {options.brollFolder}
              </span>
            )}
          </div>
        </div>
      )}

      {/* System Tab */}
      {activeTab === 'system' && (
        <div className="cf-tab-content cf-tab-panel-anim">
          <GoldSwitch
            checked={options.storage?.includes('local') || false}
            onChange={(checked) => {
              const current = options.storage || []
              const updated = checked
                ? [...current, 'local']
                : current.filter((s) => s !== 'local')
              onChange({ ...options, storage: updated })
            }}
            label="Local folder"
          />
          
          <GoldSwitch
            checked={options.storage?.includes('onedrive') || false}
            onChange={(checked) => {
              const current = options.storage || []
              const updated = checked
                ? [...current, 'onedrive']
                : current.filter((s) => s !== 'onedrive')
              onChange({ ...options, storage: updated })
            }}
            label="OneDrive"
          />
          
          <GoldSwitch
            checked={options.storage?.includes('gdrive') || false}
            onChange={(checked) => {
              const current = options.storage || []
              const updated = checked
                ? [...current, 'gdrive']
                : current.filter((s) => s !== 'gdrive')
              onChange({ ...options, storage: updated })
            }}
            label="Google Drive"
          />

          {/* Output folder — where finished clips are written */}
          <div className="cf-form-group">
            <label className="cf-label">Output folder</label>
            <div className="cf-output-row">
              <span className="cf-output-path" title={outputPath}>
                {outputPath || 'Default export folder'}
              </span>
              <div className="cf-row">
                <button className="cf-gold-btn sm" onClick={changeOutputFolder}>
                  Change Folder
                </button>
                <button className="cf-gold-btn sm" onClick={openOutputFolder}>
                  Open Folder
                </button>
              </div>
            </div>
            {options.outputFolder && (
              <span className="cf-range-value">
                Used for the clips from the next export — passed as output_folder
              </span>
            )}
          </div>

          <div className="cf-form-group">
            <label className="cf-label">Video encoder</label>
            <div className="cf-radio-group">
              {(['auto', 'gpu', 'cpu'] as const).map((enc) => (
                <button
                  key={enc}
                  className={`cf-radio ${videoEncoder === enc ? 'active' : ''}`}
                  onClick={() => onChange({ ...options, videoEncoder: enc })}
                >
                  {enc === 'auto' ? 'Auto' : enc === 'gpu' ? 'GPU' : 'CPU'}
                </button>
              ))}
            </div>
          </div>

          <div className="cf-form-group">
            <label className="cf-label">Speech recognition</label>
            <div className="cf-radio-group">
              {(['auto', 'gpu', 'cpu'] as const).map((device) => (
                <button
                  key={device}
                  className={`cf-radio ${speechDevice === device ? 'active' : ''}`}
                  onClick={() => onChange({ ...options, speechDevice: device })}
                >
                  {device === 'auto' ? 'Auto' : device === 'gpu' ? 'GPU' : 'CPU'}
                </button>
              ))}
            </div>
          </div>

          {/* This machine hardware info */}
          {machineToShow && (
            <section className="card">
              <div className="card-head">
                <h2>This machine</h2>
              </div>
              <div className="card-body">
                <dl className="kv">
                  <dt>GPU</dt>
                  <dd>{machineToShow.gpu?.name || 'CPU Mode'}</dd>
                  <dt>Compute</dt>
                  <dd>
                    {machineToShow.gpu?.compute_label || 'CPU Mode'}
                  </dd>
                  <dt>Encoder</dt>
                  <dd>{machineToShow.encoder?.active || 'libx264'}</dd>
                </dl>
                <div className="cf-row" style={{ marginTop: 16 }}>
                  <button
                    className="cf-gold-btn"
                    style={{ margin: 0 }}
                    disabled={hwBusy}
                    onClick={refreshHardware}
                  >
                    {hwLabel ?? (hwBusy ? 'Refreshing…' : 'Refresh Hardware Benchmark')}
                  </button>
                </div>
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  )
}