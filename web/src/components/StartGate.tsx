import type { UploadOptions } from '../api'

interface Props {
  file: File
  folder: string
  options: UploadOptions
  platformLabels: Record<string, string>
  onChangeFolder: () => void
  onStart: () => void
  onDiscard: () => void
}

export function StartGate({ file, folder, options, platformLabels, onChangeFolder, onStart, onDiscard }: Props) {
  const megabytes = file.size / 1e6
  const sizeLabel = megabytes < 1 ? '<1 MB' : `${Math.round(megabytes)} MB`
  const ext = (file.name.split('.').pop() || 'MP4').toUpperCase()
  const chosen = options.platforms ?? []
  const minutes = Math.max(1, Math.round(options.minDuration ?? 15))
  const maxMinutes = Math.max(minutes, Math.round(options.maxDuration ?? 90))
  const clipCount = options.maxClips ?? 6
  const effectiveFolder = folder.trim() || options.outputFolder?.trim() || 'Default export folder'

  return (
    <div className="cf-staging-card">
      <div className="cf-staging-header">
        <span className="cf-status-pill">● READY TO FORGE</span>
        <button type="button" className="cf-secondary-btn" onClick={onDiscard}>
          🔄 Choose Another Video
        </button>
      </div>

      <div className="cf-staging-banner">
        <div className="cf-staging-file">
          <span className="cf-staging-icon" aria-hidden="true">🎬</span>
          <span className="cf-staging-name" title={file.name}>{file.name}</span>
        </div>
        <div className="cf-staging-badges">
          <span className="cf-badge">{sizeLabel}</span>
          <span className="cf-badge">{ext}</span>
        </div>
      </div>

      <div className="cf-staging-grid">
        <div className="cf-staging-tile">
          <span className="cf-tile-label">SAVING CLIPS TO</span>
          <div className="cf-tile-row">
            <span className="cf-tile-value mono">{effectiveFolder}</span>
            <button type="button" className="cf-gold-link" onClick={onChangeFolder}>Change Folder</button>
          </div>
        </div>
        <div className="cf-staging-tile">
          <span className="cf-tile-label">EXPORT FORMATS</span>
          <div className="cf-tile-pills">
            {chosen.length
              ? chosen.map((p) => <span key={p} className="cf-gold-pill">{platformLabels[p] ?? p}</span>)
              : <span className="cf-tile-value">none chosen</span>}
          </div>
        </div>
        <div className="cf-staging-tile">
          <span className="cf-tile-label">CLIP SPECS</span>
          <span className="cf-tile-value">{clipCount} Clips · {minutes}s – {maxMinutes}s</span>
        </div>
      </div>

      <div className="cf-staging-actions">
        <button
          type="button"
          className="cf-gold-btn cf-staging-start"
          onClick={onStart}
          disabled={chosen.length === 0}
        >
          ⚡ Start Clipping Now
        </button>
        <button type="button" className="cf-cancel-btn" onClick={onDiscard}>
          Cancel
        </button>
      </div>
      {chosen.length === 0 && (
        <p className="cf-staging-warn">Pick at least one platform on the right first.</p>
      )}
    </div>
  )
}