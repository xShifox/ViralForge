import { useEffect, useRef, useState } from 'react'
import type { UploadOptions } from '../api'

const ACCEPT = '.mp4,.mkv,.mov,.m4v,.webm,.avi,.wmv,.flv,.mpg,.mpeg,.ts'
const MAX_BYTES = 20 * 1024 * 1024 * 1024

interface Props {
  /** True only while a file is actually being sent. */
  uploading: boolean
  /** True when a job is already running, so a second one would queue behind it. */
  busy: boolean
  uploadPercent: number | null
  onFile: (file: File) => void
  options: UploadOptions
  onToast?: (message: string, kind?: 'info' | 'success' | 'error') => void
}

export function Dropzone({ uploading, busy, uploadPercent, onFile, options, onToast }: Props) {
  const [over, setOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)

  // paste support: copy a file in the browser and drop it straight in
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      if (uploading || busy) return
      const file = Array.from(event.clipboardData?.files ?? [])[0]
      if (file) handleFile(file)
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploading, busy, options])

  function handleFile(file: File) {
    if (busy) return
    const name = file.name.toLowerCase()
    if (!name.match(/\.(mp4|mkv|mov|m4v|webm|avi|wmv|flv|mpg|mpeg|ts)$/)) {
      const message = 'Unsupported file type. Use MP4, MKV, MOV, M4V or WebM.'
      if (onToast) onToast(message, 'error')
      else alert(message)
      return
    }
    if (file.size > MAX_BYTES) {
      const message = 'That file is larger than 20 GB.'
      if (onToast) onToast(message, 'error')
      else alert(message)
      return
    }
    onFile(file)
  }

  return (
    <div
      className={`dropzone${over ? ' over' : ''}${busy ? ' busy' : ''}${busy ? ' compact' : ''}`}
      style={busy ? { minHeight: 150, padding: '14px 20px', gap: 8, flexShrink: 0 } : undefined}
      onDragEnter={(e) => {
        e.preventDefault()
        if (busy) return
        dragDepth.current += 1
        setOver(true)
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => {
        e.preventDefault()
        // Decrement unconditionally: a leave that arrives while `busy`
        // (job started mid-drag) must still cancel the highlight, or the
        // "over" state sticks on forever.
        dragDepth.current -= 1
        if (dragDepth.current <= 0) {
          dragDepth.current = 0
          setOver(false)
        }
      }}
      onDrop={(e) => {
        e.preventDefault()
        dragDepth.current = 0
        setOver(false)
        if (uploading || busy) return
        const file = e.dataTransfer.files?.[0]
        if (file) handleFile(file)
      }}
    >
      <div className="file-icon" style={busy ? { fontSize: 22, marginBottom: 2, lineHeight: 1 } : undefined}>{uploading ? <span className="spinner" /> : '🎬'}</div>
      <h2 style={busy ? { fontSize: 16, margin: '2px 0' } : undefined}>
        {uploading
          ? 'Uploading your video'
          : busy
            ? 'Drop another long video here'
            : 'Drop a long video here'}
      </h2>
      <p style={busy ? { fontSize: 12, margin: '2px 0' } : undefined}>
        {uploading
          ? 'The full pipeline starts automatically — no further prompts.'
          : busy
            ? 'This one starts as soon as the current job finishes.'
            : 'Get back vertical, subtitled short clips with a catchy headline on each one.'}
      </p>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file && !busy && !uploading) handleFile(file)
          e.target.value = ''
        }}
      />

      <button className="cf-gold-btn" style={busy && !uploading ? { margin: '6px auto', padding: '8px 18px', opacity: 0.5, cursor: 'not-allowed' } : busy ? { margin: '6px auto', padding: '8px 18px' } : undefined} onClick={() => { if (uploading || busy) return; inputRef.current?.click() }} disabled={uploading || busy}>
        {uploading ? 'Uploading…' : busy ? 'Working on it…' : 'Choose a video'}
      </button>

      {uploadPercent !== null && (
        <div className="upload-bar">
          <div style={{ width: `${uploadPercent}%` }} />
        </div>
      )}

      {!busy && (
        <>
          <div className="formats">MP4 · MKV · MOV · M4V · WEBM · AVI</div>
          <div className="hint">…or paste a file from your clipboard</div>
        </>
      )}
    </div>
  )
}
