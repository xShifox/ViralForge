import { useEffect, useRef, useState } from 'react'
import { desktopApi, chooseFolder } from '../api'

interface Props {
  initial: string
  onPick: (folder: string) => void
  onCancel: () => void
}

export function FolderPicker({ initial, onPick, onCancel }: Props) {
  const [folder, setFolder] = useState(initial || '')
  const [canPick, setCanPick] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Sync if `initial` changes while open (it is recomputed from pendingFile
  // each time the picker mounts).
  useEffect(() => {
    setFolder(initial || '')
  }, [initial])

  useEffect(() => {
    setCanPick(desktopApi() !== null)
    inputRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  async function browseNative() {
    const chosen = await chooseFolder(folder)
    if (chosen) setFolder(chosen)
  }

  return (
    <div className="fb-modal" role="dialog" aria-modal="true" aria-label="Choose output directory">
      <div className="fb-panel">
        <h3 style={{ color: '#f5c542', margin: '0 0 6px', fontSize: 18, fontWeight: 800 }}>Choose Output Directory</h3>
        <p style={{ color: '#a8a29e', fontSize: 13, margin: '0 0 16px' }}>
          Select the folder where your generated vertical clips will be saved.
        </p>

        <input
          ref={inputRef}
          value={folder}
          onChange={(e) => setFolder(e.target.value)}
          placeholder="C:\Users\you\Videos\Clips"
          spellCheck={false}
          style={{
            background: '#08080a',
            border: '1px solid rgba(212,175,55,0.3)',
            borderRadius: '10px',
            padding: '12px 16px',
            color: '#f5c542',
            fontFamily: 'monospace',
            fontSize: '13px',
            width: '100%',
            boxSizing: 'border-box',
            marginBottom: 16,
          }}
        />

        <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
          {canPick && (
            <button type="button" className="cf-secondary-btn" onClick={browseNative}>
              Browse Windows...
            </button>
          )}
          <button type="button" className="cf-gold-btn" onClick={() => onPick(folder.trim())} disabled={!folder.trim()}>
            Save &amp; Apply
          </button>
          <button type="button" className="cf-cancel-btn" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}