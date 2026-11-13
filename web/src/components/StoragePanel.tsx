import { useEffect, useState } from 'react'
import { api } from '../api'
import type { UploadOptions } from '../api'
import type { Health } from '../types'
import { GoldSwitch } from './GoldSwitch'

interface Props {
  options: UploadOptions
  onChange: (options: UploadOptions) => void
  onToast: (message: string, kind?: 'success' | 'error') => void
}

interface StorageBackend {
  key: string
  label: string
  available: boolean
}

export function StoragePanel({ options, onChange, onToast }: Props) {
  const [backends, setBackends] = useState<StorageBackend[]>([])
  const [testing, setTesting] = useState<string | null>(null)
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const status = await api.storageStatus()
        if (alive) setBackends(status.backends)
      } catch (error) {
        if (alive) onToast(`Failed to load storage status: ${(error as Error).message}`, 'error')
      }
      try {
        const healthData = await api.health()
        if (alive) setHealth(healthData)
      } catch {
        /* health is best-effort context here */
      }
    })()
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function testBackend(backend: string) {
    setTesting(backend)
    try {
      const result = await api.storageTest(backend)
      if (result.ok) {
        onToast(`${backends.find(b => b.key === backend)?.label} is working`, 'success')
      } else {
        onToast(`${backends.find(b => b.key === backend)?.label}: ${result.message}`, 'error')
      }
    } catch (error) {
      onToast(`Test failed: ${(error as Error).message}`, 'error')
    } finally {
      setTesting(null)
    }
  }

  function toggleStorage(backend: string) {
    const current = options.storage || []
    const updated = current.includes(backend)
      ? current.filter(b => b !== backend)
      : [...current, backend]
    
    onChange({ ...options, storage: updated })
    onToast(`Storage ${updated.includes(backend) ? 'enabled' : 'disabled'}: ${backends.find(b => b.key === backend)?.label}`)
  }

  return (
    <section className="card">
      <div className="card-head">
        <h2>Storage</h2>
      </div>
      <div className="card-body">
        <div className="field">
          <label>Where to save clips</label>
          <div className="storage-options">
            {backends.map((backend) => (
              <div key={backend.key} className="storage-option">
                <div className="storage-option-content">
                <GoldSwitch
                  checked={options.storage?.includes(backend.key) || false}
                  onChange={() => toggleStorage(backend.key)}
                  label={backend.label}
                />
                <div className="storage-actions">
                  <button
                    className="small ghost test-btn"
                    onClick={() => testBackend(backend.key)}
                    disabled={!backend.available || testing === backend.key}
                    title={backend.available 
                      ? `Test ${backend.label} connection` 
                      : `${backend.label} is not configured`}
                  >
                    {testing === backend.key ? 'Testing…' : 'Test'}
                  </button>
                  {backend.key === 'onedrive' && !backend.available && (
                    <button
                      className="small configure-btn"
                      onClick={() => {
                        onToast('Set CLIPFORGE_ONEDRIVE_TOKEN environment variable with Files.ReadWrite scope')
                      }}
                    >
                      Configure
                    </button>
                  )}
                  {backend.key === 'gdrive' && !backend.available && (
                    <button
                      className="small configure-btn"
                      onClick={() => {
                        onToast('Set CLIPFORGE_GDRIVE_TOKEN environment variable with drive.file scope')
                      }}
                    >
                      Configure
                    </button>
                  )}
                  {backend.key === 'local' && health && (
                    <span className="compute-info">{health.gpu?.compute_label || 'CPU Mode'}</span>
                  )}
                </div>
              </div>
              </div>
            ))}
          </div>
        </div>

        <div className="field">
          <label>Configuration</label>
          <div className="storage-config">
            {backends.map((backend) => (
              <div key={`config-${backend.key}`} className="config-item">
                <span className="config-label">{backend.label}:</span>
                {backend.available ? (
                  <span className="config-status">Available</span>
                ) : (
                  <span className="config-status unavailable">Not configured</span>
                )}
                
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}