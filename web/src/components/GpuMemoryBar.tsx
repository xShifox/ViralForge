import { useEffect, useRef, useState } from 'react'
import { readGpuMemory } from '../api'
import type { GpuMemory } from '../types'

interface Props {
  /** The card's name, so the bar is about something rather than about nothing. */
  name: string
  /** How hard the card is being driven. Shown because `light` is not a verdict. */
  mode: string
  /** True while a job is running, which is when the number moves. */
  busy: boolean
}

/** The card's video memory, live, with this process's share called out.
 *
 *  Two things are being shown and they are not the same kind of thing, which is
 *  why the bar is in two parts rather than one number:
 *
 *  - total, used and free are read straight from the driver, so they are facts.
 *  - the darker segment is this process's share. Windows does not report
 *    per-process video memory at all - `nvidia-smi --query-compute-apps` lists
 *    the process and answers `[N/A]` for its memory - so this is *measured*, as
 *    the rise above the level the card sat at while ViralForge held nothing.
 *
 *  So it is a real number with a stated limitation rather than a reading, and
 *  when there is no baseline yet it is left out entirely instead of being drawn
 *  as zero, which would read as "this process is using nothing" - a claim nobody
 *  can support at that moment. */
export function GpuMemoryBar({ name, mode, busy }: Props) {
  const [reading, setReading] = useState<GpuMemory | null>(null)
  const [failed, setFailed] = useState(false)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const next = await readGpuMemory()
        if (!cancelled) { setReading(next); setFailed(false) }
      } catch {
        // One bad poll is not worth shouting about - the driver refuses
        // occasionally - but a card that never answers is, so it is remembered.
        if (!cancelled) setFailed(true)
      } finally {
        if (!cancelled) schedule()
      }
    }

    function schedule() {
      // Faster while work is happening, because that is when the number is
      // interesting and when someone watching is waiting to see it move.
      timer.current = window.setTimeout(poll, busy ? 1000 : 4000)
    }

    void poll()
    return () => {
      cancelled = true
      if (timer.current !== null) window.clearTimeout(timer.current)
    }
  }, [busy])

  if (!name || !reading || !reading.total_mb) {
    return (
      <div className="vram" title={failed
        ? 'The graphics driver is not answering.'
        : 'Looking for the graphics card…'}>
        <span className="vram-label">VRAM</span>
        <span className="vram-value">—</span>
      </div>
    )
  }

  const total = reading.total_mb
  const used = Math.max(0, Math.min(total, reading.used_mb))
  const free = Math.max(0, Math.min(total - used, reading.free_mb))
  const reserved = Math.max(0, total - used - free)
  const ours = reading.process_known && reading.process_mb !== null
    ? Math.max(0, Math.min(used, reading.process_mb))
    : 0
  const others = Math.max(0, used - ours)
  const pct = (mb: number) => `${(mb / total) * 100}%`

  const title = [
    `${name} - ${mode} mode`,
    `${total.toLocaleString()} MB of video memory`,
    `${ours ? `${ours.toLocaleString()} MB in this process` : 'this process: not measurable yet'}`,
    `${others.toLocaleString()} MB in other programs`,
    `${free.toLocaleString()} MB free`,
    reserved ? `${reserved.toLocaleString()} MB held by the display` : '',
    reading.utilization_pct ? `card ${reading.utilization_pct}% busy` : 'card idle',
    '',
    'Windows does not report per-process video memory, so this process',
    'share is measured as the rise above the level the card sat at while',
    'ViralForge held nothing.',
  ].filter(Boolean).join('\n')

  return (
    <div className="vram" title={title}>
      <span className="vram-label">VRAM</span>
      <div className="vram-bar" role="img"
        aria-label={`${used.toLocaleString()} of ${total.toLocaleString()} megabytes of video memory used, ${ours.toLocaleString()} in this process`}>
        <span className="vram-seg vram-ours" style={{ width: pct(ours) }} />
        <span className="vram-seg vram-other" style={{ width: pct(others) }} />
        <span className="vram-seg vram-reserved" style={{ width: pct(reserved) }} />
        <span className="vram-seg vram-free" style={{ width: pct(free) }} />
      </div>
      <span className="vram-value">
        {used.toLocaleString()}
        <span className="vram-total"> / {total.toLocaleString()} MB</span>
      </span>
    </div>
  )
}
