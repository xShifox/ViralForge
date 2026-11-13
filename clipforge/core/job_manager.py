"""Job orchestration: persistence, a serialized worker queue, and event fan-out.

Jobs run one at a time on a background thread. Serialising them is deliberate:
each job can saturate the GPU and the CPU encoder, so running several
concurrently would just make every one of them slower while also multiplying
peak memory. The queue keeps the UI responsive and the machine predictable.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

from ..config import JOB_DIR, PLATFORM_PRESETS, get_settings, settings_for_job
from ..core.models import (
    ClipState,
    Event,
    Job,
    JobOptions,
    JobStatus,
    MediaInfo,
    StageName,
)
from ..pipeline.runner import (
    PipelineContext,
    _export_one_clip,
    _slugify,
    cleanup_intermediates,
    publish_clip_outputs,
    run_pipeline,
)

MAX_EVENTS = 400
MAX_JOBS_KEPT = 200


def _detect_for_job(settings, options):
    """Detection settings for a job, with that job's overrides applied."""
    from ..pipeline.runner import _apply_overrides

    return _apply_overrides(settings.detect, options)


def _storage_targets(job) -> list[str]:
    """The storage backends a job was created with, defaulting to local."""
    raw = (job.storage or {}).get("storage_targets")
    if raw:
        try:
            keys = json.loads(raw)
            if isinstance(keys, list) and keys:
                return [str(key) for key in keys]
        except (TypeError, ValueError):
            pass
    return ["local"]


class JobManager:
    """Owns job state, the worker thread and the event subscribers."""
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._events: dict[str, deque] = {}
        self._sinks: dict[str, list] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running: str | None = None
        self._last_persist: dict[str, float] = {}
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    # ------------------------------------------------------------- persistence
    def _path(self, job_id: str) -> Path:
        return JOB_DIR / job_id / "job.json"

    def _load_existing(self) -> None:
        if not JOB_DIR.exists():
            return
        try:
            folders = sorted(JOB_DIR.iterdir(), reverse=True)
        except OSError:
            return
        for folder in folders:
            record = folder / "job.json"
            if not record.is_file():
                continue
            try:
                job = Job.model_validate_json(record.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            # a job that was mid-flight when the server stopped is not resumable
            if job.status in (JobStatus.queued, JobStatus.running):
                job.status = JobStatus.failed
                job.error = "server stopped while this job was running"
                job.message = "Interrupted"
            # The same applies to a clip caught mid-render. Nothing is rendering
            # in a freshly started process, so the state is stale by definition -
            # and without this the clip stays "rendering" forever, showing as
            # permanently busy with no way to re-render it.
            for clip in job.clips:
                if clip.state in (ClipState.rendering, ClipState.pending):
                    clip.state = ClipState.stale
                    clip.error = "interrupted by a server restart - re-render to finish"
            self._jobs[job.id] = job
            self._events[job.id] = deque(job.events, maxlen=MAX_EVENTS)
            del self._jobs[job.id].events[:]
        if len(self._jobs) > MAX_JOBS_KEPT:
            for job_id in list(self._jobs)[: len(self._jobs) - MAX_JOBS_KEPT]:
                self._jobs.pop(job_id, None)
                self._events.pop(job_id, None)
                self._last_persist.pop(job_id, None)

    def persist(self, job: Job) -> None:
        """Write job.json atomically so readers never see a torn file.

        The write goes to a temp file in the same directory and is
        ``os.replace``d over the destination. A crash mid-write then
        leaves either the old or the new record, never half JSON.
        """
        import os

        folder = JOB_DIR / job.id
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "job.json.tmp"
        try:
            tmp.write_text(job.model_dump_json(indent=1), encoding="utf-8")
            os.replace(tmp, folder / "job.json")
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"[clipforge] could not persist job {job.id}: {exc}")

    # ------------------------------------------------------------------ events
    def subscribe(self, job_id: str) -> list:
        """Register a listener; returns the backlog so nothing is missed."""
        with self._lock:
            self._sinks.setdefault(job_id, [])
            return list(self._events.get(job_id, []))

    def unsubscribe(self, job_id: str, sink) -> None:
        with self._lock:
            sinks = self._sinks.get(job_id)
            if sinks and sink in sinks:
                sinks.remove(sink)

    def _emit(self, job_id: str, event: Event) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.events.append(event)
                del job.events[:-MAX_EVENTS]
            self._events.setdefault(job_id, deque(maxlen=MAX_EVENTS)).append(event)
            sinks = list(self._sinks.get(job_id, []))

        for sink in sinks:
            try:
                sink(event)
            except Exception:                       # noqa: BLE001 - a bad listener must not kill the job
                pass

    # -------------------------------------------------------------------- CRUD
    def create(
        self,
        source: Path,
        original_name: str,
        options: JobOptions | None = None,
        storage_keys: list[str] | None = None,
    ) -> Job:
        # No cache purge here. The upload was written by routes.py before this
        # call, and any sweep at this point risks deleting the new job's own
        # source. Old jobs are cleared only when the user presses Clear.
        job = Job(status=JobStatus.queued, options=options or JobOptions())
        job.source = MediaInfo(path=str(source), filename=original_name)
        job.storage["original_name"] = original_name
        if storage_keys:
            job.storage["storage_targets"] = json.dumps(storage_keys)

        with self._lock:
            self._jobs[job.id] = job
            self._events[job.id] = deque(maxlen=MAX_EVENTS)
            self._cancels[job.id] = threading.Event()
            (JOB_DIR / job.id).mkdir(parents=True, exist_ok=True)
            self.persist(job)
            self._queue.put(job.id)

        self._ensure_worker()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in (JobStatus.queued, JobStatus.running):
                return False
            event = self._cancels.get(job_id)
            if event is not None:
                event.set()
            # Stop right away instead of waiting for the worker to notice
            # between stages. The worker still exits via Cancelled inside the
            # pipeline (per-segment during transcribe, per-clip during render),
            # but the status flip is the signal the UI polls on — leaving it
            # `running` until the worker finishes made Cancel look hung.
            job.status = JobStatus.cancelled
            job.message = "Cancelled"
            job.finished_at = time.time()
            self.persist(job)
        self._emit(job_id, Event(stage=job.stage.value, level="warn", message="cancellation requested"))
        self._emit(job_id, Event(stage=job.stage.value, level="warn", message="job cancelled", progress=job.progress))
        return True

    def mark_bundle_failed(self, job_id: str, reason: str) -> Job | None:
        """Record that this job's bulk bundle is unusable, so it is never served.

        A bundle that cannot be opened is not a download, it is a corrupt file
        with a 200 attached to it. Marking the job failed here means the UI stops
        offering the download instead of handing the user a truncated archive.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = JobStatus.failed
            job.stage = StageName.export
            job.message = "Export bundle is corrupt"
            job.error = reason
            job.finished_at = job.finished_at or time.time()
            self.persist(job)
        self._emit(job_id, Event(stage=job.stage.value, level="error", message=reason))
        return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id == self._running:
                # The worker still owns this job; deleting underneath it
                # would let it recreate the folder on its next persist and
                # resurrect the record.
                return False
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            self._cancels.pop(job_id, None)
            self._sinks.pop(job_id, None)
            self._events.pop(job_id, None)
            self._last_persist.pop(job_id, None)
        folder = JOB_DIR / job_id
        try:
            if folder.exists():
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass
        try:
            from .cleanup import remove_job_source_files
            from ..config import UPLOAD_DIR
            src = getattr(getattr(job, "source", None), "path", None)
            if src:
                remove_job_source_files(folder, src)
            # also sweep uploads for safety
            if UPLOAD_DIR.exists():
                pass
        except Exception:
            pass
        return True

    def protected_active(self) -> tuple[list[str], list[Path]]:
        """Ids and source paths of jobs a purge must never delete.

        Queued and running jobs still point at their upload, so sweeping it
        would turn them into zombie records that fail with "source video is
        missing". Used by Clear, the delete route and the new-job purge.
        """
        with self._lock:
            jobs = list(self._jobs.items())
            running = self._running
        ids: list[str] = []
        paths: list[Path] = []
        for jid, job in jobs:
            if job.status not in (JobStatus.queued, JobStatus.running):
                continue
            ids.append(jid)
            raw = getattr(getattr(job, "source", None), "path", None)
            if raw:
                try:
                    paths.append(Path(raw).resolve())
                except Exception:
                    paths.append(Path(raw))
        if running and running not in ids:
            ids.append(running)
        return ids, paths

    def clear_all_ephemeral(self) -> dict:
        """Wipe all disposable cache/jobs/uploads; keep saved output folder files."""
        keep_ids, keep_paths = self.protected_active()
        try:
            from . import cleanup
            from ..config import UPLOAD_DIR
            stats = cleanup.purge_ephemeral_cache_retry(
                JOB_DIR, UPLOAD_DIR, None,
                preserve_job_ids=keep_ids,
                remove_job_dirs=True,
                preserve_paths=keep_paths,
                attempts=30, delay_s=0.4,
            )
        except Exception:
            stats = {}
        running = self._running
        with self._lock:
            for jid in list(self._jobs.keys()):
                if running and jid == running:
                    continue
                if not (JOB_DIR / jid).exists():
                    self._jobs.pop(jid, None)
                    self._events.pop(jid, None)
                    self._cancels.pop(jid, None)
                    self._sinks.pop(jid, None)
                    self._last_persist.pop(jid, None)
        return stats

    def update_clip(self, job_id: str, clip_id: str, changes: dict[str, Any]) -> Job | None:
        """Apply an edit from the UI and mark the clip as needing a re-render."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            clip = next((c for c in job.clips if c.id == clip_id), None)
            if clip is None:
                return None

            for key, value in changes.items():
                if not hasattr(clip, key):
                    continue
                if key in ("start", "end") and value is not None:
                    value = max(0.0, float(value))
                if key == "title" and value is not None:
                    value = str(value)[:120]
                setattr(clip, key, value)

            # keep the window valid
            if clip.end <= clip.start:
                clip.end = clip.start + 5.0
            clip.duration_override = None
            if clip.state == ClipState.ready:
                clip.state = ClipState.stale
            job.storage[f"clip:{clip_id}:dirty"] = str(time.time())
            self.persist(job)
        self._emit(job_id, Event(stage=job.stage.value, level="info", message="clip updated", progress=job.progress))
        return job

    def rerender_clip(self, job_id: str, clip_id: str) -> tuple[Job | None, str]:
        """Re-render one clip in the background after a UI edit."""
        from ..render import ffmpeg
        from ..render.renderer import ClipTooShort, RenderRequest, render_clip

        job = self.get(job_id)
        if job is None:
            return None, "job not found"
        clip = next((c for c in job.clips if c.id == clip_id), None)
        if clip is None:
            return None, "clip not found"
        if job.source is None:
            return None, "job has no source video"
        if not Path(job.source.path).exists():
            return None, "source video is gone"

        settings = settings_for_job(get_settings(), job.options)
        detect = _detect_for_job(settings, job.options)
        minimum, maximum = detect.bounds()
        span = clip.end - clip.start
        if span < minimum - 0.05:
            return None, (
                f"this clip is {span:.1f}s; the minimum is {minimum:.0f}s. "
                f"Extend the end point before re-rendering."
            )
        if span > maximum + 0.05:
            return None, (
                f"this clip is {span:.1f}s; the maximum is {maximum:.0f}s. "
                f"Shorten the window before re-rendering."
            )

        primary = (job.options.platforms or settings.platforms or ["tiktok"])[0]
        out_dir = JOB_DIR / job_id / "video"
        out_dir.mkdir(parents=True, exist_ok=True)
        export_dir = JOB_DIR / job_id / "exports"
        variant_dir = JOB_DIR / job_id / "variants"
        export_dir.mkdir(parents=True, exist_ok=True)
        variant_dir.mkdir(parents=True, exist_ok=True)
        platforms = [p for p in (job.options.platforms or settings.platforms)
                     if p in PLATFORM_PRESETS] or ["tiktok"]

        def work() -> None:
            # Progress is reported in whole percent only. ffmpeg calls the
            # callback far more often than that, and each call would otherwise
            # mean a persist plus an event for a change nobody can see.
            # The high-water mark (`peak`) is the fix for a bar that jumped
            # backwards: the renderer retries a failed hardware encode with a
            # second ffmpeg process that starts at 0 again, so without the
            # clamp the bar went 44% -> 2% and read as stuck/restarting. A
            # lower percent is now ignored until the new attempt passes the
            # mark. The heartbeat advances the bar toward the current stage
            # ceiling when ffmpeg's -progress pipe is silent (ASS build,
            # filtergraph, and the compat transcode's ffmpeg.run all block
            # with no per-frame callback), so 1% is never a silent wait.
            report_lock = threading.Lock()
            peak: list[int] = [-1]
            last_emit_ts: list[float] = [time.monotonic()]
            stage_ceiling: list[int] = [45]
            stage_note: list[str] = ["rendering"]
            stop_heartbeat = threading.Event()

            def report(percent: float, note: str) -> None:
                shown = max(0, min(100, int(percent)))
                with report_lock:
                    if shown < peak[0]:
                        # Restarted attempt (e.g. hardware -> CPU fallback)
                        # re-reporting from zero. Keep the fresh note, but
                        # do not regress the bar, persist, or emit.
                        clip.progress_note = note
                        return
                    if shown == peak[0]:
                        clip.progress_note = note
                        return
                    peak[0] = shown
                    clip.progress = float(shown)
                    clip.progress_note = note
                    last_emit_ts[0] = time.monotonic()
                with self._lock:
                    self.persist(job)
                self._emit(job_id, Event(
                    stage=job.stage.value, level="info",
                    message=f"re-rendering clip {clip.index + 1}: {note} {shown}%",
                    progress=shown,
                ))

            def _heartbeat() -> None:
                while not stop_heartbeat.wait(0.6):
                    ceiling = stage_ceiling[0]
                    note = stage_note[0]
                    current = peak[0]
                    if current < 0 or current >= ceiling:
                        continue
                    if time.monotonic() - last_emit_ts[0] < 0.75:
                        continue
                    report(min(ceiling, current + 3), note)

            clip.state = ClipState.rendering
            # One message per step. The first call is the "it has started" signal
            # the drawer and the pipeline log both read, so there is no separate
            # opening event to duplicate it.
            report(1, "starting")
            hb = threading.Thread(target=_heartbeat, daemon=True, name=f"rerender-hb-{clip_id}")
            hb.start()
            try:
                _t0 = time.perf_counter()
                artifacts = render_clip(RenderRequest(
                    clip=clip,
                    source=job.source.path,
                    media=job.source,
                    work_dir=out_dir,
                    render=settings.render,
                    brand=settings.brand,
                    preset_key=primary,
                    clip_index=clip.index,
                    min_duration=minimum,
                    options={
                        "track_talker": bool(getattr(job.options, "track_talker", False)),
                        "speaker_turns": job.storage.get("speaker_turns"),
                    },
                    speaker_turns=job.storage.get("speaker_turns"),
                ), on_progress=lambda fraction: report(2 + fraction * 43, "rendering"))
                _render_dt = time.perf_counter() - _t0
                self._emit(job_id, Event(
                    stage=job.stage.value, level="info",
                    message=f"re-rendering clip {clip.index + 1}: render took {_render_dt:.1f}s",
                ))
                # source_preset tells the export step which preset this render
                # used, so it can reuse the file for platforms that ask for the
                # same thing. Kept in step with the render stage.
                clip.outputs = {"source": str(artifacts["video"]),
                                "source_preset": primary}
                if artifacts.get("duration"):
                    clip.duration_override = float(artifacts["duration"])
                if artifacts["poster"] != artifacts["video"]:
                    clip.poster = artifacts["poster"].name

                # Re-rendering has to rebuild the exported files too. Leaving
                # only the scratch path in `outputs` would drop the per-platform
                # entries and the storage links, so the download button and the
                # bulk zip would lose this clip entirely.
                slug = _slugify(Path(job.source.filename).stem)
                variants = {"count": 0}

                def on_variant() -> None:
                    # The variants are a fixed, small number of steps, so this is
                    # a stepper rather than a real progress signal.
                    variants["count"] += 1
                    done = min(variants["count"], len(platforms))
                    report(45 + (done / max(1, len(platforms))) * 25,
                           f"exporting {done}/{len(platforms)}")

                def on_variant_fraction(fraction: float) -> None:
                    # Live ffmpeg progress inside one variant re-encode,
                    # mapped onto this variant's slice of the export band so
                    # the bar does not sit silent on the stepper alone.
                    idx = max(1, variants["count"])
                    per = 25.0 / max(1, len(platforms))
                    base = 45 + ((idx - 1) * per)
                    clamped = max(0.0, min(1.0, float(fraction)))
                    report(base + clamped * per,
                           f"exporting {min(idx, len(platforms))}/{len(platforms)}")

                stage_ceiling[0] = 70
                stage_note[0] = "exporting"
                _t0 = time.perf_counter()
                exported = _export_one_clip(
                    clip, job.source, settings, platforms, minimum, maximum,
                    export_dir, variant_dir, slug, out_dir, progress=on_variant,
                    variant_progress=on_variant_fraction,
                    job_options=job.options,
                    speaker_turns=job.storage.get("speaker_turns"),
                )
                _export_dt = time.perf_counter() - _t0
                self._emit(job_id, Event(
                    stage=job.stage.value, level="info",
                    message=f"re-rendering clip {clip.index + 1}: export took {_export_dt:.1f}s",
                ))
                if not exported:
                    # Marking the clip ready with no platform files would leave
                    # the download button 409ing and the clip missing from the
                    # bundle, so report it as needing attention instead.
                    clip.state = ClipState.stale
                    clip.error = (
                        "the render finished but no platform file could be "
                        "written - check the job warnings"
                    )
                    message = f"clip {clip.index + 1} re-rendered without exports"
                    level = "error"
                else:
                    # Compat pre-warm BEFORE ready: it is the instant MP4
                    # fallback the modal plays, so the clip is not usable
                    # until it exists. Webm stays on its background pool.
                    stage_ceiling[0] = 92
                    stage_note[0] = "pre-warming previews"
                    report(72, "pre-warming previews")
                    _t0 = time.perf_counter()
                    try:
                        from ..pipeline.runner import _prewarm_clip_preview

                        def _pw_log(msg: str, lvl: str = "info") -> None:
                            self._emit(job_id, Event(stage=job.stage.value, level=lvl, message=msg))

                        _prewarm_clip_preview(
                            clip,
                            artifacts["video"],
                            JOB_DIR / job_id / "previews",
                            job.options,
                            _pw_log,
                            fallback_settings=settings,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    _compat_dt = time.perf_counter() - _t0
                    self._emit(job_id, Event(
                        stage=job.stage.value, level="info",
                        message=f"re-rendering clip {clip.index + 1}: compat pre-warm took {_compat_dt:.1f}s (webm queued background)",
                    ))
                    report(92, "pre-warming previews")
                    # Local publish is a dict update (no upload), so it stays
                    # on the critical path and the download works the moment
                    # the clip is ready. Anything else (cloud) uploads real
                    # bytes and joins the background zip rebuild below.
                    targets = _storage_targets(job)
                    local_only = not targets or targets == ["local"]
                    _t0 = time.perf_counter()
                    if local_only:
                        publish_clip_outputs([clip], export_dir, slug, targets)
                        _publish_dt = time.perf_counter() - _t0
                        self._emit(job_id, Event(
                            stage=job.stage.value, level="info",
                            message=f"re-rendering clip {clip.index + 1}: publish took {_publish_dt:.1f}s",
                        ))
                    report(96, "pre-warming previews")
                    clip.state = ClipState.ready
                    clip.error = None
                    # Single-clip file copy is done; the bulk zip is NOT
                    # rebuilt on this path. Re-zipping every clip (~200MB)
                    # cost tens of seconds after the actual render, and
                    # zipping the exports folder alone would overwrite a
                    # complete bundle with just this clip. The rebuild runs
                    # in the background from every export root.
                    self._emit(job_id, Event(
                        stage=job.stage.value, level="info",
                        message=f"re-rendering clip {clip.index + 1}: zip rebuild deferred - bulk zip left stale until background finishes; single-clip download is current",
                    ))

                    def _bg_bundle() -> None:
                        from ..pipeline.runner import rebuild_bundle_from_job

                        def _bg_log(msg: str, lvl: str = "info") -> None:
                            self._emit(job_id, Event(stage=job.stage.value, level=lvl, message=msg))

                        _bg_t0 = time.perf_counter()
                        try:
                            if not local_only:
                                publish_clip_outputs([clip], export_dir, slug, targets)
                            bundle = rebuild_bundle_from_job(
                                job, JOB_DIR / job_id, slug, _bg_log)
                            if bundle is not None:
                                with self._lock:
                                    job.storage["zip"] = bundle.name
                                    self.persist(job)
                        except Exception as exc:  # noqa: BLE001 - background only
                            _bg_log(
                                f"re-rendering clip {clip.index + 1}: zip rebuild deferred "
                                f"({type(exc).__name__}: {exc}); bulk zip left stale",
                                "warn",
                            )
                            return
                        _bg_dt = time.perf_counter() - _bg_t0
                        if bundle is None:
                            _bg_log(
                                f"re-rendering clip {clip.index + 1}: zip rebuild deferred - bulk zip left stale",
                                "warn",
                            )
                        else:
                            _bg_log(
                                f"re-rendering clip {clip.index + 1}: zip rebuild finished in {_bg_dt:.1f}s (background)",
                            )

                    threading.Thread(target=_bg_bundle, daemon=True, name=f"rerender-zip-{clip_id}").start()
                    message = f"clip {clip.index + 1} re-rendered"
                    level = "success"
            except (ffmpeg.FFmpegError, ClipTooShort) as exc:
                clip.state = ClipState.failed
                clip.error = str(exc)[:400]
                message = f"re-render failed: {exc}"
                level = "error"
            except Exception as exc:                            # noqa: BLE001
                # Anything unexpected used to escape and kill this thread, leaving
                # the clip reading "rendering" for ever with no way to tell
                # whether it was still working. A stranded clip looks identical
                # to a slow one, which is the worst thing this panel can do, so
                # every failure gets reported rather than swallowed.
                clip.state = ClipState.failed
                clip.error = f"re-render failed: {type(exc).__name__}: {exc}"[:400]
                message = clip.error
                level = "error"
            finally:
                stop_heartbeat.set()
            # Cleared here rather than left at whatever it last reached: a bar
            # frozen at 94% on a clip that is ready or failed is worse than none,
            # because it reads as still working.
            clip.progress = None
            clip.progress_note = ""
            with self._lock:
                self.persist(job)
            self._emit(job_id, Event(stage=job.stage.value, level=level, message=message))

        # Mark the clip as rendering *before* the thread starts. The response to
        # the edit that triggered this is the only thing the UI has to go on, and
        # if the state is still the old one at that point there is no signal that
        # anything is in flight, so nothing knows to look for the result.
        # Progress starts at 1% here too, not only in the worker thread, so
        # the PATCH response the editor reads is never a null/old bar.
        clip.state = ClipState.rendering
        clip.progress = 1.0
        clip.progress_note = "starting"
        with self._lock:
            self.persist(job)

        threading.Thread(target=work, daemon=True, name=f"rerender-{clip_id}").start()
        return job, "re-render started"

    # ------------------------------------------------------------------ worker
    def _ensure_worker(self) -> None:
        """Start the single worker, guarded so two concurrent creates
        cannot spawn two workers (which would break the one-at-a-time
        GPU/encoder invariant and interleave ffmpeg on the same card)."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_loop, daemon=True, name="clipforge-worker")
            self._worker.start()

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            except Exception:                       # noqa: BLE001 - never kill the worker
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            cancel = self._cancels.get(job_id) or threading.Event()
        if job is None or job.status == JobStatus.cancelled:
            return

        # per-job overrides live on a copy so this run's branding and model
        # choice cannot leak into other jobs or the settings API
        settings = settings_for_job(get_settings(), job.options)
        source = Path(job.source.path) if job.source else None
        if source is None or not source.exists():
            with self._lock:
                job.status = JobStatus.failed
                job.error = "source video is missing"
                job.message = "Failed"
                self.persist(job)
            self._emit(job_id, Event(level="error", message="source video is missing"))
            return

        storage_keys: list[str] = []
        raw_targets = job.storage.get("storage_targets")
        if raw_targets:
            try:
                storage_keys = json.loads(raw_targets)
            except ValueError:
                storage_keys = []

        with self._lock:
            job.status = JobStatus.running
            job.started_at = time.time()
            job.message = "Starting"
            self._running = job_id
        try:
            from .cleanup import mark_job_active, mark_job_inactive
            mark_job_active(job_id)
        except Exception:
            pass

        ctx = PipelineContext(
            job=job,
            settings=settings,
            source_path=source,
            work_dir=JOB_DIR / job_id,
            cancel=cancel,
            emit=lambda event: self._emit(job_id, event),
            update=lambda: self._throttled_persist(job),
            storage_keys=storage_keys,
        )
        # expose the context so the runner can finish the record
        job_ctx = ctx

        try:
            run_pipeline(job_ctx)
        finally:
            # Final safety net: only the regenerable analysis wav is removed.
            # The uploaded source and every rendered file are kept for preview
            # and Save & re-render; Clear is the only thing that deletes them.
            try:
                if job.status == JobStatus.completed:
                    result = cleanup_intermediates(
                        JOB_DIR / job_id,
                        keep=bool(getattr(settings, "keep_intermediate", False)),
                        log=lambda msg, lvl="info": self._emit(
                            job_id, Event(stage=job.stage.value, level=lvl, message=msg)
                        ),
                    )
                    if result.get("bytes"):
                        self._emit(
                            job_id,
                            Event(
                                stage=job.stage.value,
                                level="info",
                                message=f"removed {result['bytes'] / 1e6:.1f} MB of analysis audio",
                            ),
                        )
            except Exception:  # noqa: BLE001 - housekeeping must never kill the worker
                pass
            try:
                from .cleanup import mark_job_inactive
                mark_job_inactive(job_id)
            except Exception:
                pass
            with self._lock:
                if self._running == job_id:
                    self._running = None
                job.finished_at = time.time()
                self.persist(job)

        # keep the job's public event list bounded
        with self._lock:
            if len(job.events) > MAX_EVENTS:
                del job.events[:-MAX_EVENTS]

    def resume_export(self, job_id: str) -> tuple[Job | None, str]:
        """Re-run only the export stage for a job whose clips are already rendered.

        Rendering is the expensive part - a long video can spend minutes in
        Whisper and again encoding every clip. If a run is interrupted during
        export, the rendered videos are still on disk, so the job can be finished
        without repeating any of that. Returns ``(job, message)``; the job is
        ``None`` when the run cannot be resumed.
        """
        from ..pipeline.runner import stage_export

        job = self.get(job_id)
        if job is None:
            return None, "job not found"
        if job.status not in (JobStatus.failed, JobStatus.completed, JobStatus.cancelled):
            return None, "this job is still running"
        if job.source is None or not Path(job.source.path).exists():
            return None, "the source video is gone, so nothing can be re-exported"
        if not job.clips:
            return None, (
                "this run never produced any clips, so there is nothing to finish - "
                "start a new job with the same video"
            )

        settings = settings_for_job(get_settings(), job.options)
        primary = (job.options.platforms or settings.platforms or ["tiktok"])[0]
        video_dir = JOB_DIR / job_id / "video"

        # The export stage consumes outputs["source"] and leaves per-platform
        # names behind, so a clip caught mid-export has neither. Point it back
        # at the render that is already on disk.
        missing: list[int] = []
        for clip in job.clips:
            if clip.outputs.get("source") and Path(clip.outputs["source"]).exists():
                continue
            rendered = video_dir / f"clip{clip.index:02d}_{primary}.mp4"
            if rendered.is_file():
                # drop half-written platform entries so the stage redoes them
                for key in [k for k in clip.outputs if not k.startswith("url:")]:
                    clip.outputs.pop(key, None)
                clip.outputs["source"] = str(rendered)
            else:
                missing.append(clip.index + 1)
        if missing:
            return None, (
                f"clips {', '.join(str(n) for n in missing)} were never rendered, "
                f"so this run cannot be finished without starting over"
            )

        job.status = JobStatus.running
        job.stage = StageName.export
        job.progress = 0.0
        job.error = None
        job.message = "Finishing the export"
        self._emit(job_id, Event(stage=job.stage.value, level="info",
                                 message="resuming the export stage"))

        ctx = PipelineContext(
            job=job,
            settings=settings,
            source_path=Path(job.source.path),
            work_dir=JOB_DIR / job_id,
            cancel=threading.Event(),
            emit=lambda event: self._emit(job_id, event),
            update=lambda: self._throttled_persist(job),
            storage_keys=_storage_targets(job),
        )
        try:
            stage_export(ctx)
        except Exception as exc:                            # noqa: BLE001
            job.status = JobStatus.failed
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "Failed"
            with self._lock:
                self.persist(job)
            return None, f"the export failed again: {exc}"
        finally:
            job.finished_at = time.time()

        job.stage = StageName.done
        job.progress = 1.0
        job.status = JobStatus.completed
        job.message = f"{sum(1 for c in job.clips if c.state == ClipState.ready)} clips ready"
        with self._lock:
            self.persist(job)
        self._emit(job_id, Event(stage=StageName.done.value, level="success",
                                 message="completed"))
        return job, "export finished"

    def _throttled_persist(self, job: Job) -> None:
        """Persist at most ~4x/second: the stages call this per frame.

        The last-persist timestamp lives in a manager-side dict guarded
        by ``_lock``. Stashing it on the pydantic ``Job`` with
        ``object.__setattr__`` (the old code) was a race: two threads
        could both pass the 0.25 s check and both write, and the
        attribute never survived a reload anyway.
        """
        now = time.time()
        with self._lock:
            last = self._last_persist.get(job.id, 0.0)
            if now - last < 0.25:
                return
            self._last_persist[job.id] = now
            job.updated_at = now
        self.persist(job)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status.value] = counts.get(job.status.value, 0) + 1
            return {
                "total": len(self._jobs),
                "queued": self._queue.qsize(),
                "running": self._running,
                "by_status": counts,
            }


manager = JobManager()
