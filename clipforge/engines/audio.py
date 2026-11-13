"""Acoustic analysis: frame features, speech/silence, laughter, speaker turns.

Everything is computed with numpy from a 16 kHz mono PCM wav decoded by
ffmpeg, so there is no dependency on librosa/soundfile.

The detectors here are *heuristics*, deliberately documented as such:

* speech/silence  - adaptive energy + zero-crossing gating against a measured
  noise floor, which is far more robust on real recordings than a fixed dBFS.
* pauses          - runs of gated-out frames.
* speaker turns   - pitch/energy discontinuity across transcript segment
  boundaries. This is a lightweight approximation, not a trained diariser:
  two speakers with a similar pitch will merge, which is an acceptable
  trade-off compared to pulling in a multi-megabyte model.
* laughter        - loud, high-centroid, pitch-unstable bursts with 3-8 Hz
  amplitude modulation. Combined downstream with textual laughter cues
  ("haha", "hilarious"), which are far more reliable than audio alone.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.models import (
    AudioAnalysis,
    Frame,
    LaughterEvent,
    PauseEvent,
    Segment,
    SpeakerTurn,
)

#: How far below the speech level the noise floor must stay. Without a cap,
#: wall-to-wall speech has no silence to measure the floor from, and the floor
#: estimate lands inside the speech band - then ``floor + 5`` gates the speech
#: out.
NOISE_FLOOR_HEADROOM = 6.0


def speech_spans(analysis, window_start: float, window_end: float,
                 min_speech: float = 0.05) -> list[tuple[float, float]]:
    """Speech intervals inside [window_start, window_end] - complement of pauses.

    Pauses are merged and clipped to the window; gaps shorter than
    ``min_speech`` are breath, not speech, and are dropped.
    """
    start, end = float(window_start), float(window_end)
    if end <= start:
        return []
    pauses: list[tuple[float, float]] = []
    for pause in (getattr(analysis, "pauses", None) or []):
        lo = max(start, float(pause.start))
        hi = min(end, float(pause.end))
        if hi > lo:
            pauses.append((lo, hi))
    pauses.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in pauses:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    spans: list[tuple[float, float]] = []
    cursor = start
    for lo, hi in merged:
        if lo > cursor and lo - cursor >= min_speech:
            spans.append((cursor, lo))
        cursor = max(cursor, hi)
    if end - cursor >= min_speech:
        spans.append((cursor, end))
    return spans


# ------------------------------------------------------------------ wav loading

def load_wav_mono(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Read a PCM wav as float32 mono in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if rate != target_sr and data.size:
        duration = data.size / rate
        n_out = int(round(duration * target_sr))
        src_idx = np.linspace(0.0, data.size - 1, n_out, dtype=np.float64)
        data = np.interp(src_idx, np.arange(data.size, dtype=np.float64), data).astype(np.float32)
        rate = target_sr
    return np.ascontiguousarray(data, dtype=np.float32), rate


# ------------------------------------------------------------------ framing

def _frames(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    if signal.size < win:
        signal = np.pad(signal, (0, win - signal.size))
    n = 1 + (signal.size - win) // hop
    strides = (signal.strides[0] * hop, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=(n, win), strides=strides)


def _autocorr_f0(block: np.ndarray, sr: int, fmin: float = 55.0, fmax: float = 400.0) -> float:
    """Normalised-autocorrelation pitch estimate. Returns 0.0 when unvoiced."""
    block = block - block.mean()
    energy = float(np.dot(block, block))
    if energy < 1e-9:
        return 0.0

    size = block.size
    spectrum = np.fft.rfft(block, n=2 * size)
    corr = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    if corr[0] <= 0:
        return 0.0
    corr = corr / corr[0]

    lag_min = max(2, int(sr / fmax))
    lag_max = min(size - 2, int(sr / fmin))
    if lag_max <= lag_min:
        return 0.0
    window = corr[lag_min:lag_max]
    peak = int(np.argmax(window))
    value = float(window[peak])
    if value < 0.30:                      # below this the "pitch" is noise
        return 0.0
    lag = lag_min + peak
    # parabolic interpolation for sub-sample precision
    if 0 < peak < window.size - 1:
        a, b, c = window[peak - 1], window[peak], window[peak + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            lag += float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    f0 = sr / lag
    return float(f0) if fmin <= f0 <= fmax else 0.0


def frame_features(
    signal: np.ndarray,
    sr: int,
    hop_s: float = 0.02,
    win_s: float = 0.04,
    with_pitch: bool = True,
) -> tuple[list[Frame], float]:
    """Compute per-frame acoustic descriptors.

    Returns ``(frames, noise_floor_db)``.
    """
    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * win_s))
    blocks = _frames(signal, win, hop)
    if blocks.size == 0:
        return [], -70.0

    times = np.arange(blocks.shape[0]) * hop / sr
    energy = np.sqrt(np.maximum((blocks ** 2).mean(axis=1), 1e-12))
    rms_db = 20.0 * np.log10(energy)

    # Noise floor = median of the quietest 40% of frames. Using a low
    # percentile here is a trap: files assembled from clips often contain
    # stretches of true digital silence, which would drag an estimate down to
    # -120 dB and make every real signal look like background noise.
    floor_cut = float(np.percentile(rms_db, 40))
    quiet = rms_db[rms_db <= floor_cut]
    noise_floor = float(np.median(quiet)) if quiet.size else float(np.median(rms_db))
    # Cap the floor below the speech level: wall-to-wall speech leaves no
    # silence to measure from, so the estimate must not sit inside the speech.
    p90 = float(np.percentile(rms_db, 90)) if rms_db.size else noise_floor
    noise_floor = min(noise_floor, p90 - NOISE_FLOOR_HEADROOM)

    # zero-crossing rate
    signs = np.signbit(blocks)
    zcr = (signs[:, 1:] != signs[:, :-1]).mean(axis=1)

    # spectral centroid (normalised 0..1 of nyquist)
    window = np.hanning(blocks.shape[1]).astype(np.float32)
    spectra = np.abs(np.fft.rfft(blocks * window, axis=1))
    freqs = np.fft.rfftfreq(blocks.shape[1], d=1.0 / sr)
    total = spectra.sum(axis=1) + 1e-9
    centroid = (spectra * freqs).sum(axis=1) / total
    centroid_n = np.clip(centroid / (sr / 2), 0.0, 1.0)

    f0 = np.zeros(blocks.shape[0], dtype=np.float32)
    if with_pitch:
        # only analyse frames that plausibly contain voice
        gate = rms_db > (noise_floor + 6.0)
        for i in np.nonzero(gate)[0]:
            f0[i] = _autocorr_f0(blocks[i], sr)

    frames = [
        Frame(
            t=float(times[i]),
            rms_db=float(rms_db[i]),
            energy=float(energy[i]),
            zcr=float(zcr[i]),
            centroid=float(centroid_n[i]),
            f0=float(f0[i]),
        )
        for i in range(blocks.shape[0])
    ]
    return frames, noise_floor


# ------------------------------------------------------------------ gating

@dataclass
class Gate:
    voiced: np.ndarray
    speech: np.ndarray
    noise_floor: float
    speech_floor: float


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's between-class-variance threshold.

    Far more reliable than a fixed dBFS gate for real recordings, where the
    speech/silence levels move around a lot between sources.
    """
    if values.size < 8:
        return float(np.median(values))
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(np.median(values))
    p = hist / total
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    idx = int(np.argmax(sigma_b))
    return float(centers[idx])


def _gate(frames: list[Frame], noise_floor: float) -> Gate:
    """Mark frames as voiced / speech using an adaptive threshold."""
    db = np.array([f.rms_db for f in frames], dtype=np.float32)
    zcr = np.array([f.zcr for f in frames], dtype=np.float32)

    # Otsu split, but never let the threshold sink into the noise floor.
    speech_floor = max(_otsu(db), noise_floor + 5.0)
    loud = db > (speech_floor - 1.0)
    # a frame is speech if it is loud enough, or loud-ish with speech-like
    # zero-crossing behaviour (fricatives sit below the energy threshold)
    speech = loud | ((db > (speech_floor - 4.0)) & (zcr < 0.28))
    voiced = speech & (db > (speech_floor + 2.0))
    return Gate(voiced=voiced, speech=speech, noise_floor=noise_floor, speech_floor=speech_floor)


def _runs(mask: np.ndarray, hop: float) -> list[tuple[float, float, bool]]:
    """Collapse a boolean mask into (start, end, value) runs."""
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [mask.size]))
    out: list[tuple[float, float, bool]] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        out.append((start * hop, stop * hop, bool(mask[start])))
    return out


def detect_pauses(
    frames: list[Frame],
    gate: Gate,
    hop: float,
    min_silence: float = 0.28,
    segments: list[Segment] | None = None,
) -> list[PauseEvent]:
    """Silence runs long enough to read as a deliberate beat or a speaker change."""
    quiet = ~gate.speech
    events: list[PauseEvent] = []
    for start, end, is_speech in _runs(quiet, hop):
        if is_speech or (end - start) < min_silence:
            continue
        before = ""
        if segments:
            prior = [s for s in segments if s.end <= start + 0.05]
            if prior:
                before = prior[-1].text[-120:]
        events.append(PauseEvent(start=start, end=end, duration=end - start, before=before))
    return events


# ------------------------------------------------------------------ laughter

def _local_modulation(energy: np.ndarray, hop: float, band: tuple[float, float] = (3.0, 8.0)) -> np.ndarray:
    """Per-frame 3-8 Hz amplitude-modulation strength.

    Laughter is strongly modulated at syllabic rate. The spectrum has to be
    computed over sliding windows: a single global spectrum would hand the
    same value to every frame and carry no timing information at all.
    """
    n = energy.size
    out = np.zeros(n, dtype=np.float32)
    if n < 16:
        return out

    win_frames = max(16, int(1.0 / hop))       # 1 s analysis window
    step = max(1, win_frames // 4)
    env = np.convolve(energy, np.ones(5) / 5.0, mode="same")
    env = env - env.mean()

    for start in range(0, max(1, n - win_frames + 1), step):
        stop = min(n, start + win_frames)
        block = env[start:stop]
        if block.size < 8:
            continue
        spectrum = np.abs(np.fft.rfft(block * np.hanning(block.size)))
        freqs = np.fft.rfftfreq(block.size, d=hop)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not mask.any():
            continue
        total = float(spectrum.sum()) + 1e-9
        value = float((spectrum * mask).sum() / total)
        np.maximum(out[start:stop], value, out=out[start:stop])

    # normalise against the recording's own distribution
    positive = out[out > 0]
    if positive.size > 8:
        p50, p95 = np.percentile(positive, 50), np.percentile(positive, 95)
        span = max(p95 - p50, 1e-6)
        out = np.clip((out - p50) / span, 0.0, 1.5)
    return out


def detect_laughter(
    frames: list[Frame],
    gate: Gate,
    hop: float,
    min_len: float = 0.25,
    max_len: float = 3.0,
    threshold: float = 0.55,
) -> list[LaughterEvent]:
    """Score candidate laughter bursts from energy, brightness and pitch chaos.

    Laughter differs from speech by being simultaneously loud, spectrally
    bright, strongly amplitude-modulated (~3-8 Hz) and pitch-unstable. Each cue
    is normalised against the recording's own speech distribution and then
    combined *multiplicatively* with hard minimum gates, so a single strong cue
    (e.g. loud speech) can never on its own register as laughter.
    """
    if len(frames) < 24:
        return []
    n = len(frames)
    energy = np.array([f.energy for f in frames], dtype=np.float32)
    centroid = np.array([f.centroid for f in frames], dtype=np.float32)
    f0 = np.array([f.f0 for f in frames], dtype=np.float32)
    speech = gate.speech

    def robust_norm(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.sum() < 8:
            return np.zeros_like(values)
        ref = values[mask]
        p10, p90 = np.percentile(ref, 10), np.percentile(ref, 90)
        span = max(p90 - p10, 1e-6)
        return np.clip((values - p10) / span, 0.0, 1.5)

    n_energy = robust_norm(energy, speech)
    n_centroid = robust_norm(centroid, speech)

    # Frame-to-frame pitch instability. This must be measured as *excess* over
    # the recording's own speech-typical movement: normalising by the observed
    # distribution (as for energy/centroid) would map ordinary pitch movement
    # onto 1.0, and then every voiced frame looks like laughter.
    voiced = (f0 > 0) & speech
    n_pitch = np.zeros(n, dtype=np.float32)
    if voiced.sum() > 8:
        d = np.abs(np.diff(f0))
        ref = d[voiced[1:] & voiced[:-1]]
        if ref.size > 8:
            typical = float(np.percentile(ref, 25))
            extreme = float(np.percentile(ref, 95))
            floor = 0.80 * typical          # ignore normal speech movement
            span = max(extreme - floor, 1e-6)
            n_pitch[1:] = np.clip((d - floor) / span, 0.0, 1.5)

    modulation = _local_modulation(energy, hop)

    # hard gates: no single loud/animated phrase should qualify
    gate_energy = np.clip((n_energy - 0.45) / 0.55, 0.0, 1.0)
    gate_bright = np.clip((n_centroid - 0.35) / 0.65, 0.0, 1.0)
    gate_chaos = np.clip((n_pitch - 0.30) / 0.70, 0.0, 1.0)
    gate_mod = np.clip((modulation - 0.20) / 0.80, 0.0, 1.0)

    score = (0.30 * gate_energy + 0.20 * gate_bright + 0.28 * gate_chaos + 0.22 * gate_mod)
    score *= (0.25 + 0.75 * gate_chaos)          # pitch chaos is the real discriminator
    score *= np.where(speech, 1.0, 0.0)
    score = np.convolve(score, np.ones(3) / 3.0, mode="same")

    events: list[LaughterEvent] = []
    for start, end, hot in _runs(score > threshold, hop):
        length = end - start
        if length < min_len:
            continue
        if length > max_len:
            # keep the strongest max_len slice of an over-long hot region
            inner = score[int(start / hop): int(end / hop)]
            window = max(1, int(max_len / hop))
            best_start, best = 0, -1.0
            for i in range(0, max(1, inner.size - window)):
                total = float(inner[i: i + window].sum())
                if total > best:
                    best, best_start = total, i
            start = start + best_start * hop
            end = min(end, start + max_len)
        peak = float(score[int(start / hop): max(int(start / hop) + 1, int(end / hop))].max())
        events.append(LaughterEvent(start=start, end=end, strength=min(1.0, peak)))
    return events


# ------------------------------------------------------- speaker turn detection

def _windowed_pitch(
    times: np.ndarray, f0: np.ndarray, window: float, hop: float
) -> tuple[np.ndarray, np.ndarray]:
    """Median F0 over sliding time windows. Returns (centres, pitch)."""
    if times.size == 0:
        return np.zeros(0), np.zeros(0)
    win = max(1, int(window / hop))
    step = max(1, win // 2)
    centres: list[float] = []
    values: list[float] = []
    n = f0.size
    for start in range(0, max(1, n - win + 1), step):
        block = f0[start: start + win]
        voiced = block[block > 0]
        if voiced.size < max(3, win // 8):
            continue
        centres.append(float(times[min(n - 1, start + win // 2)]))
        values.append(float(np.median(voiced)))
    return np.array(centres, dtype=np.float32), np.array(values, dtype=np.float32)


def _assign_speaker_labels(
    turns: list[dict], max_speakers: int = 4
) -> list[dict]:
    """Cluster turn pitches into speakers with 1-D k-means on log F0."""
    pitches = np.array([t["pitch"] for t in turns], dtype=np.float32)
    usable = pitches[pitches > 0]
    if usable.size < 2 or max_speakers < 2:
        for t in turns:
            t["speaker"] = "Speaker A"
        return turns

    log_pitch = np.log(np.maximum(pitches, 1.0))
    k = 2 if max_speakers == 2 else min(max_speakers, max(2, usable.size // 3))
    valid = log_pitch > 0
    if valid.sum() < k * 2:
        for t in turns:
            t["speaker"] = "Speaker A"
        return turns

    data = log_pitch[valid]
    order = np.sort(data)
    # seed centroids at the extremes so a bimodal split is found immediately
    centroids = np.array([order[: max(1, len(order) // 10)].mean(), order[-max(1, len(order) // 10):].mean()])
    for _ in range(40):
        distances = np.abs(data[:, None] - centroids[None, :])
        assign = np.argmin(distances, axis=1)
        new = centroids.copy()
        for i in range(centroids.size):
            members = data[assign == i]
            if members.size:
                new[i] = float(members.mean())
        if np.allclose(new, centroids, atol=1e-4):
            centroids = new
            break
        centroids = new

    # A single dominant cluster means we did not really find two speakers.
    if float(abs(centroids[1] - centroids[0])) < 0.12:      # < ~13% pitch gap
        for t in turns:
            t["speaker"] = "Speaker A"
        return turns

    # label by first appearance so "Speaker A" is whoever talks first
    distances = np.abs(log_pitch[:, None] - centroids[None, :])
    raw = np.argmin(distances, axis=1)
    names: dict[int, str] = {}
    for t, r in zip(turns, raw):
        key = int(r)
        if key not in names:
            names[key] = chr(ord("A") + len(names))
        t["speaker"] = f"Speaker {names[key]}"
    return turns


def segment_speaker_turns(
    words: list,
    frames: list[Frame],
    hop: float,
    min_gap: float = 0.20,
    pitch_shift: float = 0.22,
    window: float = 0.6,
) -> list[SpeakerTurn]:
    """Split a word-level transcript into speaker turns.

    Whisper segments routinely span several speakers, so segmentation happens at
    the *word* level: a new turn starts at a speech gap, or where the windowed
    pitch steps past ``pitch_shift`` in log space *and stays there*. Turn pitches
    are then clustered into speakers, and adjacent turns that end up with the
    same label are merged back together - which is what stops ordinary
    intonation (rising question endings, emphasis) from shredding a single
    speaker into a dozen turns.
    """
    turns: list[SpeakerTurn] = []
    if not words or not frames:
        return turns

    times = np.array([f.t for f in frames], dtype=np.float32)
    f0 = np.array([f.f0 for f in frames], dtype=np.float32)
    centres, pitch = _windowed_pitch(times, f0, window, hop)

    def pitch_at(t: float) -> float:
        if centres.size == 0:
            return 0.0
        idx = int(np.searchsorted(centres, t))
        idx = max(0, min(centres.size - 1, idx))
        return float(pitch[idx])

    current_words: list = []
    current_start = float(words[0].start)
    last_end = float(words[0].end)
    baseline = pitch_at(last_end)

    def flush(end_time: float) -> None:
        nonlocal current_words
        if current_words:
            sample = float(np.median([w.start for w in current_words] + [end_time]))
            turns.append({
                "start": current_start,
                "end": end_time,
                "pitch": pitch_at(sample) or baseline,
                "words": list(current_words),
            })
        current_words = []

    def sustained_shift(t: float, reference: float) -> float:
        """Confirm a pitch step persists at two lookahead points."""
        confirm = 0.0
        for offset in (0.35, 0.8):
            value = pitch_at(t + offset)
            if value <= 0 or reference <= 0:
                continue
            if abs(math.log(value / reference)) >= pitch_shift * 0.7:
                confirm = value
                break
        return confirm

    for word in words:
        gap = word.start - last_end
        if current_words and gap >= min_gap:
            flush(last_end)
            current_start = float(word.start)
            baseline = pitch_at(word.start)
        elif current_words:
            new_pitch = pitch_at(word.start)
            if new_pitch > 0 and baseline > 0:
                if abs(math.log(new_pitch / baseline)) >= pitch_shift:
                    confirm = sustained_shift(word.end, baseline)
                    if confirm > 0:
                        flush(last_end)
                        current_start = float(word.start)
                        baseline = confirm
        current_words.append(word)
        last_end = float(word.end)
    flush(last_end)

    if not turns:
        return []

    # absorb very short turns into their neighbour
    merged: list[dict] = []
    for turn in turns:
        if merged and (turn["end"] - turn["start"]) < 0.45:
            merged[-1]["end"] = turn["end"]
            merged[-1]["words"].extend(turn["words"])
            continue
        merged.append(turn)
    turns = merged

    _assign_speaker_labels(turns)

    # fold back splits that were just intonation, not a speaker change
    collapsed: list[dict] = []
    for turn in turns:
        if collapsed and collapsed[-1]["speaker"] == turn["speaker"]:
            collapsed[-1]["end"] = turn["end"]
            collapsed[-1]["words"].extend(turn["words"])
            continue
        collapsed.append(turn)
    turns = collapsed

    result: list[SpeakerTurn] = []
    prev: SpeakerTurn | None = None
    prev_pitch = 0.0
    for turn in turns:
        start, end = float(turn["start"]), float(turn["end"])
        boundary = 0.0
        confidence = 0.0
        if prev is not None and prev.speaker != turn["speaker"]:
            boundary = start
            if turn["pitch"] > 0 and prev_pitch > 0:
                shift = abs(math.log(turn["pitch"] / prev_pitch))
                confidence = min(1.0, shift / 0.35)
        result.append(SpeakerTurn(start=start, end=end, speaker=turn["speaker"],
                                  boundary=boundary, confidence=confidence))
        prev = result[-1]
        if turn["pitch"] > 0:
            prev_pitch = turn["pitch"]
    return result


def apply_turns_to_segments(segments: list[Segment], turns: list[SpeakerTurn]) -> None:
    """Label each transcript segment with the dominant speaker over its span."""
    if not turns:
        return
    for seg in segments:
        seg.speaker = "Speaker A"
        best_overlap = -1.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                seg.speaker = turn.speaker


def detect_speaker_turns(
    segments: list[Segment],
    frames: list[Frame],
    hop: float,
    min_gap: float = 0.14,
) -> list[SpeakerTurn]:
    """Assign speaker labels by clustering per-segment pitch and loudness.

    Returns turns ordered in time. Labels are ``Speaker A`` / ``Speaker B`` ...
    in order of first appearance.
    """
    if not segments:
        return []

    times = np.array([f.t for f in frames], dtype=np.float32)
    f0 = np.array([f.f0 for f in frames], dtype=np.float32)
    rms = np.array([f.rms_db for f in frames], dtype=np.float32)

    stats: list[tuple[float, float]] = []
    for seg in segments:
        lo = int(np.searchsorted(times, seg.start, "left"))
        hi = int(np.searchsorted(times, seg.end, "right"))
        if hi <= lo:
            lo, hi = int(np.searchsorted(times, max(0.0, seg.start - 0.2), "left")), int(
                np.searchsorted(times, seg.end + 0.2, "right")
            )
        block_f0 = f0[lo:hi]
        voiced = block_f0[block_f0 > 0]
        pitch = float(np.median(voiced)) if voiced.size >= 3 else 0.0
        loud = float(np.median(rms[lo:hi])) if hi > lo else -60.0
        stats.append((pitch, loud))

    known = [(p, l) for p, l in stats if p > 0]
    speaker_count = 2 if len(known) >= 4 else 1
    threshold = _pitch_split_threshold([p for p, _ in known]) if speaker_count > 1 else None

    labels: list[str] = []
    for pitch, loud in stats:
        if speaker_count == 1 or threshold is None or pitch <= 0:
            labels.append("A")
            continue
        if abs(pitch - threshold[0]) <= abs(pitch - threshold[1]):
            labels.append("A" if threshold[0] < threshold[1] else "B")
        else:
            labels.append("B" if threshold[0] < threshold[1] else "A")

    # force a change at long pauses even when the pitch barely moved: a >1.2 s
    # gap in a two-voice conversation is almost always a turn boundary
    for i in range(1, len(segments)):
        if labels[i] == labels[i - 1]:
            gap = segments[i].start - segments[i - 1].end
            if gap > 1.2 and speaker_count > 1:
                labels[i] = "B" if labels[i - 1] == "A" else "A"

    names: dict[str, str] = {}
    order: list[str] = []
    for raw in labels:
        if raw not in names:
            names[raw] = chr(ord("A") + len(order))
            order.append(raw)

    turns: list[SpeakerTurn] = []
    prev_pitch = 0.0
    for i, (seg, raw, (pitch, _loud)) in enumerate(zip(segments, labels, stats)):
        label = f"Speaker {names[raw]}"
        boundary = 0.0
        confidence = 0.0
        if i > 0 and turns[-1].speaker != label:
            boundary = seg.start
            gap = seg.start - segments[i - 1].end
            gap_bonus = min(0.35, max(0.0, gap) / 1.2 * 0.35)
            if pitch > 0 and prev_pitch > 0:
                shift = abs(pitch - prev_pitch) / max(prev_pitch, 1.0)
                confidence = min(1.0, shift * 3.0 + gap_bonus)
            else:
                confidence = gap_bonus
        seg.speaker = label
        turns.append(
            SpeakerTurn(start=seg.start, end=seg.end, speaker=label, boundary=boundary, confidence=confidence)
        )
        if pitch > 0:
            prev_pitch = pitch

    return turns


def _pitch_split_threshold(pitches: list[float]) -> tuple[float, float] | None:
    """1-D two-means clustering on pitch; returns the two centroids."""
    if len(pitches) < 4:
        return None
    values = sorted(pitches)
    low, high = values[0], values[-1]
    if high / max(low, 1e-6) < 1.18:          # not enough spread to split
        return None
    for _ in range(24):
        mid = (low + high) / 2.0
        left = [p for p in values if p <= mid]
        right = [p for p in values if p > mid]
        if not left or not right:
            break
        new_low, new_high = sum(left) / len(left), sum(right) / len(right)
        if abs(new_low - low) < 0.5 and abs(new_high - high) < 0.5:
            low, high = new_low, new_high
            break
        low, high = new_low, new_high
    if abs(high - low) / max(low, 1e-6) < 0.18:
        return None
    return (low, high)


# ------------------------------------------------------------------ loudness

def integrated_lufs(path: str | Path) -> tuple[float, float]:
    """Return (integrated LUFS, true peak dBFS) using ffmpeg's ebur128 filter."""
    from ..render import ffmpeg

    try:
        out = ffmpeg.run([
            "-i", str(path), "-filter_complex", "ebur128=peak=true", "-f", "null", "-",
        ], timeout=1800)
    except ffmpeg.FFmpegError:
        return -23.0, -1.0
    lufs = -23.0
    peak = -1.0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(":")]
        if len(parts) >= 2 and parts[0] == "I":
            try:
                lufs = float(parts[1])
            except ValueError:
                pass
        if len(parts) >= 3 and parts[0] == "Peak":
            try:
                peak = float(parts[2])
            except ValueError:
                pass
    return lufs, peak


# ------------------------------------------------------------------ entrypoint

def analyze(
    wav_path: str | Path,
    segments: list[Segment] | None = None,
    with_pitch: bool = True,
    with_loudness: bool = True,
) -> AudioAnalysis:
    """Run the full acoustic analysis for one media file."""
    signal, sr = load_wav_mono(wav_path)
    hop_s = 0.02
    win_s = 0.04
    frames, noise_floor = frame_features(signal, sr, hop_s, win_s, with_pitch=with_pitch)
    gate = _gate(frames, noise_floor)
    hop = hop_s

    duration = len(signal) / sr if sr else 0.0
    analysis = AudioAnalysis(
        hop=hop,
        window=win_s,
        frames=frames,
        speech_ratio=float(gate.speech.mean()) if frames else 0.0,
    )
    analysis.pauses = detect_pauses(frames, gate, hop, segments=segments)
    analysis.laughter = detect_laughter(frames, gate, hop)
    if with_loudness and duration > 1:
        analysis.lufs, analysis.peak_db = integrated_lufs(wav_path)
    return analysis
