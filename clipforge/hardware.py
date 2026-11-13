"""Universal hardware detection and encoder fallback system.

This module provides comprehensive hardware detection for NVIDIA, AMD, Intel GPUs and CPU fallback,
along with dynamic FFmpeg encoder selection with automatic fallbacks.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Literal, Optional, Tuple

log = logging.getLogger("clipforge")

CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0


class EncoderType(Enum):
    """Supported encoder types with priority order."""
    NVIDIA_NVENC = "h264_nvenc"
    AMD_AMF = "h264_amf"
    Intel_QSV = "h264_qsv"
    MEDIA_FOUNDATION = "h264_mf"
    CPU_X264 = "libx264"


@dataclass
class HardwareInfo:
    """Comprehensive hardware information."""
    gpu_name: str = "CPU Mode"
    gpu_vendor: str = "cpu"
    compute_label: str = "cpu · int8 (Multi-Core)"
    video_encoder: str = "libx264"
    encoder_type: EncoderType = EncoderType.CPU_X264
    vram_mb: int = 0
    cuda_available: bool = False
    encoder_available: bool = True
    encoder_params: dict = field(default_factory=dict)


@dataclass
class EncoderTest:
    """Encoder test result."""
    encoder_type: EncoderType
    available: bool
    error_message: Optional[str] = None


class HardwareDetector:
    """Universal hardware detection and encoder management."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._cached_info: Optional[HardwareInfo] = None
        self._encoder_tests: dict[EncoderType, EncoderTest] = {}
        self._gpu_cache: Optional[Tuple[str, str, int]] = None
        # NVENC (and friends) reject tiny inputs like 256x256 and a list of
        # flags older builds tolerated, so the test uses the smallest size the
        # hardware encoders accept and no tuning flags at all.
        self._test_commands = {
            EncoderType.NVIDIA_NVENC: [
                "ffmpeg", "-hide_banner", "-f", "lavfi",
                "-i", "color=c=black:s=640x360:r=30:d=0.1",
                "-frames:v", "1", "-c:v", "h264_nvenc",
                "-pix_fmt", "yuv420p", "-f", "null", "-"
            ],
            EncoderType.AMD_AMF: [
                "ffmpeg", "-hide_banner", "-f", "lavfi",
                "-i", "color=c=black:s=640x360:r=30:d=0.1",
                "-frames:v", "1", "-c:v", "h264_amf",
                "-pix_fmt", "yuv420p", "-f", "null", "-"
            ],
            EncoderType.Intel_QSV: [
                "ffmpeg", "-hide_banner", "-f", "lavfi",
                "-i", "color=c=black:s=640x360:r=30:d=0.1",
                "-frames:v", "1", "-c:v", "h264_qsv",
                "-pix_fmt", "yuv420p", "-f", "null", "-"
            ],
            EncoderType.MEDIA_FOUNDATION: [
                "ffmpeg", "-hide_banner", "-f", "lavfi",
                "-i", "color=c=black:s=640x360:r=30:d=0.1",
                "-frames:v", "1", "-c:v", "h264_mf",
                "-pix_fmt", "yuv420p", "-f", "null", "-"
            ],
            EncoderType.CPU_X264: [
                "ffmpeg", "-hide_banner", "-f", "lavfi",
                "-i", "color=c=black:s=640x360:r=30:d=0.1",
                "-frames:v", "1", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-f", "null", "-"
            ]
        }
        
        self._encoder_params = {
            EncoderType.NVIDIA_NVENC: {
                "preset": "p4",
                "rc": "vbr",
                "cq": "20",
                "bf": "2",
                "g": "60",
                "pix_fmt": "yuv420p"
            },
            EncoderType.AMD_AMF: {
                "quality": "quality",
                "rc": "cqp",
                "qp_i": "20",
                "qp_p": "20",
                "pix_fmt": "yuv420p"
            },
            EncoderType.Intel_QSV: {
                "preset": "medium",
                "global_quality": "20",
                "look_ahead": "1",
                "pix_fmt": "yuv420p"
            },
            EncoderType.MEDIA_FOUNDATION: {
                "rate_control": "quality",
                "quality": "80",
                "pix_fmt": "yuv420p"
            },
            EncoderType.CPU_X264: {
                "preset": "veryfast",
                "crf": "20",
                "bf": "2",
                "g": "60",
                "pix_fmt": "yuv420p"
            }
        }
    
    def _run_command(self, cmd: list[str], timeout: int = 15) -> Tuple[bool, str]:
        """Run a command and return (success, error_message)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=CREATE_NO_WINDOW,
                check=True
            )
            return True, ""
        except subprocess.SubprocessError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    def _gpu_candidates(self) -> list[tuple[str, str]]:
        """(vendor, name) pairs for every display adapter, best first.

        ``nvidia-smi`` is tried first because it is authoritative, works without
        admin, and names the GPU even when Windows also reports an integrated
        Intel adapter. ``wmic`` is gone on recent Windows 11 builds, so the
        general listing comes from PowerShell's CIM cmdlets instead.
        """
        found: list[tuple[str, str]] = []

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
                creationflags=CREATE_NO_WINDOW,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    name = line.strip().strip('"')
                    if name:
                        found.append(("nvidia", name))
        except Exception as exc:  # noqa: BLE001
            log.debug("nvidia-smi probe failed: %s", exc)

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=15,
                creationflags=CREATE_NO_WINDOW,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    name = line.strip()
                    if not name:
                        continue
                    low = name.lower()
                    if "nvidia" in low:
                        vendor = "nvidia"
                    elif "amd" in low or "radeon" in low:
                        vendor = "amd"
                    elif "intel" in low:
                        vendor = "intel"
                    else:
                        vendor = "cpu"
                    pair = (vendor, name)
                    if pair not in found:
                        found.append(pair)
        except Exception as exc:  # noqa: BLE001
            log.debug("Get-CimInstance probe failed: %s", exc)

        # NVIDIA first (NVENC + CUDA), then AMD, then Intel, then unknown.
        order = {"nvidia": 0, "amd": 1, "intel": 2, "cpu": 3}
        found.sort(key=lambda pair: order.get(pair[0], 4))
        return found

    def _detect_gpu(self) -> Tuple[str, str, int]:
        """Detect GPU name, vendor, and VRAM amount."""
        if self._gpu_cache is not None:
            return self._gpu_cache

        gpu_name = "CPU Mode"
        gpu_vendor = "cpu"
        vram_mb = 0

        try:
            candidates = self._gpu_candidates()
            for vendor, name in candidates:
                if vendor in ("nvidia", "amd", "intel"):
                    gpu_name, gpu_vendor = name, vendor
                    break

            # VRAM only means something on NVIDIA; nvidia-smi is the only source.
            if gpu_vendor == "nvidia":
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.total",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        creationflags=CREATE_NO_WINDOW,
                        check=True
                    )
                    vram_mb = int(result.stdout.strip().splitlines()[0].strip())
                except Exception as exc:  # noqa: BLE001
                    log.debug("nvidia-smi memory query failed: %s", exc)

            log.info(f"Detected GPU: {gpu_name} ({gpu_vendor}, {vram_mb}MB)")

        except Exception as e:
            log.debug(f"GPU detection failed: {e}")

        self._gpu_cache = (gpu_name, gpu_vendor, vram_mb)
        return self._gpu_cache
    
    def _detect_cuda(self) -> bool:
        """Detect if CUDA is available.

        NVIDIA hardware alone is not enough — transcription runs through
        CTranslate2, so the card only counts for ASR when CT2 can see it.
        ``ctranslate2.get_cuda_device_count()`` is the authoritative check;
        ``nvidia-smi`` answering is the fallback when CT2 cannot import.
        """
        try:
            # First check if we have an NVIDIA GPU at all
            gpu_name, gpu_vendor, vram_mb = self._detect_gpu()
            if gpu_vendor != "nvidia":
                log.debug("No NVIDIA GPU detected, CUDA not available")
                return False

            try:
                import ctranslate2
                count = ctranslate2.get_cuda_device_count()
                log.debug("ctranslate2 reports %d CUDA device(s)", count)
                if count > 0:
                    return True
            except Exception as exc:  # noqa: BLE001
                log.debug("ctranslate2 CUDA check failed: %s", exc)

            # Fallback: nvidia-smi answering means the driver can see the GPU
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=10,
                creationflags=CREATE_NO_WINDOW,
            )
            return result.returncode == 0

        except Exception as e:
            log.debug(f"CUDA detection failed: {e}")

        return False
    
    def _test_encoder(self, encoder_type: EncoderType) -> EncoderTest:
        """Test if an encoder is available by running a short test."""
        if encoder_type in self._encoder_tests:
            return self._encoder_tests[encoder_type]
        
        cmd = self._test_commands[encoder_type]
        success, error = self._run_command(cmd)
        
        test = EncoderTest(
            encoder_type=encoder_type,
            available=success,
            error_message=error if not success else None
        )
        
        self._encoder_tests[encoder_type] = test
        return test
    
    def _get_best_encoder(self) -> Tuple[EncoderType, bool]:
        """Get the best available encoder in priority order."""
        gpu_name, gpu_vendor, vram_mb = self._detect_gpu()
        
        # Priority order based on GPU vendor
        if gpu_vendor == "nvidia":
            priority_order = [
                EncoderType.NVIDIA_NVENC,
                EncoderType.AMD_AMF,
                EncoderType.Intel_QSV,
                EncoderType.MEDIA_FOUNDATION,
                EncoderType.CPU_X264
            ]
        elif gpu_vendor == "amd":
            priority_order = [
                EncoderType.AMD_AMF,
                EncoderType.Intel_QSV,
                EncoderType.NVIDIA_NVENC,
                EncoderType.MEDIA_FOUNDATION,
                EncoderType.CPU_X264
            ]
        elif gpu_vendor == "intel":
            priority_order = [
                EncoderType.Intel_QSV,
                EncoderType.AMD_AMF,
                EncoderType.NVIDIA_NVENC,
                EncoderType.MEDIA_FOUNDATION,
                EncoderType.CPU_X264
            ]
        else:  # CPU or unknown
            # libx264 first: it is the only encoder that behaves identically
            # everywhere and is the expected CPU fallback. h264_mf stays as a
            # fallback for machines where libx264 is missing.
            priority_order = [
                EncoderType.CPU_X264,
                EncoderType.MEDIA_FOUNDATION,
            ]
        
        for encoder_type in priority_order:
            test = self._test_encoder(encoder_type)
            if test.available:
                return encoder_type, True
        
        # All hardware encoders failed, fall back to CPU
        return EncoderType.CPU_X264, True
    
    def get_hardware_info(self) -> HardwareInfo:
        """Get comprehensive hardware information."""
        with self._lock:
            if self._cached_info:
                return self._cached_info
            
            # Detect GPU
            gpu_name, gpu_vendor, vram_mb = self._detect_gpu()
            
            # Detect CUDA availability
            cuda_available = self._detect_cuda()
            
            # Get best encoder
            best_encoder, encoder_available = self._get_best_encoder()
            
            # Determine compute label
            if gpu_vendor == "nvidia" and cuda_available:
                compute_label = f"cuda · float16"
            elif gpu_vendor == "amd":
                compute_label = f"gpu · int8 (Multi-Core)"
            elif gpu_vendor == "intel":
                compute_label = f"gpu · int8 (Multi-Core)"
            else:
                compute_label = f"cpu · int8 (Multi-Core)"
            
            # Add GPU info to compute label if available
            if gpu_vendor != "cpu":
                compute_label += f" ({gpu_name})"
            
            # Create hardware info
            hardware_info = HardwareInfo(
                gpu_name=gpu_name,
                gpu_vendor=gpu_vendor,
                compute_label=compute_label,
                video_encoder=best_encoder.value,
                encoder_type=best_encoder,
                vram_mb=vram_mb,
                cuda_available=cuda_available,
                encoder_available=encoder_available,
                encoder_params=self._encoder_params[best_encoder]
            )
            
            self._cached_info = hardware_info
            return hardware_info
    
    def get_encoder_command(self, encoder_type: Optional[EncoderType] = None) -> list[str]:
        """Get FFmpeg command for the specified encoder type."""
        if encoder_type is None:
            hardware_info = self.get_hardware_info()
            encoder_type = hardware_info.encoder_type
        
        params = self._encoder_params[encoder_type]
        # Encoder-only arguments; ffmpeg itself is prepended by the caller, so
        # the list must not start with the literal "ffmpeg".
        cmd = ["-c:v", encoder_type.value]
        
        for key, value in params.items():
            if key != "pix_fmt":  # pix_fmt goes last
                cmd.extend([f"-{key}", str(value)])
        
        cmd.extend(["-pix_fmt", params["pix_fmt"]])
        return cmd
    
    def get_encoder_fallback_chain(self) -> list[EncoderType]:
        """Get the fallback chain for encoders."""
        return [
            EncoderType.NVIDIA_NVENC,
            EncoderType.AMD_AMF,
            EncoderType.Intel_QSV,
            EncoderType.MEDIA_FOUNDATION,
            EncoderType.CPU_X264
        ]


# Global hardware detector instance
_hardware_detector = HardwareDetector()


def get_hardware_info() -> HardwareInfo:
    """Get global hardware information."""
    return _hardware_detector.get_hardware_info()


def refresh_hardware_info() -> HardwareInfo:
    """Forget the cached probe and run it again.

    Backs the UI's "Refresh Hardware Benchmark" action: the cached result is
    what the header and the options panel display, so without dropping it a
    driver update or a newly connected GPU would never be picked up until the
    next restart.
    """
    detector = _hardware_detector
    with detector._lock:
        detector._cached_info = None
        detector._gpu_cache = None
        detector._encoder_tests.clear()
    try:
        from . import gpu
        gpu.profile.cache_clear()
    except Exception:                                      # noqa: BLE001
        pass
    return detector.get_hardware_info()


def get_encoder_command(encoder_type: Optional[EncoderType] = None) -> list[str]:
    """Get FFmpeg command for the specified encoder type."""
    return _hardware_detector.get_encoder_command(encoder_type)


def get_encoder_fallback_chain() -> list[EncoderType]:
    """Get the fallback chain for encoders."""
    return _hardware_detector.get_encoder_fallback_chain()


def test_encoder_safely(encoder_type: EncoderType) -> bool:
    """Safely test an encoder without affecting the main cache."""
    return _hardware_detector._test_encoder(encoder_type).available