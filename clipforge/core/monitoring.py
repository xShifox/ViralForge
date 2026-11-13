"""Monitoring telemetry system for clip processing."""

from __future__ import annotations
import logging
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..core.models import Clip

log = logging.getLogger("clipforge")


class EventType(Enum):
    """Types of monitoring events."""
    HOOK_ENABLED = "hook_enabled"
    HOOK_DISABLED = "hook_disabled"
    HOOK_SELECTED = "hook_selected"
    HOOK_DECLINED = "hook_declined"
    HOOK_SCORE = "hook_score"
    COMPLIANCE_RESULT = "compliance_result"
    PROCESSING_START = "processing_start"
    PROCESSING_END = "processing_end"
    ERROR = "error"


@dataclass
class TelemetryEvent:
    """A telemetry event with structured data."""
    event_type: EventType
    timestamp: float
    clip_id: str
    job_id: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "clip_id": self.clip_id,
            "job_id": self.job_id,
            "data": self.data
        }


@dataclass
class ComplianceResult:
    """Compliance check results."""
    passed: bool
    checks: Dict[str, bool]
    score: float
    issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "score": self.score,
            "issues": self.issues
        }


class MonitoringTelemetry:
    """Monitoring telemetry system for tracking clip processing."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.events: List[TelemetryEvent] = []
        self._lock = threading.Lock()
        self.session_id = f"session_{int(time.time())}_{threading.get_ident()}"
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize event log file
        self.event_log_file = self.log_dir / f"telemetry_{self.session_id}.jsonl"
        
    def log_event(self, event_type: EventType, clip_id: str, job_id: str, **data) -> None:
        """Log a telemetry event."""
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            clip_id=clip_id,
            job_id=job_id,
            data=data
        )
        
        with self._lock:
            self.events.append(event)
        self._write_event_to_log(event)
        
        log.debug(f"Logged telemetry event: {event_type.value} for clip {clip_id}")
    
    def _write_event_to_log(self, event: TelemetryEvent) -> None:
        """Write event to log file."""
        try:
            with self._lock:
                with open(self.event_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            log.error(f"Failed to write telemetry event: {e}")
    
    def log_hook_enabled(self, clip_id: str, job_id: str, hook_config: Dict[str, Any]) -> None:
        """Log when hook is enabled for a clip."""
        self.log_event(
            EventType.HOOK_ENABLED,
            clip_id,
            job_id,
            hook_config=hook_config
        )
    
    def log_hook_disabled(self, clip_id: str, job_id: str, reason: str) -> None:
        """Log when hook is disabled for a clip."""
        self.log_event(
            EventType.HOOK_DISABLED,
            clip_id,
            job_id,
            reason=reason
        )
    
    def log_hook_selected(self, clip_id: str, job_id: str, hook_data: Dict[str, Any]) -> None:
        """Log when a hook is selected for a clip."""
        self.log_event(
            EventType.HOOK_SELECTED,
            clip_id,
            job_id,
            hook_data=hook_data
        )
    
    def log_hook_declined(self, clip_id: str, job_id: str, reason: str) -> None:
        """Log when a hook is declined for a clip."""
        self.log_event(
            EventType.HOOK_DECLINED,
            clip_id,
            job_id,
            reason=reason
        )
    
    def log_hook_score(self, clip_id: str, job_id: str, score: float, breakdown: Dict[str, float]) -> None:
        """Log hook scoring data."""
        self.log_event(
            EventType.HOOK_SCORE,
            clip_id,
            job_id,
            score=score,
            breakdown=breakdown
        )
    
    def log_compliance_result(self, clip_id: str, job_id: str, result: ComplianceResult) -> None:
        """Log compliance check results."""
        self.log_event(
            EventType.COMPLIANCE_RESULT,
            clip_id,
            job_id,
            compliance=result.to_dict()
        )
    
    def log_processing_start(self, clip_id: str, job_id: str, platform: str) -> None:
        """Log processing start."""
        self.log_event(
            EventType.PROCESSING_START,
            clip_id,
            job_id,
            platform=platform,
            start_time=time.time()
        )
    
    def log_processing_end(self, clip_id: str, job_id: str, duration: float, success: bool) -> None:
        """Log processing end."""
        self.log_event(
            EventType.PROCESSING_END,
            clip_id,
            job_id,
            duration=duration,
            success=success,
            end_time=time.time()
        )
    
    def log_error(self, clip_id: str, job_id: str, error_type: str, error_message: str) -> None:
        """Log error event."""
        self.log_event(
            EventType.ERROR,
            clip_id,
            job_id,
            error_type=error_type,
            error_message=error_message
        )
    
    def get_events_for_clip(self, clip_id: str) -> List[TelemetryEvent]:
        """Get all events for a specific clip."""
        with self._lock:
            return [event for event in self.events if event.clip_id == clip_id]
    
    def get_events_for_job(self, job_id: str) -> List[TelemetryEvent]:
        """Get all events for a specific job."""
        with self._lock:
            return [event for event in self.events if event.job_id == job_id]
    
    def get_hook_statistics(self, job_id: str) -> Dict[str, Any]:
        """Get hook-related statistics for a job."""
        events = self.get_events_for_job(job_id)
        
        hook_enabled = 0
        hook_disabled = 0
        hook_selected = 0
        hook_declined = 0
        hook_scores = []
        
        for event in events:
            if event.event_type == EventType.HOOK_ENABLED:
                hook_enabled += 1
            elif event.event_type == EventType.HOOK_DISABLED:
                hook_disabled += 1
            elif event.event_type == EventType.HOOK_SELECTED:
                hook_selected += 1
            elif event.event_type == EventType.HOOK_DECLINED:
                hook_declined += 1
            elif event.event_type == EventType.HOOK_SCORE:
                hook_scores.append(event.data.get("score", 0))
        
        avg_hook_score = sum(hook_scores) / len(hook_scores) if hook_scores else 0
        
        return {
            "hook_enabled": hook_enabled,
            "hook_disabled": hook_disabled,
            "hook_selected": hook_selected,
            "hook_declined": hook_declined,
            "average_hook_score": avg_hook_score,
            "total_hooks": hook_enabled + hook_disabled
        }
    
    def get_compliance_summary(self, job_id: str) -> Dict[str, Any]:
        """Get compliance summary for a job."""
        events = self.get_events_for_job(job_id)
        
        compliance_events = [
            event for event in events 
            if event.event_type == EventType.COMPLIANCE_RESULT
        ]
        
        if not compliance_events:
            return {"total_clips": 0, "passed": 0, "failed": 0, "average_score": 0}
        
        passed = 0
        failed = 0
        total_score = 0
        
        for event in compliance_events:
            compliance_data = event.data.get("compliance", {})
            if compliance_data.get("passed", False):
                passed += 1
            else:
                failed += 1
            
            total_score += compliance_data.get("score", 0)
        
        total_clips = len(compliance_events)
        average_score = total_score / total_clips if total_clips > 0 else 0
        
        return {
            "total_clips": total_clips,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total_clips if total_clips > 0 else 0,
            "average_score": average_score
        }
    
    def export_telemetry_report(self, job_id: str, output_path: Path) -> None:
        """Export telemetry report to file."""
        report = {
            "job_id": job_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "hook_statistics": self.get_hook_statistics(job_id),
            "compliance_summary": self.get_compliance_summary(job_id),
            "total_events": len(self.get_events_for_job(job_id))
        }
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            log.info(f"Telemetry report exported to: {output_path}")
        except Exception as e:
            log.error(f"Failed to export telemetry report: {e}")
    
    def clear_session(self) -> None:
        """Clear current session data."""
        with self._lock:
            self.events.clear()
        self.session_id = f"session_{int(time.time())}"
        self.event_log_file = self.log_dir / f"telemetry_{self.session_id}.jsonl"


# Global telemetry instance
_telemetry_instance: Optional[MonitoringTelemetry] = None
_telemetry_lock = threading.Lock()


def get_telemetry() -> MonitoringTelemetry:
    """Get the global telemetry instance."""
    global _telemetry_instance
    if _telemetry_instance is None:
        with _telemetry_lock:
            if _telemetry_instance is None:
                from ..config import DATA_DIR

                _telemetry_instance = MonitoringTelemetry(DATA_DIR / "logs")
    return _telemetry_instance


def initialize_telemetry(log_dir: Path) -> MonitoringTelemetry:
    """Initialize the global telemetry instance."""
    global _telemetry_instance
    with _telemetry_lock:
        _telemetry_instance = MonitoringTelemetry(log_dir)
    return _telemetry_instance


def log_telemetry_event(event_type: EventType, clip_id: str, job_id: str, **data) -> None:
    """Convenience function to log telemetry events."""
    telemetry = get_telemetry()
    telemetry.log_event(event_type, clip_id, job_id, **data)


def create_compliance_result(clip: Clip, checks: Dict[str, bool], 
                           score: float, issues: List[str]) -> ComplianceResult:
    """Create a compliance result for a clip."""
    return ComplianceResult(
        passed=all(checks.values()),
        checks=checks,
        score=score,
        issues=issues
    )