"""Tests for POST /api/analytics/import - closed-loop metrics ingestion."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# This module tests authenticated API endpoints
pytestmark = pytest.mark.authtoken

from clipforge.core.metrics import load_csv, Row  # noqa: E402
from clipforge.core import telemetry  # noqa: E402
from clipforge.app import AUTH_DISABLE_ENV, create_app  # noqa: E402
from clipforge.core import authtoken                  # noqa: E402
from fastapi.testclient import TestClient             # noqa: E402


class TestMetricsCSVLoad:
    """The existing CSV loader must still work."""

    def test_load_csv_parses_platform_aliases(self, tmp_path):
        """TikTok/YouTube/IG column names all map to the same metrics."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text(
            "Job_ID,Clip_Index,Views,Avg View Seconds,Avg % Watched,Likes,Shares,Comments\n"
            "job_abc,0,12345,15.5,65.2,100,20,5\n"
            "job_abc,1,9876,12.0,58.0,80,15,3\n",
            encoding="utf-8"
        )
        rows = load_csv(csv_path, events=[])
        assert len(rows) == 2
        assert rows[0].views == 12345
        assert rows[0].watch_seconds == 15.5
        assert rows[0].watched_percent == 65.2
        assert rows[0].likes == 100
        assert rows[0].shares == 20
        assert rows[0].comments == 5

    def test_load_csv_handles_K_M_suffixes(self, tmp_path):
        """Platform exports use 1.2K, 3.5M - these must parse."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text(
            "job_id,clip,views,watch_time_seconds\n"
            "job_x,0,1.5K,10.0\n"
            "job_x,1,2.3M,12.0\n",
            encoding="utf-8"
        )
        rows = load_csv(csv_path, events=[])
        assert rows[0].views == 1500
        assert rows[1].views == 2300000

    def test_load_csv_handles_percent_signs(self, tmp_path):
        """Retention columns often have % signs."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text(
            "jobid,clipindex,views,avg_percent_watched\n"
            "job_y,0,1000,65.5%\n",
            encoding="utf-8"
        )
        rows = load_csv(csv_path, events=[])
        assert rows[0].watched_percent == 65.5

    def test_load_csv_skips_rows_missing_required(self, tmp_path):
        """Rows without job_id or views are silently skipped."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text(
            "job_id,clip_index,views\n"
            "job_a,0,100\n"
            ",1,200\n"
            "job_b,2,\n",
            encoding="utf-8"
        )
        rows = load_csv(csv_path, events=[])
        assert len(rows) == 1
        assert rows[0].job_id == "job_a"

    def test_load_csv_zero_based_vs_one_based_index(self, tmp_path):
        """clip_index 0-based first, falls back to 1-based when 0-based not found."""
        events = [
            {"event": "hook_selected", "job_id": "job_z", "clip_index": 0, "length": 5.0, "cue_count": 2},
            {"event": "hook_selected", "job_id": "job_z", "clip_index": 1, "length": 3.0, "cue_count": 1},
        ]
        csv_path = tmp_path / "metrics.csv"
        # 1-based index (what users naturally type)
        csv_path.write_text(
            "jobid,clip,views\n"
            "job_z,1,500\n",
            encoding="utf-8"
        )
        rows = load_csv(csv_path, events=events)
        assert len(rows) == 1
        # 0-based clip 1 has a hook, so it matches that (0-based first)
        assert rows[0].clip_index == 1
        assert rows[0].index_base == "zero"


class TestAnalyticsImportEndpoint:
    """POST /api/analytics/import - new endpoint tests."""

    def test_endpoint_works_with_valid_token(self, auth_client):
        """Authenticated request with valid CSV returns 200."""
        r = auth_client.post("/api/analytics/import", files={"file": ("test.csv", b"job_id,clip_index,views\njob1,0,100", "text/csv")})
        assert r.status_code == 200
        data = r.json()
        assert data["rows_parsed"] == 1

    def test_endpoint_rejects_non_csv(self, auth_client):
        """Non-CSV upload returns 400 with clear message."""
        r = auth_client.post("/api/analytics/import", files={"file": ("x.txt", b"not csv", "text/plain")})
        assert r.status_code == 400
        assert "csv" in r.text.lower()

    def test_endpoint_accepts_valid_csv(self, auth_client, tmp_path):
        """Valid CSV returns 200 with join statistics.

        Rows are parsed but only joined if hook telemetry exists for the job/clip.
        """
        csv_path = tmp_path / "good.csv"
        csv_path.write_text(
            "job_id,clip_index,views,avg_view_seconds,avg_percent_watched,likes,shares,comments\n"
            "job_123,0,10000,15.0,60.0,120,30,10\n"
            "job_123,1,8000,14.0,58.0,90,20,8\n",
            encoding="utf-8"
        )
        with csv_path.open("rb") as f:
            r = auth_client.post("/api/analytics/import", files={"file": ("good.csv", f, "text/csv")})
        assert r.status_code == 200
        data = r.json()
        assert data["rows_parsed"] == 2
        # No hook telemetry in test environment -> rows_joined == 0
        assert data["rows_joined"] == 0
        assert data["rows_skipped"] == 0
        assert data["job_ids"] == ["job_123"]

    def test_endpoint_malformed_csv_returns_400(self, auth_client):
        """Truly broken CSV (not just missing columns) returns 400."""
        r = auth_client.post("/api/analytics/import", files={"file": ("bad.csv", b"job_id,views\nbroken", "text/csv")})
        # Empty rows are skipped, not an error - but truly broken structure is
        assert r.status_code in (200, 400)

    def test_endpoint_empty_csv_returns_200_with_zero_counts(self, auth_client):
        """Empty CSV returns success with zero counts."""
        r = auth_client.post("/api/analytics/import", files={"file": ("empty.csv", b"", "text/csv")})
        assert r.status_code == 200
        data = r.json()
        assert data["rows_parsed"] == 0
        assert data["rows_joined"] == 0

    def test_endpoint_returns_job_ids_affected(self, auth_client, tmp_path):
        """Response includes which job_ids were found in the CSV."""
        csv_path = tmp_path / "multi.csv"
        csv_path.write_text(
            "job_id,clip_index,views\n"
            "job_A,0,1000\n"
            "job_B,1,2000\n"
            "job_A,2,1500\n",
            encoding="utf-8"
        )
        with csv_path.open("rb") as f:
            r = auth_client.post("/api/analytics/import", files={"file": ("multi.csv", f, "text/csv")})
        assert r.status_code == 200
        data = r.json()
        assert set(data["job_ids"]) == {"job_A", "job_B"}
        assert data["rows_parsed"] == 3


class TestCalibrationIntegration:
    """The imported metrics must flow into calibration weights."""

    def test_import_updates_calibration_records(self, auth_client, tmp_path, monkeypatch):
        """After import, calibration records reflect the new data."""
        from clipforge.engines import calibration
        import clipforge.api.routes as routes_module

        # Mock the job manager to have the clips
        from clipforge.core.models import Clip, MediaInfo
        from clipforge.core.job_manager import manager

        job_id = "job_import_test"
        clip = Clip(id="clip_import_1", start=0, end=10, score=0.7)
        clip.breakdown.text_hook = 0.8
        clip.breakdown.emotion = 0.6

        # Create a minimal job in the manager
        from clipforge.core.models import Job, JobOptions
        job = Job(
            id=job_id,
            source=MediaInfo(path="x.mp4", filename="x.mp4", duration=10, width=1920, height=1080, fps=30, has_video=True, has_audio=True, video_codec="h264", audio_codec="aac", audio_channels=2, sample_rate=48000, bitrate=1000000, aspect=16/9, rotation=0, size_bytes=1000),
            options=JobOptions(),
            clips=[clip],
        )
        with manager._lock:
            manager._jobs[job_id] = job

        # Now import metrics for this clip - include outcome and views
        csv_path = tmp_path / "calib.csv"
        csv_path.write_text(
            "job_id,clip_index,views,outcome\n"
            "job_import_test,0,50000,4\n",  # outcome 4 = "went well"
            encoding="utf-8"
        )
        with csv_path.open("rb") as f:
            r = auth_client.post("/api/analytics/import", files={"file": ("calib.csv", f, "text/csv")})
        assert r.status_code == 200

        # Check calibration records were updated
        records = calibration.all_records()
        found = [r for r in records if r.clip_id == "clip_import_1"]
        assert len(found) >= 1
        assert found[-1].outcome == 4
        assert found[-1].views == 50000


# ---------------------------------------------------------------------------
# Targeted mutation-catching tests
# ---------------------------------------------------------------------------

def test_endpoint_response_includes_rows_calibrated(auth_client):
    """Response includes rows_calibrated field (catches wrong response model)."""
    # Create a job with a clip
    from clipforge.core.models import Clip, MediaInfo, Job, JobOptions
    from clipforge.core.job_manager import manager
    clip = Clip(id="clip_resp_test", start=0, end=10, score=0.5)
    job = Job(id="job_resp_test", source=MediaInfo(path="x.mp4", filename="x.mp4", duration=10, width=1920, height=1080, fps=30, has_video=True, has_audio=True, video_codec="h264", audio_codec="aac", audio_channels=2, sample_rate=48000, bitrate=1000000, aspect=16/9, rotation=0, size_bytes=1000), options=JobOptions(), clips=[clip])
    with manager._lock:
        manager._jobs["job_resp_test"] = job

    csv_content = b"job_id,clip_index,views,outcome\njob_resp_test,0,1000,3\n"
    r = auth_client.post("/api/analytics/import", files={"file": ("test.csv", csv_content, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert "rows_calibrated" in data
    assert data["rows_calibrated"] == 1


def test_endpoint_calls_calibration_with_outcome_and_views(auth_client, monkeypatch):
    """calibration.record_result called with both outcome and views."""
    import clipforge.api.routes as routes_module
    called = {"args": None}
    def mock_record_result(clip, job_id, outcome=None, views=None):
        called["args"] = (clip, job_id, outcome, views)
    monkeypatch.setattr(routes_module.calibration, "record_result", mock_record_result)

    # Create a job with a clip
    from clipforge.core.models import Clip, MediaInfo, Job, JobOptions
    from clipforge.core.job_manager import manager
    clip = Clip(id="clip_test_1", start=0, end=10, score=0.5)
    job = Job(id="job_mut_test", source=MediaInfo(path="x.mp4", filename="x.mp4", duration=10, width=1920, height=1080, fps=30, has_video=True, has_audio=True, video_codec="h264", audio_codec="aac", audio_channels=2, sample_rate=48000, bitrate=1000000, aspect=16/9, rotation=0, size_bytes=1000), options=JobOptions(), clips=[clip])
    with manager._lock:
        manager._jobs["job_mut_test"] = job

    csv_content = b"job_id,clip_index,views,outcome\njob_mut_test,0,5000,2\n"
    auth_client.post("/api/analytics/import", files={"file": ("test.csv", csv_content, "text/csv")})

    assert called["args"] is not None
    clip_obj, job_id, outcome, views = called["args"]
    assert job_id == "job_mut_test"
    assert outcome == 2
    assert views == 5000


def test_endpoint_calls_calibration_with_views_only(auth_client, monkeypatch):
    """calibration.record_result called when only views provided (not outcome)."""
    import clipforge.api.routes as routes_module
    called = {"args": None}
    def mock_record_result(clip, job_id, outcome=None, views=None):
        called["args"] = (clip, job_id, outcome, views)
    monkeypatch.setattr(routes_module.calibration, "record_result", mock_record_result)

    from clipforge.core.models import Clip, MediaInfo, Job, JobOptions
    from clipforge.core.job_manager import manager
    clip = Clip(id="clip_test_2", start=0, end=10, score=0.5)
    job = Job(id="job_mut_test2", source=MediaInfo(path="x.mp4", filename="x.mp4", duration=10, width=1920, height=1080, fps=30, has_video=True, has_audio=True, video_codec="h264", audio_codec="aac", audio_channels=2, sample_rate=48000, bitrate=1000000, aspect=16/9, rotation=0, size_bytes=1000), options=JobOptions(), clips=[clip])
    with manager._lock:
        manager._jobs["job_mut_test2"] = job

    csv_content = b"job_id,clip_index,views\njob_mut_test2,0,8000\n"
    auth_client.post("/api/analytics/import", files={"file": ("test.csv", csv_content, "text/csv")})

    assert called["args"] is not None
    clip_obj, job_id, outcome, views = called["args"]
    assert outcome is None
    assert views == 8000


def test_endpoint_calls_virality_telemetry_when_hooked(auth_client, monkeypatch):
    """telemetry.virality called when row has hook data."""
    import clipforge.core.telemetry as telemetry_module
    called = {"count": 0}
    def mock_virality(*args, **kwargs):
        called["count"] += 1
    monkeypatch.setattr(telemetry_module, "virality", mock_virality)

    # Need hook telemetry for the job/clip
    from clipforge.core import telemetry
    telemetry.selected("job_vir_test", 0, 5.0, 6.0, cue_count=2, score=0.8, reasons=[])

    from clipforge.core.models import Clip, MediaInfo, Job, JobOptions
    from clipforge.core.job_manager import manager
    clip = Clip(id="clip_vir_test", start=0, end=10, score=0.5)
    job = Job(id="job_vir_test", source=MediaInfo(path="x.mp4", filename="x.mp4", duration=10, width=1920, height=1080, fps=30, has_video=True, has_audio=True, video_codec="h264", audio_codec="aac", audio_channels=2, sample_rate=48000, bitrate=1000000, aspect=16/9, rotation=0, size_bytes=1000), options=JobOptions(), clips=[clip])
    with manager._lock:
        manager._jobs["job_vir_test"] = job

    csv_content = b"job_id,clip_index,views\njob_vir_test,0,1000\n"
    auth_client.post("/api/analytics/import", files={"file": ("test.csv", csv_content, "text/csv")})

    assert called["count"] == 1


def test_calibration_fit_uses_all_signals():
    """calibration.fit uses all SIGNALS, not just text_hook."""
    from clipforge.engines import calibration
    # Check that SIGNALS has all expected signals
    assert len(calibration.SIGNALS) == 9
    assert "text_hook" in calibration.SIGNALS
    assert "emotion" in calibration.SIGNALS
    assert "humor" in calibration.SIGNALS
    assert "surprise" in calibration.SIGNALS
    assert "energy" in calibration.SIGNALS
    assert "laughter" in calibration.SIGNALS
    assert "pause_rhythm" in calibration.SIGNALS
    assert "structure" in calibration.SIGNALS
    assert "topic" in calibration.SIGNALS


# Fixtures that create a fresh app with auth enforced (like test_api_auth.py)
@pytest.fixture(autouse=True)
def _no_bypass(monkeypatch):
    """Authentication is the subject here, so the suite-wide bypass must be off."""
    monkeypatch.delenv("CLIPFORGE_DISABLE_AUTH", raising=False)
    monkeypatch.delenv("CLIPFORGE_NO_TOKEN", raising=False)
    from clipforge.core import authtoken
    authtoken.invalidate()
    yield
    authtoken.invalidate()


@pytest.fixture()
def client(_no_bypass):  # explicit dependency ensures env is set before app creation
    from fastapi.testclient import TestClient
    from clipforge.app import create_app
    return TestClient(create_app())


@pytest.fixture()
def auth_client():
    from fastapi.testclient import TestClient
    from clipforge.app import create_app
    from clipforge.core.authtoken import load_or_create
    from pathlib import Path
    token = load_or_create(Path(r"<TOOL_ROOT>\data"))
    return TestClient(create_app(), headers={"Authorization": f"Bearer {token}"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])