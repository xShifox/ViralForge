"""Weekly analytics API endpoints."""

from __future__ import annotations
import logging
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel

from ..core.monitoring import get_telemetry, ComplianceResult
from ..core.models import Clip

log = logging.getLogger("clipforge")

router = APIRouter()


class AnalyticsImportRequest(BaseModel):
    """Request for importing analytics data."""
    platform: str  # "tiktok", "youtube", "instagram"
    data: List[Dict[str, Any]]


class AnalyticsSummary(BaseModel):
    """Analytics summary response."""
    platform: str
    total_clips: int
    hooked_clips: int
    non_hooked_clips: int
    hooked_performance: Dict[str, float]
    non_hooked_performance: Dict[str, float]
    improvement_metrics: Dict[str, float]
    period: str
    generated_at: str


class ClipAnalytics(BaseModel):
    """Individual clip analytics data."""
    clip_id: str
    platform: str
    title: str
    hooked: bool
    views: int
    avg_watch_time: float
    percent_watched: float
    likes: int
    shares: int
    comments: int
    engagement_rate: float
    performance_score: float


def _zero_performance() -> Dict[str, float]:
    """All-zero metric bundle for a hooked / non-hooked group with no data."""
    return {
        "avg_views": 0.0,
        "avg_watch_time": 0.0,
        "avg_percent_watched": 0.0,
        "avg_likes": 0.0,
        "avg_shares": 0.0,
        "avg_comments": 0.0,
        "avg_engagement_rate": 0.0,
        "avg_performance_score": 0.0,
    }


def _empty_summary(platform: Optional[str], period: str) -> AnalyticsSummary:
    """200 summary with zero metrics when no analytics data exists (Defect 4).

    The dashboard polls this endpoint before anything has been imported; a
    404/500 there is a new-install failure, not a missing record, so the
    honest equivalent of "nothing imported yet" is a zeroed bundle.
    """
    return AnalyticsSummary(
        platform=platform or "all",
        total_clips=0,
        hooked_clips=0,
        non_hooked_clips=0,
        hooked_performance=_zero_performance(),
        non_hooked_performance=_zero_performance(),
        improvement_metrics={
            "views_improvement": 0.0,
            "watch_time_improvement": 0.0,
            "percent_watched_improvement": 0.0,
            "engagement_improvement": 0.0,
            "performance_score_improvement": 0.0,
        },
        period=period,
        generated_at=datetime.now().isoformat(),
    )


def parse_tiktok_csv(file_content: str) -> List[Dict[str, Any]]:
    """Parse TikTok analytics CSV format."""
    clips = []
    reader = csv.DictReader(file_content.splitlines())
    
    for row in reader:
        try:
            clip = {
                "clip_id": row.get("Video ID", ""),
                "title": row.get("Video title", ""),
                "views": int(row.get("Views", 0)),
                "avg_watch_time": float(row.get("Average watch time", 0)) / 1000,  # Convert ms to s
                "percent_watched": float(row.get("Average view percentage", 0)),
                "likes": int(row.get("Likes", 0)),
                "shares": int(row.get("Shares", 0)),
                "comments": int(row.get("Comments", 0)),
                "platform": "tiktok"
            }
            
            # Calculate engagement rate
            engagement = (clip["likes"] + clip["shares"] + clip["comments"]) / max(clip["views"], 1)
            clip["engagement_rate"] = engagement
            
            # Calculate performance score (simple weighted sum)
            clip["performance_score"] = (
                clip["views"] * 0.3 +
                clip["avg_watch_time"] * 10 * 0.2 +
                clip["percent_watched"] * 0.2 +
                clip["engagement_rate"] * 1000 * 0.3
            )
            
            clips.append(clip)
        except (ValueError, KeyError) as e:
            log.warning(f"Failed to parse TikTok CSV row: {e}")
    
    return clips


def parse_youtube_csv(file_content: str) -> List[Dict[str, Any]]:
    """Parse YouTube Shorts analytics CSV format."""
    clips = []
    reader = csv.DictReader(file_content.splitlines())
    
    for row in reader:
        try:
            clip = {
                "clip_id": row.get("Video ID", ""),
                "title": row.get("Video title", ""),
                "views": int(row.get("Views", 0)),
                "avg_watch_time": float(row.get("Average percentage viewed", 0)) / 100,  # Convert % to fraction
                "percent_watched": float(row.get("Average percentage viewed", 0)),
                "likes": int(row.get("Likes", 0)),
                "shares": int(row.get("Shares", 0)),
                "comments": int(row.get("Comments", 0)),
                "platform": "youtube"
            }
            
            # Calculate engagement rate
            engagement = (clip["likes"] + clip["shares"] + clip["comments"]) / max(clip["views"], 1)
            clip["engagement_rate"] = engagement
            
            # Calculate performance score
            clip["performance_score"] = (
                clip["views"] * 0.3 +
                clip["avg_watch_time"] * 60 * 0.2 +  # Convert to seconds
                clip["percent_watched"] * 0.2 +
                clip["engagement_rate"] * 1000 * 0.3
            )
            
            clips.append(clip)
        except (ValueError, KeyError) as e:
            log.warning(f"Failed to parse YouTube CSV row: {e}")
    
    return clips


def parse_instagram_csv(file_content: str) -> List[Dict[str, Any]]:
    """Parse Instagram Reels analytics CSV format."""
    clips = []
    reader = csv.DictReader(file_content.splitlines())
    
    for row in reader:
        try:
            clip = {
                "clip_id": row.get("Media ID", ""),
                "title": row.get("Media title", ""),
                "views": int(row.get("Impressions", 0)),
                "avg_watch_time": float(row.get("Average watch time", 0)) / 1000,  # Convert ms to s
                "percent_watched": float(row.get("Average view percentage", 0)),
                "likes": int(row.get("Likes", 0)),
                "shares": int(row.get("Shares", 0)),
                "comments": int(row.get("Comments", 0)),
                "platform": "instagram"
            }
            
            # Calculate engagement rate
            engagement = (clip["likes"] + clip["shares"] + clip["comments"]) / max(clip["views"], 1)
            clip["engagement_rate"] = engagement
            
            # Calculate performance score
            clip["performance_score"] = (
                clip["views"] * 0.3 +
                clip["avg_watch_time"] * 10 * 0.2 +
                clip["percent_watched"] * 0.2 +
                clip["engagement_rate"] * 1000 * 0.3
            )
            
            clips.append(clip)
        except (ValueError, KeyError) as e:
            log.warning(f"Failed to parse Instagram CSV row: {e}")
    
    return clips


@router.post("/import")
async def import_analytics(
    platform: Optional[str] = None,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Import a metrics CSV and close the loop into calibration + telemetry."""
    from ..core import metrics as _metrics_module
    from ..core import telemetry as _telemetry_module
    from ..engines import calibration as _calibration
    from ..core.job_manager import manager

    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="only a csv file can be imported")
    content = await file.read()
    tmp = Path("data") / "analytics" / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(content)
        try:
            rows = _metrics_module.load_csv(tmp)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not parse the csv file: {exc}")
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read the csv file: {exc}")

    job_ids = sorted({r.job_id for r in rows})
    joined = sum(1 for r in rows if r.hooked)
    calibrated = 0
    for row in rows:
        try:
            job = manager._jobs.get(row.job_id)
        except Exception:
            job = None
        if job is None:
            continue
        clip = None
        for candidate in (job.clips or []):
            if candidate.index == row.clip_index:
                clip = candidate
                break
        if clip is None and job.clips:
            by_position = [c for c in job.clips if c.index == row.clip_index]
            if by_position:
                clip = by_position[0]
        if clip is None:
            continue
        try:
            _calibration.record_result(clip, row.job_id, outcome=row.outcome,
                                       views=int(row.views) if row.views is not None else None)
            calibrated += 1
        except Exception:
            continue
        if row.hooked:
            try:
                _telemetry_module.virality(row.job_id, row.clip_index, views=row.views)
            except Exception:
                pass
    return {
        "rows_parsed": len(rows),
        "rows_joined": joined,
        "rows_skipped": 0,
        "job_ids": job_ids,
        "rows_calibrated": calibrated,
    }


@router.get("/summary")
async def get_analytics_summary(
    platform: Optional[str] = None,
    period: str = "latest"
) -> AnalyticsSummary:
    """Get analytics summary for hooked vs non-hooked performance."""
    analytics_dir = Path("data/analytics")
    if not analytics_dir.exists():
        return _empty_summary(platform, period)
    files = list(analytics_dir.glob(f"{platform or '*'}_analytics_*.json"))
    if not files:
        return _empty_summary(platform, period)
    try:
        # Use the most recent file
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, "r", encoding="utf-8") as f:
            clips = json.load(f)
        
        # Separate hooked and non-hooked clips
        hooked_clips = []
        non_hooked_clips = []
        
        for clip in clips:
            # Determine if clip is hooked (check if title contains hook indicators)
            hooked = any(indicator in clip.get("title", "").lower() 
                         for indicator in ["hook", "intro", "hooked", "intro"])
            
            if hooked:
                hooked_clips.append(clip)
            else:
                non_hooked_clips.append(clip)
        
        # Calculate performance metrics
        def calculate_performance_metrics(group_clips):
            if not group_clips:
                return {
                    "avg_views": 0,
                    "avg_watch_time": 0,
                    "avg_percent_watched": 0,
                    "avg_likes": 0,
                    "avg_shares": 0,
                    "avg_comments": 0,
                    "avg_engagement_rate": 0,
                    "avg_performance_score": 0
                }
            
            return {
                "avg_views": sum(c["views"] for c in group_clips) / len(group_clips),
                "avg_watch_time": sum(c["avg_watch_time"] for c in group_clips) / len(group_clips),
                "avg_percent_watched": sum(c["percent_watched"] for c in group_clips) / len(group_clips),
                "avg_likes": sum(c["likes"] for c in group_clips) / len(group_clips),
                "avg_shares": sum(c["shares"] for c in group_clips) / len(group_clips),
                "avg_comments": sum(c["comments"] for c in group_clips) / len(group_clips),
                "avg_engagement_rate": sum(c["engagement_rate"] for c in group_clips) / len(group_clips),
                "avg_performance_score": sum(c["performance_score"] for c in group_clips) / len(group_clips)
            }
        
        hooked_performance = calculate_performance_metrics(hooked_clips)
        non_hooked_performance = calculate_performance_metrics(non_hooked_clips)
        
        # Calculate improvement metrics
        def calculate_improvement(hooked_val, non_hooked_val):
            if non_hooked_val == 0:
                return 0 if hooked_val == 0 else float('inf')
            return ((hooked_val - non_hooked_val) / non_hooked_val) * 100
        
        improvement_metrics = {
            "views_improvement": calculate_improvement(
                hooked_performance["avg_views"], 
                non_hooked_performance["avg_views"]
            ),
            "watch_time_improvement": calculate_improvement(
                hooked_performance["avg_watch_time"], 
                non_hooked_performance["avg_watch_time"]
            ),
            "percent_watched_improvement": calculate_improvement(
                hooked_performance["avg_percent_watched"], 
                non_hooked_performance["avg_percent_watched"]
            ),
            "engagement_improvement": calculate_improvement(
                hooked_performance["avg_engagement_rate"], 
                non_hooked_performance["avg_engagement_rate"]
            ),
            "performance_score_improvement": calculate_improvement(
                hooked_performance["avg_performance_score"], 
                non_hooked_performance["avg_performance_score"]
            )
        }
        
        return AnalyticsSummary(
            platform=platform or "all",
            total_clips=len(clips),
            hooked_clips=len(hooked_clips),
            non_hooked_clips=len(non_hooked_clips),
            hooked_performance=hooked_performance,
            non_hooked_performance=non_hooked_performance,
            improvement_metrics=improvement_metrics,
            period=period,
            generated_at=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics summary: {str(e)}")


@router.get("/clips")
async def get_analytics_clips(
    platform: Optional[str] = None,
    limit: int = 50
) -> List[ClipAnalytics]:
    """Get list of clips with analytics data."""
    analytics_dir = Path("data/analytics")
    if not analytics_dir.exists():
        return []
    files = list(analytics_dir.glob(f"{platform or '*'}_analytics_*.json"))
    if not files:
        return []
    try:
        # Use the most recent file
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, "r", encoding="utf-8") as f:
            clips = json.load(f)
        
        # Convert to ClipAnalytics
        clip_analytics = []
        for clip in clips[:limit]:
            hooked = any(indicator in clip.get("title", "").lower() 
                         for indicator in ["hook", "intro", "hooked", "intro"])
            
            clip_analytics.append(ClipAnalytics(
                clip_id=clip["clip_id"],
                platform=clip["platform"],
                title=clip["title"],
                hooked=hooked,
                views=clip["views"],
                avg_watch_time=clip["avg_watch_time"],
                percent_watched=clip["percent_watched"],
                likes=clip["likes"],
                shares=clip["shares"],
                comments=clip["comments"],
                engagement_rate=clip["engagement_rate"],
                performance_score=clip["performance_score"]
            ))
        
        return clip_analytics

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get analytics clips: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics clips: {str(e)}")


@router.delete("/data")
async def clear_analytics_data() -> Dict[str, Any]:
    """Clear all analytics data."""
    try:
        analytics_dir = Path("data/analytics")
        if analytics_dir.exists():
            for file in analytics_dir.glob("*.json"):
                file.unlink()
        
        return {
            "success": True,
            "message": "All analytics data cleared"
        }
        
    except Exception as e:
        log.error(f"Failed to clear analytics data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear analytics data: {str(e)}")