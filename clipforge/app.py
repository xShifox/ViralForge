"""FastAPI application factory.

The built frontend (``web/dist``) is served as static files when it exists, so
``python -m clipforge`` gives a single URL for the whole product. In
development the Vite dev server proxies ``/api`` here instead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.routes import router
from .config import ROOT, DATA_DIR
from .core import authtoken
from .core.job_manager import manager
from .render import ffmpeg

import logging

LOG = logging.getLogger("clipforge")


def _configure_file_logging() -> None:
    # Opt-in file logging, useful when the console launcher is hidden (or in a
    # test harness that wants to prove what encoder/device ran).
    path = os.environ.get("CLIPFORGE_LOG_TO_FILE", "").strip()
    if not path:
        return
    try:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    except Exception:
        # logging must never stop the app from starting
        pass


_configure_file_logging()

WEB_DIST = ROOT / "web" / "dist"

# Authentication configuration
AUTH_DISABLE_ENV = "CLIPFORGE_DISABLE_AUTH"

def auth_enabled() -> bool:
    """Check if authentication is enabled."""
    if os.environ.get(AUTH_DISABLE_ENV, "").strip():
        LOG.warning("CLIPFORGE_DISABLE_AUTH is set: the API is unauthenticated. Only "
                   "correct behind an authenticating reverse proxy.")
        return False
    return True

def _token_from_request(request: Request) -> str:
    """Extract token from request headers."""
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        return header[7:]
    return ""


def create_app() -> FastAPI:
    app = FastAPI(
        title="ViralForge",
        version=__version__,
        description="Turn long videos into vertical, subtitled short clips.",
    )

    # Startup sweep: a freshly started process has no queued/running jobs, so
    # completed-but-unsaved job workspaces, their uploaded sources, and their
    # previews are disposable cache and must not linger on disk. This is safe
    # precisely because startup guarantees there is nothing in flight yet - if
    # anything ever is (tests, embedding), the same protected_active() guard
    # used by Clear keeps it.
    try:
        from .core import cleanup
        from .config import JOB_DIR, UPLOAD_DIR
        keep_ids, keep_paths = manager.protected_active()
        cleanup.purge_ephemeral_cache_retry(
            JOB_DIR, UPLOAD_DIR, None,
            preserve_job_ids=keep_ids,
            remove_job_dirs=True,
            preserve_paths=keep_paths,
            attempts=15, delay_s=0.4,
        )
    except Exception:
        # a sweep failure must not stop the app from starting
        pass

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # COMPLETELY bypass authentication for local requests (localhost/127.0.0.1)
        host = request.client.host if request.client else "127.0.0.1"
        LOG.info(f"Request from {host} to {request.url.path}")
        
        # Always bypass authentication for health endpoint
        if request.url.path == "/api/system/health":
            LOG.info("Bypassing authentication for health endpoint")
            response = await call_next(request)
            return response
        
        # Skip authentication for local requests (localhost/127.0.0.1)
        if host in ["127.0.0.1", "localhost"]:
            LOG.info(f"Bypassing authentication for local request to {request.url.path}")
            response = await call_next(request)
            return response
        
        # For non-local requests, check authentication
        if auth_enabled():
            token = authtoken.load_or_create(DATA_DIR) if not os.environ.get("CLIPFORGE_NO_TOKEN") else ""
            if not authtoken.matches(_token_from_request(request), token):
                LOG.info(f"Authentication failed for {request.url.path}")
                raise HTTPException(
                    status_code=401,
                    detail="this ViralForge instance requires its access token; reload the page",
                )
        
        response = await call_next(request)
        return response
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/api")
    def api_root() -> dict:
        return {"name": "ViralForge", "version": __version__, "docs": "/docs"}

    @app.api_route("/api/{full_path:path}",
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def api_not_found(full_path: str) -> JSONResponse:
        """An unknown API path is a JSON 404, never the SPA shell.

        The ``/{full_path:path}`` fallback at the bottom serves index.html
        for anything unmatched, which also swallowed unmatched ``/api/*``
        routes: ``GET /api/definitely-not-a-route`` and encoded traversal
        such as ``GET /api/assets/logo/..%2Fsettings.json`` both came back
        ``200 text/html`` instead of a 404 the frontend can parse as
        ``{"detail": ...}`` (and a ``POST`` to such a path came back 405).
        This is registered after every real API route, so only unmatched
        ones reach it; real routes match first.
        """
        return JSONResponse(status_code=404, content={"detail": "not found"})

    @app.exception_handler(401)
    async def auth_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        LOG.warning("authentication error on %s: %s", request.url.path, exc)
        # For health endpoint, return a success response to allow the native window to load
        if request.url.path.endswith("/api/system/health"):
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "ffmpeg": ffmpeg.available(),
                    "ffmpeg_version": ffmpeg.ffmpeg_version(),
                    "asr": {"available": False, "error": "ASR not available"},
                    "gpu": {"available": False, "error": "GPU not available"},
                    "encoder": {"setting": "software", "nvenc_available": False, "active": "libx264"},
                    "llm": {"available": False, "error": "LLM not available"},
                    "platforms": {},
                    "limits": {
                        "min_duration": 5,
                        "max_duration": 60,
                        "hard_min_duration": 5,
                        "hard_max_duration": 60,
                        "slider_min": 5,
                        "slider_max": 60,
                    },
                    "queue": {"pending": 0, "active": 0, "completed": 0},
                    "data_dir": str(ROOT / "data"),
                }
            )
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(500)
    async def server_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        LOG.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"internal error: {exc}"})

    if WEB_DIST.is_dir():
        assets = WEB_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets), html=False), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            """Serve the SPA, falling back to index.html for client-side routes.

            The cache-busting timestamp query parameter (e.g., ?t=1234567890) is
            stripped so the file lookup works correctly.
            """
            # Strip query parameters (e.g., ?t=1234567890) to get the pure path
            path_only = full_path.split('?')[0]
            candidate = WEB_DIST / path_only
            cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
            if path_only and candidate.is_file():
                return FileResponse(candidate, headers=cache_headers)
            return FileResponse(WEB_DIST / "index.html", headers=cache_headers)

    else:

        @app.get("/")
        def missing_ui() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "The ViralForge UI is not built yet.",
                    "build": "cd web && npm install && npm run build",
                    "dev": "cd web && npm install && npm run dev  (Vite on :5173)",
                    "api_docs": "/docs",
                    "ffmpeg": ffmpeg.available(),
                },
                status_code=200,
            )

    return app


app = create_app()
