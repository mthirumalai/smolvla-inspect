"""FastAPI app — serves the smolvla-inspect web viewer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .routers import compare, internals, llm, llm_cache, notes, runs, visualizations

_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(s: Settings) -> None:
    global _settings
    _settings = s


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.base_dir = Path(s.base_dir).resolve()
    print(f"smolvla-inspect web viewer")
    print(f"  Base directory: {s.base_dir}")
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings:
        set_settings(settings)

    app = FastAPI(
        title="smolvla-inspect",
        description="Web viewer for SmolVLA inspection runs",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_origin_regex=get_settings().cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(runs.router)
    app.include_router(visualizations.router)
    app.include_router(internals.router)
    app.include_router(compare.router)
    app.include_router(llm.router)
    app.include_router(notes.router)
    app.include_router(llm_cache.router)

    @app.get("/api/health")
    async def api_health():
        return {"status": "ok"}

    # Mount static files for built frontend (if present)
    s = get_settings()
    if s.static_dir and Path(s.static_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(s.static_dir), html=True),
                  name="frontend")

    return app


# For `uvicorn web.backend.main:app`
app = create_app()
