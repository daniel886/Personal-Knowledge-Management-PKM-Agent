"""FastAPI server with built-in web search panel."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.routes import chat, ingest, review, search
from core.config import settings
from core.scheduler import register_default_jobs
from models.database import init_db
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting PKM Agent…")
    await init_db()
    try:
        await register_default_jobs()
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Scheduler init skipped: {exc}")
    yield
    logger.info("🛑 Shutting down PKM Agent…")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Personal Knowledge Management (PKM) Agent",
        description="AI-powered personal knowledge management with Obsidian sync, "
        "vector search and weekly auto-reviews.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files (web search panel)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routers
    app.include_router(ingest.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(review.router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": app.version}

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index = static_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>PKM Agent</h1><p>Visit /docs for API.</p>")

    return app


app = create_app()


def run() -> None:
    """Used by `python main.py serve` / docker entrypoint."""
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_level=settings.app_log_level.lower(),
    )


if __name__ == "__main__":
    run()
