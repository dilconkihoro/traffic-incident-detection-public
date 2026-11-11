"""FastAPI app for the supplementary web dashboard (decision D-25, the decision log).

Refuses to start on a missing or invalid configuration (fail closed, same principle as
trafficwatch.config): see webapp/backend/settings.py. Never modifies src/trafficwatch/ behaviour
for the CLI; only calls into it, the same way scripts/make_demo_clip.py already does.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend import db
from webapp.backend.auth import RateLimiter, require_roles, require_session
from webapp.backend.jobs import JobRegistry
from webapp.backend.routers import auth as auth_router
from webapp.backend.routers import models as models_router
from webapp.backend.routers import runs as runs_router
from webapp.backend.routers import uploads as uploads_router
from webapp.backend.routers import users as users_router
from webapp.backend.settings import load_settings

# Vite's default dev server port; the built frontend is served separately in production and
# needs no CORS entry (same origin).
DEV_FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()  # raises SettingsError and refuses to start if misconfigured
    app.state.settings = settings
    app.state.db = db.connect(settings.database_path)
    db.fail_interrupted_runs(app.state.db)
    db.ensure_bootstrap_admin(
        app.state.db,
        settings.admin_username,
        settings.admin_password_hash,
        datetime.now(timezone.utc).isoformat(),
    )
    app.state.jobs = JobRegistry(settings.max_concurrent_runs, settings.max_cpu_threads)
    app.state.login_rate_limiter = RateLimiter(
        settings.login_rate_limit_attempts, settings.login_rate_limit_window_s
    )
    try:
        yield
    finally:
        app.state.db.close()


def create_app() -> FastAPI:
    app = FastAPI(title="TrafficWatch dashboard", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_FRONTEND_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router.router)
    # Every route in these routers requires a valid session, checked before the route body
    # runs (fail closed): require_session reads settings from request.app.state at request time.
    # Routers that start or prepare runs are limited to the roles that may do so; the runs
    # router narrows its own write routes further.
    session_required = [Depends(require_session)]
    runs_allowed = [Depends(require_roles("admin", "operator"))]
    app.include_router(uploads_router.router, dependencies=runs_allowed)
    app.include_router(runs_router.router, dependencies=session_required)
    app.include_router(models_router.router, dependencies=runs_allowed)
    app.include_router(users_router.router, dependencies=[Depends(require_roles("admin"))])
    return app


app = create_app()
