"""Web dashboard settings, validated at startup, fail closed (same principle as
trafficwatch.config: a missing or too-short secret must stop the app, not run insecurely).

This layer is supplementary (decision D-25, the decision log): it never touches
src/trafficwatch/, only calls it, the same way scripts/make_demo_clip.py already does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MIN_SECRET_KEY_LENGTH = 32
REPO_ROOT = Path(__file__).resolve().parents[2]


class SettingsError(RuntimeError):
    """Raised at startup; the app must refuse to run rather than start insecurely."""


@dataclass(frozen=True)
class Settings:
    secret_key: str
    admin_username: str
    admin_password_hash: str
    database_path: Path
    uploads_dir: Path
    reports_dir: Path
    pipeline_config_path: Path
    login_rate_limit_attempts: int
    login_rate_limit_window_s: float
    session_max_age_s: int
    max_upload_frames: int
    max_concurrent_runs: int = 1
    max_cpu_threads: int = 1


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    secret_key = env.get("TW_WEBAPP_SECRET_KEY", "")
    if len(secret_key) < MIN_SECRET_KEY_LENGTH:
        raise SettingsError(
            f"TW_WEBAPP_SECRET_KEY must be set and at least {MIN_SECRET_KEY_LENGTH} characters "
            '(generate one with: python -c "import secrets; print(secrets.token_hex(32))")'
        )

    admin_username = env.get("TW_ADMIN_USERNAME", "").strip()
    if not admin_username:
        raise SettingsError("TW_ADMIN_USERNAME must be set")

    admin_password_hash = env.get("TW_ADMIN_PASSWORD_HASH", "")
    if not admin_password_hash:
        raise SettingsError(
            "TW_ADMIN_PASSWORD_HASH must be set (run scripts/set_admin_password.py to create one; "
            "never put a plaintext password here)"
        )

    max_concurrent_runs = int(env.get("TW_MAX_CONCURRENT_RUNS", "1"))
    if max_concurrent_runs < 1:
        raise SettingsError("TW_MAX_CONCURRENT_RUNS must be at least 1 (0 would queue forever)")
    # Half the cores: a run is CPU-bound, and using every core for minutes can freeze the
    # desktop the dashboard is running on.
    max_cpu_threads = int(env.get("TW_MAX_CPU_THREADS", str(max(1, (os.cpu_count() or 2) // 2))))
    if max_cpu_threads < 1:
        raise SettingsError("TW_MAX_CPU_THREADS must be at least 1")

    return Settings(
        secret_key=secret_key,
        admin_username=admin_username,
        admin_password_hash=admin_password_hash,
        database_path=Path(
            env.get("TW_WEBAPP_DB", str(REPO_ROOT / "webapp" / "backend" / "webapp.db"))
        ),
        uploads_dir=Path(
            env.get("TW_WEBAPP_UPLOADS_DIR", str(REPO_ROOT / "data" / "webapp_uploads"))
        ),
        reports_dir=Path(env.get("TW_RUNS_DIR", str(REPO_ROOT / "reports"))),
        pipeline_config_path=Path(
            env.get("TW_CONFIG", str(REPO_ROOT / "configs" / "pipeline.yaml"))
        ),
        login_rate_limit_attempts=int(env.get("TW_LOGIN_RATE_LIMIT_ATTEMPTS", "5")),
        login_rate_limit_window_s=float(env.get("TW_LOGIN_RATE_LIMIT_WINDOW_S", "300")),
        session_max_age_s=int(env.get("TW_SESSION_MAX_AGE_S", str(8 * 3600))),
        # This machine has no GPU (about 1.5 fps measured); 180 frames keeps a worst-case run
        # under roughly two minutes so an upload can't quietly overwhelm the machine.
        max_upload_frames=int(env.get("TW_MAX_UPLOAD_FRAMES", "180")),
        max_concurrent_runs=max_concurrent_runs,
        max_cpu_threads=max_cpu_threads,
    )
