"""Starts a throwaway dashboard backend for the end-to-end tests.

Every value that authenticates is created fresh for the run by playwright.config.ts and passed in
through the environment; the database, uploads and reports live in a temporary folder, so the
tests never touch the real dashboard's data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    work = Path(os.environ["E2E_WORK_DIR"])
    work.mkdir(parents=True, exist_ok=True)

    from webapp.backend.auth import hash_password

    os.environ.update(
        {
            "TW_WEBAPP_SECRET_KEY": os.environ["E2E_SECRET_KEY"],
            "TW_ADMIN_USERNAME": os.environ["E2E_ADMIN_USERNAME"],
            "TW_ADMIN_PASSWORD_HASH": hash_password(os.environ["E2E_ADMIN_PASSWORD"]),
            "TW_WEBAPP_DB": str(work / "webapp.db"),
            "TW_WEBAPP_UPLOADS_DIR": str(work / "uploads"),
            "TW_RUNS_DIR": str(work / "reports"),
        }
    )
    if os.environ.get("E2E_WEIGHTS"):
        # Resolved before the chdir below, so a path relative to webapp/frontend still works.
        os.environ["TW_WEIGHTS"] = str(Path(os.environ["E2E_WEIGHTS"]).resolve())
    else:
        os.environ.pop("TW_WEIGHTS", None)

    import uvicorn

    os.chdir(REPO_ROOT)
    uvicorn.run(
        "webapp.backend.main:app",
        host="127.0.0.1",
        port=int(os.environ["E2E_API_PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
