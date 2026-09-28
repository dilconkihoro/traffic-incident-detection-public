"""Admin auth: one account, hashed password, signed session cookie, rate-limited login.

Fail-closed rules (the project rules security defaults, applied here):
- password verification uses bcrypt's own constant-time check, never `==`
- every route but /api/login and /api/logout requires a valid session, checked before any
  business logic runs
- a locked-out or invalid login never reveals whether the username or the password was wrong

Uses the bcrypt package directly, not passlib: passlib 1.7.4 (the newest release on PyPI) reads
an attribute bcrypt itself removed in its 4.x line, so passlib's bcrypt backend fails outright
against any bcrypt newer than 3.x. bcrypt's own hashpw/checkpw need no such wrapper.
"""

from __future__ import annotations

import time
from collections import deque

import bcrypt
from fastapi import Cookie, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from webapp.backend.settings import Settings

SESSION_COOKIE = "tw_session"
MAX_PASSWORD_BYTES = 72  # bcrypt's own hard limit; reject rather than silently truncate


class PasswordTooLongError(ValueError):
    pass


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


def _verify_password(password: str, password_hash: str) -> bool:
    encoded = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("ascii"))
    except ValueError:
        return False  # malformed stored hash: fail closed, not an exception past the auth check


class RateLimiter:
    """A sliding-window limiter per username, held in memory (single-process, local dashboard)."""

    def __init__(self, max_attempts: int, window_s: float) -> None:
        self._max_attempts = max_attempts
        self._window_s = window_s
        self._attempts: dict[str, deque[float]] = {}

    def check(self, key: str) -> bool:
        """Returns True if another attempt is allowed for ``key`` right now."""
        now = time.monotonic()
        attempts = self._attempts.setdefault(key, deque())
        while attempts and now - attempts[0] > self._window_s:
            attempts.popleft()
        return len(attempts) < self._max_attempts

    def record(self, key: str) -> None:
        self._attempts.setdefault(key, deque()).append(time.monotonic())


# A fixed, valid bcrypt hash of no real password, used only so a wrong username still runs a
# genuine bcrypt comparison (same cost as a real one), rather than returning early and leaking
# which part of the login was wrong through response timing.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt()).decode("ascii")


def verify_login(settings: Settings, username: str, password: str) -> bool:
    """Constant-time-verified check; a wrong username still runs a hash to avoid a timing tell."""
    if username != settings.admin_username:
        _verify_password(password, _DUMMY_HASH)
        return False
    return _verify_password(password, settings.admin_password_hash)


def create_session_token(settings: Settings, username: str) -> str:
    serializer = URLSafeTimedSerializer(settings.secret_key, salt="tw-webapp-session")
    return serializer.dumps({"username": username})


def read_session_token(settings: Settings, token: str) -> str | None:
    serializer = URLSafeTimedSerializer(settings.secret_key, salt="tw-webapp-session")
    try:
        data = serializer.loads(token, max_age=settings.session_max_age_s)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("username")


def require_session(request: Request, tw_session: str | None = Cookie(default=None)) -> str:
    """FastAPI dependency: fails closed with 401 before any route body runs. Reads settings from
    request.app.state at request time, since they are only known once lifespan() has run, which
    is after the app (and its dependencies) are constructed."""
    settings: Settings = request.app.state.settings
    username = read_session_token(settings, tw_session) if tw_session else None
    if username is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    return username
