"""Dashboard auth: role-based accounts, hashed passwords, signed session cookie, rate-limited
login.

Fail-closed rules (project security defaults, applied here):
- password verification uses bcrypt's own constant-time check, never `==`
- every route but /api/login and /api/logout requires a valid session, checked before any
  business logic runs; the account is re-read on every request, so disabling a user, changing
  their role or resetting their password takes effect on their next request
- a route limited to some roles refuses every other role with 403 before its body runs
- a locked-out or invalid login never reveals whether the username or the password was wrong

Uses the bcrypt package directly, not passlib: passlib 1.7.4 (the newest release on PyPI) reads
an attribute bcrypt itself removed in its 4.x line, so passlib's bcrypt backend fails outright
against any bcrypt newer than 3.x. bcrypt's own hashpw/checkpw need no such wrapper.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import bcrypt
from fastapi import Cookie, Depends, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from webapp.backend import db
from webapp.backend.settings import Settings

SESSION_COOKIE = "tw_session"
MAX_PASSWORD_BYTES = 72  # bcrypt's own hard limit; reject rather than silently truncate
MIN_PASSWORD_LENGTH = 8


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


@dataclass(frozen=True)
class CurrentUser:
    id: int
    username: str
    role: str


def validate_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")


def verify_login(conn: sqlite3.Connection, username: str, password: str) -> sqlite3.Row | None:
    """Constant-time-verified check; an unknown or disabled user still runs a hash, so response
    timing does not tell which part of the login was wrong."""
    user = db.get_user(conn, username)
    if user is None or not user["active"]:
        _verify_password(password, _DUMMY_HASH)
        return None
    return user if _verify_password(password, user["password_hash"]) else None


def _password_fingerprint(settings: Settings, password_hash: str) -> str:
    """Binds a session to the password it was issued under: a reset changes the hash and so
    ends every session issued before it. Keyed, so the cookie reveals nothing about the hash."""
    key = settings.secret_key.encode("utf-8")
    return hmac.new(key, password_hash.encode("ascii"), hashlib.sha256).hexdigest()[:32]


def create_session_token(settings: Settings, user: sqlite3.Row) -> str:
    serializer = URLSafeTimedSerializer(settings.secret_key, salt="tw-webapp-session")
    return serializer.dumps(
        {"uid": user["id"], "pw": _password_fingerprint(settings, user["password_hash"])}
    )


def read_session_token(settings: Settings, token: str) -> dict | None:
    serializer = URLSafeTimedSerializer(settings.secret_key, salt="tw-webapp-session")
    try:
        data = serializer.loads(token, max_age=settings.session_max_age_s)
    except (BadSignature, SignatureExpired):
        return None
    return data if isinstance(data, dict) else None


def require_session(request: Request, tw_session: str | None = Cookie(default=None)) -> CurrentUser:
    """FastAPI dependency: fails closed with 401 before any route body runs. Reads settings from
    request.app.state at request time, since they are only known once lifespan() has run, which
    is after the app (and its dependencies) are constructed."""
    settings: Settings = request.app.state.settings
    data = read_session_token(settings, tw_session) if tw_session else None
    user = None
    if data is not None and isinstance(data.get("uid"), int):
        user = db.get_user_by_id(request.app.state.db, data["uid"])
    if (
        user is None
        or not user["active"]
        or not hmac.compare_digest(
            str(data.get("pw", "")), _password_fingerprint(settings, user["password_hash"])
        )
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    return CurrentUser(id=user["id"], username=user["username"], role=user["role"])


def require_roles(*roles: str) -> Callable[..., CurrentUser]:
    """FastAPI dependency factory: 401 without a session, 403 for any role not listed."""
    unknown = set(roles) - set(db.ROLES)
    if unknown:
        raise ValueError(f"unknown roles: {sorted(unknown)}")

    def dependency(user: CurrentUser = Depends(require_session)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, detail="your role does not allow this action"
            )
        return user

    return dependency
