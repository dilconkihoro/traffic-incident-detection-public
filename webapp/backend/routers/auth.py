from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from webapp.backend.auth import SESSION_COOKIE, create_session_token, verify_login
from webapp.backend.settings import Settings

router = APIRouter(prefix="/api")


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    limiter = request.app.state.login_rate_limiter
    if not limiter.check(payload.username):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, detail="too many attempts, try later"
        )

    if not verify_login(settings, payload.username, payload.password):
        limiter.record(payload.username)  # only failures spend rate-limit budget
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid username or password")

    token = create_session_token(settings, payload.username)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_max_age_s,
        httponly=True,
        samesite="lax",
    )
    return {"username": payload.username}


@router.post("/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}
