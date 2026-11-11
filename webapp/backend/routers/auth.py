from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from webapp.backend.auth import (
    SESSION_COOKIE,
    CurrentUser,
    create_session_token,
    require_session,
    verify_login,
)
from webapp.backend.settings import Settings

router = APIRouter(prefix="/api")


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    limiter = request.app.state.login_rate_limiter
    key = payload.username.strip().lower()
    if not limiter.check(key):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, detail="too many attempts, try later"
        )

    user = verify_login(request.app.state.db, payload.username.strip(), payload.password)
    if user is None:
        limiter.record(key)  # only failures spend rate-limit budget
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid username or password")

    token = create_session_token(settings, user)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_max_age_s,
        httponly=True,
        samesite="lax",
    )
    return {"username": user["username"], "role": user["role"]}


@router.post("/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@router.get("/me")
def me(user: CurrentUser = Depends(require_session)) -> dict[str, str]:
    return {"username": user.username, "role": user.role}
