from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from webapp.backend import db
from webapp.backend.auth import (
    CurrentUser,
    PasswordTooLongError,
    hash_password,
    require_roles,
    validate_new_password,
)

router = APIRouter(prefix="/api/users")

Role = Literal["admin", "operator", "analyst"]
USERNAME_MAX_LENGTH = 64


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: Role


class UpdateUserRequest(BaseModel):
    role: Role | None = None
    active: bool | None = None
    password: str | None = None


def _public(row) -> dict[str, object]:
    """Never returns password_hash."""
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "active": bool(row["active"]),
        "is_bootstrap": bool(row["is_bootstrap"]),
        "created_at": row["created_at"],
        "created_by": row["created_by"],
    }


def _checked_hash(password: str) -> str:
    try:
        validate_new_password(password)
        return hash_password(password)
    except (ValueError, PasswordTooLongError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("")
def list_users(
    request: Request, _: CurrentUser = Depends(require_roles("admin"))
) -> list[dict[str, object]]:
    return [_public(row) for row in db.list_users(request.app.state.db)]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_user(
    payload: CreateUserRequest,
    request: Request,
    admin: CurrentUser = Depends(require_roles("admin")),
) -> dict[str, object]:
    username = payload.username.strip()
    if not username or len(username) > USERNAME_MAX_LENGTH or not username.isprintable():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"username must be 1 to {USERNAME_MAX_LENGTH} printable characters",
        )
    conn = request.app.state.db
    try:
        user_id = db.create_user(
            conn,
            username,
            _checked_hash(payload.password),
            payload.role,
            datetime.now(timezone.utc).isoformat(),
            admin.username,
        )
    except db.UserExistsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _public(db.get_user_by_id(conn, user_id))


@router.patch("/{user_id}")
def update_user(
    user_id: int,
    payload: UpdateUserRequest,
    request: Request,
    admin: CurrentUser = Depends(require_roles("admin")),
) -> dict[str, object]:
    conn = request.app.state.db
    target = db.get_user_by_id(conn, user_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="user not found")
    changes_access = payload.role is not None or payload.active is not None
    if target["is_bootstrap"] and changes_access:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="the bootstrap administrator is managed through .env, not the dashboard",
        )
    if target["id"] == admin.id and changes_access:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="you cannot change your own role or disable yourself"
        )
    if target["is_bootstrap"] and payload.password is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="the bootstrap administrator's password is set with set_admin_password.py",
        )

    fields: dict[str, object] = {}
    if payload.role is not None:
        fields["role"] = payload.role
    if payload.active is not None:
        fields["active"] = int(payload.active)
    if payload.password is not None:
        fields["password_hash"] = _checked_hash(payload.password)
    db.update_user(conn, user_id, **fields)
    return _public(db.get_user_by_id(conn, user_id))
