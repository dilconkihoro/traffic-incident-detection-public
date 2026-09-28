"""Password hashing, session tokens, and the login rate limiter, all offline (no server)."""

from __future__ import annotations

import time

import pytest

from webapp.backend.auth import (
    MAX_PASSWORD_BYTES,
    PasswordTooLongError,
    RateLimiter,
    _verify_password,
    create_session_token,
    hash_password,
    read_session_token,
    verify_login,
)
from webapp.backend.settings import Settings


def make_settings(**overrides) -> Settings:
    defaults = dict(
        secret_key="x" * 32,
        admin_username="admin",
        admin_password_hash=hash_password("correct-horse-battery-staple"),
        database_path=None,
        uploads_dir=None,
        reports_dir=None,
        pipeline_config_path=None,
        login_rate_limit_attempts=5,
        login_rate_limit_window_s=60.0,
        session_max_age_s=3600,
        max_upload_frames=180,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_hash_and_verify_round_trip():
    hashed = hash_password("a real password")
    assert _verify_password("a real password", hashed)
    assert not _verify_password("wrong password", hashed)


def test_password_over_72_bytes_is_rejected_not_silently_truncated():
    with pytest.raises(PasswordTooLongError):
        hash_password("x" * (MAX_PASSWORD_BYTES + 1))


def test_verify_login_checks_username_and_password():
    settings = make_settings()
    assert verify_login(settings, "admin", "correct-horse-battery-staple")
    assert not verify_login(settings, "admin", "wrong")
    assert not verify_login(settings, "someone-else", "correct-horse-battery-staple")


def test_verify_login_runs_a_real_hash_check_even_on_an_unknown_username():
    """Otherwise a wrong username returns fast and a wrong password does not, which leaks which
    field was wrong through response timing."""
    settings = make_settings()
    slow = time.perf_counter()
    verify_login(settings, "admin", "wrong")
    slow_elapsed = time.perf_counter() - slow
    fast = time.perf_counter()
    verify_login(settings, "does-not-exist", "wrong")
    fast_elapsed = time.perf_counter() - fast
    # Both do a real bcrypt comparison; neither should be orders of magnitude faster.
    assert fast_elapsed > slow_elapsed / 5


def test_session_token_round_trips_and_rejects_a_tampered_token():
    settings = make_settings()
    token = create_session_token(settings, "admin")
    assert read_session_token(settings, token) == "admin"
    assert read_session_token(settings, token + "x") is None


def test_session_token_from_a_different_secret_is_rejected():
    settings_a = make_settings(secret_key="a" * 32)
    settings_b = make_settings(secret_key="b" * 32)
    token = create_session_token(settings_a, "admin")
    assert read_session_token(settings_b, token) is None


def test_session_token_expires():
    settings = make_settings(session_max_age_s=0)
    token = create_session_token(settings, "admin")
    time.sleep(1.1)
    assert read_session_token(settings, token) is None


def test_rate_limiter_blocks_after_the_configured_number_of_attempts():
    limiter = RateLimiter(max_attempts=3, window_s=60.0)
    for _ in range(3):
        assert limiter.check("admin")
        limiter.record("admin")
    assert not limiter.check("admin")


def test_rate_limiter_is_per_key():
    limiter = RateLimiter(max_attempts=1, window_s=60.0)
    limiter.record("admin")
    assert not limiter.check("admin")
    assert limiter.check("someone-else")


def test_rate_limiter_window_expires():
    limiter = RateLimiter(max_attempts=1, window_s=0.5)
    limiter.record("admin")
    assert not limiter.check("admin")
    time.sleep(0.6)
    assert limiter.check("admin")
