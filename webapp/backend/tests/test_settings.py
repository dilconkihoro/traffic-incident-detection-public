"""Startup validation fails closed (same principle as trafficwatch.config)."""

from __future__ import annotations

import pytest

from webapp.backend.settings import MIN_SECRET_KEY_LENGTH, SettingsError, load_settings

VALID_ENV = {
    "TW_WEBAPP_SECRET_KEY": "x" * MIN_SECRET_KEY_LENGTH,
    "TW_ADMIN_USERNAME": "admin",
    "TW_ADMIN_PASSWORD_HASH": "$2b$12$abcdefghijklmnopqrstuv",
}


def test_valid_env_loads():
    settings = load_settings(dict(VALID_ENV))
    assert settings.admin_username == "admin"
    assert settings.login_rate_limit_attempts == 5


def test_missing_secret_key_refuses_to_start():
    env = {k: v for k, v in VALID_ENV.items() if k != "TW_WEBAPP_SECRET_KEY"}
    with pytest.raises(SettingsError, match="TW_WEBAPP_SECRET_KEY"):
        load_settings(env)


def test_too_short_secret_key_refuses_to_start():
    env = {**VALID_ENV, "TW_WEBAPP_SECRET_KEY": "short"}
    with pytest.raises(SettingsError, match="TW_WEBAPP_SECRET_KEY"):
        load_settings(env)


def test_missing_admin_username_refuses_to_start():
    env = {k: v for k, v in VALID_ENV.items() if k != "TW_ADMIN_USERNAME"}
    with pytest.raises(SettingsError, match="TW_ADMIN_USERNAME"):
        load_settings(env)


def test_missing_admin_password_hash_refuses_to_start():
    env = {k: v for k, v in VALID_ENV.items() if k != "TW_ADMIN_PASSWORD_HASH"}
    with pytest.raises(SettingsError, match="TW_ADMIN_PASSWORD_HASH"):
        load_settings(env)


def test_blank_admin_username_refuses_to_start():
    env = {**VALID_ENV, "TW_ADMIN_USERNAME": "   "}
    with pytest.raises(SettingsError, match="TW_ADMIN_USERNAME"):
        load_settings(env)


def test_by_default_one_run_at_a_time_on_at_most_half_the_cores():
    import os

    settings = load_settings(dict(VALID_ENV))
    assert settings.max_concurrent_runs == 1
    assert 1 <= settings.max_cpu_threads <= max(1, (os.cpu_count() or 2) // 2)


def test_run_limits_can_be_overridden():
    env = {**VALID_ENV, "TW_MAX_CONCURRENT_RUNS": "2", "TW_MAX_CPU_THREADS": "3"}
    settings = load_settings(env)
    assert settings.max_concurrent_runs == 2
    assert settings.max_cpu_threads == 3


@pytest.mark.parametrize("name", ["TW_MAX_CONCURRENT_RUNS", "TW_MAX_CPU_THREADS"])
def test_a_zero_run_limit_refuses_to_start_rather_than_queueing_forever(name):
    with pytest.raises(SettingsError, match=name):
        load_settings({**VALID_ENV, name: "0"})
