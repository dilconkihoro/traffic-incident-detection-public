"""Hash an admin password for the web dashboard. Prints env-var lines to add to your own .env;
never writes or prints the plaintext password, and never stores it anywhere."""

from __future__ import annotations

import getpass
import secrets
import sys
from pathlib import Path

# webapp/ is a standalone layer (decision D-25), not part of the installed trafficwatch package,
# so its own repo root needs to be on the path when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.backend.auth import (  # noqa: E402
    MAX_PASSWORD_BYTES,
    PasswordTooLongError,
    hash_password,
)


def main() -> int:
    username = input("Admin username: ").strip()
    if not username:
        print("error: username cannot be empty", file=sys.stderr)
        return 2
    password = getpass.getpass("Admin password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("error: passwords did not match", file=sys.stderr)
        return 2
    if len(password) < 8:
        print("error: password must be at least 8 characters", file=sys.stderr)
        return 2

    try:
        password_hash = hash_password(password)
    except PasswordTooLongError:
        print(f"error: password must be at most {MAX_PASSWORD_BYTES} bytes", file=sys.stderr)
        return 2
    secret_key = secrets.token_hex(32)

    print("\nAdd these to your own .env (never commit .env, never share these values):\n")
    print(f"TW_ADMIN_USERNAME={username}")
    print(f"TW_ADMIN_PASSWORD_HASH={password_hash}")
    print(f"TW_WEBAPP_SECRET_KEY={secret_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
