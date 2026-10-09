"""Authentication utilities for Flask-based serve commands.

This module provides cookie-based authentication for web servers.
Authentication works via:
1. Query parameter `?secret=<value>` - sets a session cookie if valid
2. Cookie `glashelder_auth` - checked on subsequent requests

The expected secret should be set in GLASHELDER_SECRET environment variable.
"""

import hashlib
import logging
import os
import secrets

# Cookie settings for authentication
AUTH_COOKIE_NAME = "glashelder_auth"
AUTH_COOKIE_MAX_AGE = 86400  # 1 day in seconds

logger = logging.getLogger(__name__)


def get_secret_from_env() -> str | None:
    """Get the authentication secret from environment variable."""
    return os.getenv("GLASHELDER_SECRET")


def hash_secret(secret: str) -> str:
    """Hash a secret for comparison/storage."""
    return hashlib.sha256(secret.encode()).hexdigest()


def require_auth(app):
    """Add authentication middleware to a Flask app.

    Authentication works via:
    1. Query parameter `?secret=<value>` - sets a session cookie if valid
    2. Cookie `glashelder_auth` - checked on subsequent requests

    The expected secret should be set in GLASHELDER_SECRET environment variable.
    If not set, a random secret is generated and logged.

    Args:
        app: Flask application instance

    Returns:
        The secret clients must provide
    """
    from flask import request

    expected_secret = get_secret_from_env()

    if not expected_secret:
        logger.warning(
            "GLASHELDER_SECRET not set - generating a random secret for this session"
        )
        expected_secret = secrets.token_urlsafe(32)
        logger.info(f"Access the server with: ?secret={expected_secret}")

    expected_hash = hash_secret(expected_secret)

    @app.before_request
    def check_auth():
        # Check query parameter first
        provided_secret = request.args.get("secret")
        if provided_secret:
            if hash_secret(provided_secret) == expected_hash:
                # Valid secret in query - will set cookie in after_request
                request.auth_valid = True
                request.set_auth_cookie = True
                return None
            else:
                return "Unauthorized: invalid secret", 401

        # Check cookie
        auth_cookie = request.cookies.get(AUTH_COOKIE_NAME)
        if auth_cookie and auth_cookie == expected_hash:
            request.auth_valid = True
            request.set_auth_cookie = False
            return None

        return "Unauthorized: access /?secret=<your-secret> to authenticate", 401

    @app.after_request
    def set_auth_cookie(response):
        if getattr(request, "set_auth_cookie", False):
            response.set_cookie(
                AUTH_COOKIE_NAME,
                expected_hash,
                max_age=AUTH_COOKIE_MAX_AGE,
                httponly=True,
                samesite="Lax",
            )
        return response

    return expected_secret
