"""Authentication utilities for Flask-based serve commands.

This module provides cookie-based authentication for web servers.
Authentication works via:
1. Query parameter `?secret=<value>` - sets a session cookie if valid
2. Cookie `glashelder_auth` - checked on subsequent requests

The expected secret is the `secret` config value (`GLASHELDER_SECRET` in the environment).
"""

import hashlib
import logging
import os
import secrets

# Cookie settings for authentication
AUTH_COOKIE_NAME = "glashelder_auth"
AUTH_COOKIE_MAX_AGE = 86400  # 1 day in seconds
# app.config key that holds the expected secret
AUTH_SECRET_KEY = "GLASHELDER_SECRET"

logger = logging.getLogger(__name__)


def hash_secret(secret: str) -> str:
    """Hash a secret for comparison/storage."""
    return hashlib.sha256(secret.encode()).hexdigest()


def require_auth(app):
    """Add authentication middleware to a Flask app.

    Authentication works via:
    1. Query parameter `?secret=<value>` - sets a session cookie if valid
    2. Cookie `glashelder_auth` - checked on subsequent requests

    The expected secret is the `secret` config value. If not set, a random
    secret is generated. Either way it is stored in `app.config[AUTH_SECRET_KEY]`.

    Args:
        app: Flask application instance

    Returns:
        The secret clients must provide
    """
    from flask import g, request

    from .config import config

    expected_secret = config().secret

    if not expected_secret:
        logger.warning("no secret configured - generating a random one for this run")
        expected_secret = secrets.token_urlsafe(32)
        # the debug reloader's child process inherits this, so it keeps the same secret
        os.environ["GLASHELDER_SECRET"] = expected_secret

    app.config[AUTH_SECRET_KEY] = expected_secret
    expected_hash = hash_secret(expected_secret)

    def matches(value: str) -> bool:
        # bytes, since compare_digest rejects non-ASCII str
        return secrets.compare_digest(value.encode(), expected_hash.encode())

    @app.before_request
    def check_auth():
        # Check query parameter first
        provided_secret = request.args.get("secret")
        if provided_secret:
            if matches(hash_secret(provided_secret)):
                # Valid secret in query - will set cookie in after_request
                g.set_auth_cookie = True
                return None
            else:
                return "Unauthorized: invalid secret", 401

        # Check cookie
        auth_cookie = request.cookies.get(AUTH_COOKIE_NAME)
        if auth_cookie and matches(auth_cookie):
            return None

        return "Unauthorized: access /?secret=<your-secret> to authenticate", 401

    @app.after_request
    def set_auth_cookie(response):
        if g.get("set_auth_cookie", False):
            response.set_cookie(
                AUTH_COOKIE_NAME,
                expected_hash,
                max_age=AUTH_COOKIE_MAX_AGE,
                httponly=True,
                samesite="Lax",
            )
        return response

    return expected_secret
