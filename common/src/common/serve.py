"""Shared startup for the Flask-based serve commands."""

import logging
import os
import threading
import webbrowser
from urllib.parse import urlencode

from .auth import AUTH_SECRET_KEY, require_auth

logger = logging.getLogger(__name__)


def run_app(
    app, *, host: str, port: int, auth: bool, debug: bool, open_browser: bool
) -> None:
    """Run `app`, behind the secret when `auth` is set. Takes the options of
    `common.cli.serve_options`.

    Logs the local url (with the secret when auth is on) and optionally opens it.
    """
    if auth:
        require_auth(app)
    secret = app.config.get(AUTH_SECRET_KEY)
    query = f"?{urlencode({'secret': secret})}" if secret else ""
    url = f"http://localhost:{port}/{query}"

    logger.info(f"Starting Flask server on {host}:{port}...")
    logger.info(f"Open {url}")
    # the debug reloader runs the command again in a child process
    if open_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    app.run(host=host, port=port, debug=debug)
