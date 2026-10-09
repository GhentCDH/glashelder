"""Shared startup for the Flask-based serve commands."""

import logging
import os
import threading
import webbrowser
from urllib.parse import urlencode

from .auth import AUTH_SECRET_KEY

logger = logging.getLogger(__name__)


def run_app(app, host: str, port: int, *, open_browser: bool, **kwargs) -> None:
    """Log the local url (with the secret when auth is on), optionally open it,
    then run `app`. Extra keyword arguments go to `app.run`."""
    secret = app.config.get(AUTH_SECRET_KEY)
    query = f"?{urlencode({'secret': secret})}" if secret else ""
    url = f"http://localhost:{port}/{query}"

    logger.info(f"Starting Flask server on {host}:{port}...")
    logger.info(f"Open {url}")
    # the debug reloader runs the command again in a child process
    if open_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    app.run(host=host, port=port, **kwargs)
