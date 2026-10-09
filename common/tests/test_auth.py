"""Checks for the secret based authentication of the serve commands."""

import os

from flask import Flask

from common import require_auth
from common.auth import AUTH_COOKIE_NAME, hash_secret
from common.config import config


def client(secret: str):
    os.environ["GLASHELDER_SECRET"] = secret
    config.cache_clear()
    app = Flask(__name__)
    require_auth(app)
    app.add_url_rule("/", view_func=lambda: "ok")
    return app.test_client()


def test_requires_secret():
    assert client("s3cret").get("/").status_code == 401


def test_secret_in_query_sets_cookie():
    test = client("s3cret")
    response = test.get("/?secret=s3cret")
    assert response.status_code == 200
    assert test.get_cookie(AUTH_COOKIE_NAME).value == hash_secret("s3cret")
    assert test.get("/").status_code == 200


def test_wrong_secret():
    assert client("s3cret").get("/?secret=nope").status_code == 401


def test_non_ascii_cookie_is_rejected():
    test = client("s3cret")
    test.set_cookie(AUTH_COOKIE_NAME, "é")
    assert test.get("/").status_code == 401
