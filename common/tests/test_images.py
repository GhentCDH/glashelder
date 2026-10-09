"""Checks for streaming images from local paths and URLs."""

import asyncio
import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from common import images


def collect(refs) -> list[str]:
    async def run() -> list[str]:
        return [ref async for ref, _, _ in images.stream(refs, cache=False)]

    return asyncio.run(asyncio.wait_for(run(), timeout=10))


def test_streams_local_files():
    folder = Path(tempfile.mkdtemp())
    refs = []
    for n in range(3):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, "PNG")
        path = folder / f"{n}.png"
        path.write_bytes(buffer.getvalue())
        refs.append(str(path))
    assert sorted(collect(refs)) == refs


def test_failed_run_raises_instead_of_hanging(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("no client")

    monkeypatch.setattr(images.httpx, "AsyncClient", broken)
    with pytest.raises(RuntimeError, match="no client"):
        collect(["https://example.org/a.jpg"])
