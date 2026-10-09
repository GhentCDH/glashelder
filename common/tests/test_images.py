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


def test_list_images():
    folder = Path(tempfile.mkdtemp())
    for name in ["b.jpg", "a.png", "notes.txt", "sub/c.jpg", "sub/d.tiff", ".x.jpg"]:
        (folder / name).parent.mkdir(exist_ok=True)
        (folder / name).touch()
    names = [str(p.relative_to(folder)) for p in images.list_images(folder)]
    assert names == ["a.png", "b.jpg", "sub/c.jpg", "sub/d.tiff"]
    flat = images.list_images(folder, recursive=False, suffixes=(".jpg",))
    assert flat == [folder / "b.jpg"]


def test_thumbnail_keeps_aspect_and_never_grows():
    folder = Path(tempfile.mkdtemp())
    Image.new("RGB", (400, 200)).save(folder / "wide.jpg")
    Image.new("RGB", (10, 20)).save(folder / "small.png")
    assert images.thumbnail(folder / "wide.jpg", 100).size == (100, 50)
    assert images.thumbnail(folder / "small.png", 100).size == (10, 20)
    images.save_resized(folder / "wide.jpg", folder / "out.jpg", 64)
    assert Image.open(folder / "out.jpg").size == (64, 32)
