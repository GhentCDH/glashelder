"""Checks for replacing files in one step."""

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from common.fs import atomic_write, atomic_writer


def test_writes_text_and_bytes():
    folder = Path(tempfile.mkdtemp())
    atomic_write(folder / "a.txt", "héllo")
    atomic_write(folder / "b.bin", b"\x00\x01")
    assert (folder / "a.txt").read_text(encoding="utf-8") == "héllo"
    assert (folder / "b.bin").read_bytes() == b"\x00\x01"
    assert sorted(p.name for p in folder.iterdir()) == ["a.txt", "b.bin"]


def test_replaces_an_existing_file():
    path = Path(tempfile.mkdtemp()) / "a.json"
    path.write_text("old")
    with atomic_writer(path) as file:
        json.dump({"new": True}, file)
    assert json.loads(path.read_text()) == {"new": True}


def test_failure_keeps_the_old_file_and_leaves_no_temporary():
    folder = Path(tempfile.mkdtemp())
    path = folder / "a.txt"
    path.write_text("old")
    with pytest.raises(RuntimeError), atomic_writer(path) as file:
        file.write("half")
        raise RuntimeError
    assert path.read_text() == "old"
    assert [p.name for p in folder.iterdir()] == ["a.txt"]


def test_concurrent_writers_do_not_collide():
    folder = Path(tempfile.mkdtemp())
    path = folder / "a.txt"
    texts = [str(n) * 10_000 for n in range(10)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda text: atomic_write(path, text), texts * 5))
    assert path.read_text() in texts
    assert [p.name for p in folder.iterdir()] == ["a.txt"]
