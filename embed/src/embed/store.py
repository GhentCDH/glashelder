"""Embeddings on disk as flat files, keyed by image reference, safe to extend.

A store is a directory:

    meta.json     model, model_id, dim, dtype and the image root
    vectors.bin   N x dim matrix, one fixed-size row per image
    hashes.bin    N x 16 bytes: a hash of the image bytes each row was made from
    refs.txt      one reference per line, line i belongs to row i

The matrix is memory-mapped, so opening a store reads nothing up front and the OS
keeps as much of it in RAM as fits. New rows are appended with their ref line
written last, which is what commits a row: a run that dies halfway leaves at most
a partial tail, which the next open cuts off. Changed rows are overwritten in
place, so row indices never move (labeling sessions refer to them), except when
pruning.

Local images are stored relative to `root`, so a store made on one machine works
on another; URLs are stored as they are.
"""

import json
import logging
import math
import os
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np

META, VECTORS, HASHES, REFS = "meta.json", "vectors.bin", "hashes.bin", "refs.txt"
VERSION = 1
HASH_BYTES = 16
UNKNOWN = bytes(HASH_BYTES)
"""What `hashes.bin` holds for a row whose image bytes were never hashed."""
PRUNE_CHUNK = 65_536

logger = logging.getLogger(__name__)


def is_store(path: str | Path) -> bool:
    return (Path(path) / META).is_file()


class Store:
    """One embedding store on disk. Cheap to open; nothing is held but the refs."""

    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self._refs = self._recover()
        self._rows: dict[str, int] | None = None

    @classmethod
    def open(cls, path: str | Path) -> "Store":
        path = Path(path)
        if not is_store(path):
            raise ValueError(f"{path} is not an embedding store")
        meta = json.loads((path / META).read_text())
        if meta.get("version") != VERSION:
            raise ValueError(f"{path}: unsupported store version {meta.get('version')}")
        return cls(path, meta)

    @classmethod
    def create_or_open(
        cls,
        path: str | Path,
        *,
        model: str,
        model_id: str,
        root: str | Path | None = None,
        dtype: str = "float16",
    ) -> "Store":
        """Open the store at `path`, or start an empty one there.

        An existing store must have been made by the same model: mixing vectors
        from two models in one matrix would make every similarity meaningless.
        """
        path = Path(path)
        if is_store(path):
            store = cls.open(path)
            if store.model_id != model_id:
                raise ValueError(
                    f"{path} holds {store.model_id} embeddings, not {model_id}"
                )
            return store
        if path.exists() and (path.is_file() or any(path.iterdir())):
            raise ValueError(f"{path} exists and is not an embedding store")

        path.mkdir(parents=True, exist_ok=True)
        for name in (VECTORS, HASHES, REFS):
            (path / name).touch()
        meta = {
            "version": VERSION,
            "model": model,
            "model_id": model_id,
            "dim": None,
            "dtype": np.dtype(dtype).name,
            "root": str(root) if root is not None else None,
        }
        store = cls(path, meta)
        store._write_meta()
        return store

    @property
    def model(self) -> str:
        return self.meta["model"]

    @property
    def model_id(self) -> str:
        return self.meta["model_id"]

    @property
    def dim(self) -> int | None:
        return self.meta["dim"]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.meta["dtype"]).newbyteorder("<")

    @property
    def root(self) -> str | None:
        return self.meta["root"]

    def __len__(self) -> int:
        return len(self._refs)

    @property
    def nbytes(self) -> int:
        """Size of the matrix, which is what has to fit in RAM or VRAM."""
        return len(self) * (self.dim or 0) * self.dtype.itemsize

    def refs(self) -> list[str]:
        return self._refs

    def hashes(self) -> dict[str, str | None]:
        """Every ref with the hash of the image it was embedded from, if known."""
        raw = (self.path / HASHES).read_bytes()
        rows = (raw[i * HASH_BYTES : (i + 1) * HASH_BYTES] for i in range(len(self)))
        return {
            ref: None if row == UNKNOWN else row.hex()
            for ref, row in zip(self._refs, rows, strict=True)
        }

    def vectors(self) -> np.ndarray:
        """The matrix, memory-mapped copy-on-write: writes never reach the file."""
        if not self._refs or self.dim is None:
            return np.empty((0, self.dim or 0), dtype=self.dtype)
        return np.memmap(
            self.path / VECTORS, dtype=self.dtype, mode="c", shape=(len(self), self.dim)
        )

    def put(
        self, refs: Sequence[str], hashes: Sequence[str | None], vectors: np.ndarray
    ) -> None:
        """Store vectors for refs: new refs are appended, known ones overwritten in place."""
        vectors = np.ascontiguousarray(vectors, dtype=self.dtype)
        if vectors.ndim != 2 or not len(vectors) == len(refs) == len(hashes):
            raise ValueError("refs, hashes and vectors must line up")
        if self.dim is None:
            self.meta["dim"] = int(vectors.shape[1])
            self._write_meta()
        elif vectors.shape[1] != self.dim:
            raise ValueError(
                f"vectors have dim {vectors.shape[1]}, store has {self.dim}"
            )
        if bad := next((ref for ref in refs if "\n" in ref), None):
            raise ValueError(f"a ref cannot contain a newline: {bad!r}")

        rows = self._index()
        appended: list[int] = []
        updated: list[tuple[int, int]] = []
        for i, ref in enumerate(refs):
            if ref in rows:
                updated.append((rows[ref], i))
            else:
                rows[ref] = len(self._refs) + len(appended)
                appended.append(i)

        if appended:
            # ref lines last: a row only exists once its ref line does
            with (self.path / VECTORS).open("ab") as file:
                file.write(vectors[appended].tobytes())
            with (self.path / HASHES).open("ab") as file:
                file.write(b"".join(_hash_bytes(hashes[i]) for i in appended))
            with (self.path / REFS).open("a", encoding="utf-8") as file:
                file.write("".join(f"{refs[i]}\n" for i in appended))
            self._refs.extend(refs[i] for i in appended)

        if updated:
            row_bytes = self.meta["dim"] * self.dtype.itemsize
            with (
                (self.path / VECTORS).open("r+b") as vector_file,
                (self.path / HASHES).open("r+b") as hash_file,
            ):
                for row, i in updated:
                    vector_file.seek(row * row_bytes)
                    vector_file.write(vectors[i].tobytes())
                    hash_file.seek(row * HASH_BYTES)
                    hash_file.write(_hash_bytes(hashes[i]))

    def put_hashes(self, hashes: Mapping[str, str]) -> None:
        """Record hashes for rows that already exist, leaving their vectors alone."""
        if not hashes:
            return
        rows = self._index()
        with (self.path / HASHES).open("r+b") as file:
            for ref, digest in hashes.items():
                file.seek(rows[ref] * HASH_BYTES)
                file.write(_hash_bytes(digest))

    def prune(self, keep: Collection[str]) -> int:
        """Drop every row whose ref is not in `keep`. Shifts the rows after it."""
        kept = [i for i, ref in enumerate(self._refs) if ref in keep]
        removed = len(self._refs) - len(kept)
        if not removed:
            return 0

        vectors = self.vectors()
        hashes = (self.path / HASHES).read_bytes()
        with (
            (self.path / f"{VECTORS}.tmp").open("wb") as vector_file,
            (self.path / f"{HASHES}.tmp").open("wb") as hash_file,
        ):
            # in chunks: the matrix may be larger than RAM
            for start in range(0, len(kept), PRUNE_CHUNK):
                chunk = kept[start : start + PRUNE_CHUNK]
                vector_file.write(np.asarray(vectors[chunk]).tobytes())
                hash_file.write(
                    b"".join(
                        hashes[i * HASH_BYTES : (i + 1) * HASH_BYTES] for i in chunk
                    )
                )
        del vectors
        refs = [self._refs[i] for i in kept]
        (self.path / f"{REFS}.tmp").write_text(
            "".join(f"{ref}\n" for ref in refs), encoding="utf-8"
        )
        for name in (VECTORS, HASHES, REFS):
            os.replace(self.path / f"{name}.tmp", self.path / name)

        self._refs, self._rows = refs, None
        return removed

    def _index(self) -> dict[str, int]:
        if self._rows is None:
            self._rows = {ref: i for i, ref in enumerate(self._refs)}
        return self._rows

    def _write_meta(self) -> None:
        temporary = self.path / f"{META}.tmp"
        temporary.write_text(json.dumps(self.meta, indent=2))
        os.replace(temporary, self.path / META)

    def _recover(self) -> list[str]:
        """The committed refs, cutting off whatever an interrupted write left behind."""
        *refs, tail = (self.path / REFS).read_text(encoding="utf-8").split("\n")
        partial = bool(tail)  # a line without its newline was never committed
        row_bytes = (self.dim or 0) * self.dtype.itemsize
        vector_rows = (
            (self.path / VECTORS).stat().st_size // row_bytes if row_bytes else 0
        )
        hash_rows = (self.path / HASHES).stat().st_size // HASH_BYTES
        count = min(len(refs), vector_rows, hash_rows)

        if count < len(refs) or partial:
            dropped = len(refs) - count + partial
            logger.warning(
                f"{self.path}: dropping {dropped} rows an interrupted write left unfinished"
            )
            refs = refs[:count]
            (self.path / REFS).write_text(
                "".join(f"{ref}\n" for ref in refs), encoding="utf-8"
            )
        # only write when there is something to repair: readers may lack write access
        for name, size in ((VECTORS, count * row_bytes), (HASHES, count * HASH_BYTES)):
            if (self.path / name).stat().st_size != size:
                os.truncate(self.path / name, size)
        return refs


def format_bytes(size: float) -> str:
    power = min(3, int(math.log10(size) // 3)) if size >= 1 else 0
    unit = ("B", "KB", "MB", "GB")[power]
    return f"{size / 1000**power:.1f} {unit}" if power else f"{size:.0f} B"


def _hash_bytes(digest: str | None) -> bytes:
    return bytes.fromhex(digest) if digest else UNKNOWN
