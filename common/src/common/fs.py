"""Writing files so that readers never see half of one.

Everything is written to a temporary file next to the target and moved over it
in one step. Each write gets its own temporary name, so threads that write the
same file at once (a threaded Flask app saving annotations) do not collide.
"""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

# read once: asking for the umask means changing it, which is not thread safe
_UMASK = os.umask(0)
os.umask(_UMASK)


@contextmanager
def atomic_writer(path: str | Path, *, binary: bool = False) -> Generator[IO[Any]]:
    """A file to write `path` through, for writers that stream (`json.dump`,
    `np.savez`, ...). It replaces `path` when the block ends without an error,
    and is removed otherwise. Text is written as UTF-8."""
    path = Path(path)
    file = tempfile.NamedTemporaryFile(
        "wb" if binary else "w",
        encoding=None if binary else "utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with file:
            yield file
        # the same permissions a plain open() would have given
        os.chmod(file.name, 0o666 & ~_UMASK)
        os.replace(file.name, path)
    except BaseException:
        Path(file.name).unlink(missing_ok=True)
        raise


def atomic_write(path: str | Path, data: bytes | str) -> None:
    """Replace `path` with `data` in one step. Text is written as UTF-8."""
    with atomic_writer(path, binary=isinstance(data, bytes)) as file:
        file.write(data)
