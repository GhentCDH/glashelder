import shutil
import random
import argparse
from pathlib import Path


def copy_random_files(src: Path, dst: Path, n: int) -> None:
    files = [f for f in src.iterdir() if f.is_file()]
    if n > len(files):
        print(f"Requested {n} files but only {len(files)} available. Copying all.")
        n = len(files)

    dst.mkdir(parents=True, exist_ok=True)
    for f in random.sample(files, n):
        shutil.copy2(f, dst / f.name)
        print(f"Copied: {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy n random files from src to dst.")
    parser.add_argument("src", type=Path, help="Source folder")
    parser.add_argument("dst", type=Path, help="Destination folder")
    parser.add_argument("n", type=int, help="Number of files to copy")
    args = parser.parse_args()

    copy_random_files(args.src, args.dst, args.n)
