#!/usr/bin/env python3
"""Upload the images listed in a metadata CSV to a Label Studio project.

Uses the same approach as `labelstudio_metadata_import sample` (branch
`labelstudi`): each image is uploaded to Label Studio's file storage, one file
per request, then referenced from a plain {"image": ...} task. That only works
for projects whose labeling config has a single data key (just $image).

Images are matched by pid: <pid>.jpg or <pid>_<suffix>.jpg (e.g. abc123_tiff.jpg).
Images already uploaded to the project are skipped, so re-running is safe.

Settings come from the environment / .env:
    LS_URL (default http://localhost:8080), LS_TOKEN, LS_PROJECT_ID, ASSETS_DIR
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}


def find_images(image_dir: Path, pids: list[str]) -> tuple[list[Path], list[str]]:
    """Match pids to files in image_dir (whole stem, or the part before the first '_')."""
    by_key: dict[str, Path] = {}
    for p in image_dir.iterdir():
        if p.suffix.lower() in EXTENSIONS:
            by_key.setdefault(p.stem, p)
            by_key.setdefault(p.stem.split("_", 1)[0], p)
    found, missing = [], []
    for pid in pids:
        (found if pid in by_key else missing).append(by_key.get(pid) or pid)
    return found, missing


def check(r: requests.Response) -> None:
    """raise_for_status, but show what Label Studio actually answered."""
    if not r.ok:
        sys.exit(
            f"HTTP {r.status_code} from {r.request.method} {r.url}\n{r.text[:1000]}"
        )


class LabelStudio:
    def __init__(self, url: str, token: str, project_id: int):
        self.url = url.rstrip("/")
        self.project_id = project_id
        self.headers = {"Authorization": f"Token {token}"}

    def uploaded(self) -> dict[str, str]:
        """{original filename: /data/upload/<project>/<stored name>} for existing uploads."""
        paths: dict[str, str] = {}
        url = f"{self.url}/api/projects/{self.project_id}/file-uploads"
        params = {"all": "true", "page_size": 10000}
        while url:
            r = requests.get(url, headers=self.headers, params=params, timeout=60)
            check(r)
            data = r.json()
            for item in data if isinstance(data, list) else data.get("results", []):
                stored = Path(item["file"]).name
                paths[stored.split("-", 1)[-1]] = (
                    f"/data/upload/{self.project_id}/{stored}"
                )
            url = data.get("next") if isinstance(data, dict) else None
            params = None
        return paths

    def upload(self, path: Path) -> None:
        # one file per request: LS only registers the first file of a multi-file POST
        with path.open("rb") as fh:
            r = requests.post(
                f"{self.url}/api/projects/{self.project_id}/import",
                headers=self.headers,
                params={"commit_to_project": "false"},
                files={"file": (path.name, fh, "application/octet-stream")},
                timeout=300,
            )
        check(r)

    def import_tasks(self, tasks: list[dict]) -> dict:
        r = requests.post(
            f"{self.url}/api/projects/{self.project_id}/import",
            headers={**self.headers, "Content-Type": "application/json"},
            data=json.dumps(tasks, ensure_ascii=False).encode("utf-8"),
            timeout=300,
        )
        check(r)
        return r.json()


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--csv",
        required=True,
        help="CSV with the images to upload (e.g. from filter_collections.py)",
    )
    p.add_argument(
        "--image-dir",
        default=os.getenv("ASSETS_DIR"),
        help="folder with the images (default: $ASSETS_DIR)",
    )
    p.add_argument(
        "--id-column",
        default="pid",
        help="CSV column matching the image filenames (default: %(default)s)",
    )
    p.add_argument(
        "--project-id",
        type=int,
        default=os.getenv("LS_PROJECT_ID"),
        help="default: $LS_PROJECT_ID",
    )
    p.add_argument("--ls-url", default=os.getenv("LS_URL", "http://localhost:8080"))
    p.add_argument(
        "--limit", type=int, help="only upload the first N images (for a test run)"
    )
    p.add_argument(
        "--embed",
        action="store_true",
        help="embed each image in its task as a base64 data: URI instead of using file upload "
        "(works for any labeling config, but makes tasks larger)",
    )
    p.add_argument(
        "--test", action="store_true", help="upload just one image (same as --limit 1)"
    )
    p.add_argument(
        "--tasks-json",
        default="taken.json",
        help="where the dry run writes its tasks (default: %(default)s)",
    )
    p.add_argument(
        "--commit", action="store_true", help="actually upload (default is a dry run)"
    )
    args = p.parse_args()

    if not args.image_dir:
        p.error("--image-dir or ASSETS_DIR is required")
    if args.project_id is None:
        p.error("--project-id or LS_PROJECT_ID is required")
    token = os.getenv("LS_TOKEN")  # env only, so it never lands in shell history
    if args.commit and not token:
        sys.exit("LS_TOKEN is not set")

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and args.id_column not in rows[0]:
        sys.exit(f"Column '{args.id_column}' not found in {args.csv}")
    pids = list(
        dict.fromkeys(
            r[args.id_column].strip() for r in rows if r[args.id_column].strip()
        )
    )

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        sys.exit(f"Image directory not found: {image_dir}")
    images, missing = find_images(image_dir, pids)
    print(
        f"{len(pids)} pids in CSV: {len(images)} found in {image_dir}, {len(missing)} missing"
    )
    if missing:
        print("  missing e.g.: " + ", ".join(missing[:10]))
    if args.test:
        args.limit = 1
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("Nothing to upload.")
        return 0

    ls = LabelStudio(args.ls_url, token or "", args.project_id)

    if not args.commit:
        tasks = [
            {"data": {"image": f"/data/upload/{args.project_id}/<uuid>-{p.name}"}}
            for p in images
        ]
        Path(args.tasks_json).write_text(
            json.dumps(tasks, indent=2, ensure_ascii=False)
        )
        print(
            f"Dry run: would upload {len(images)} image(s) to project {args.project_id}. "
            f"Sample tasks in {args.tasks_json}. Re-run with --commit."
        )
        return 0

    if args.embed:
        import base64
        import mimetypes

        tasks = []
        for path in images:
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            tasks.append(
                {
                    "data": {
                        "image": f"data:{mime};base64,{b64}",
                        "image_filename": path.name,
                    }
                }
            )
        ls.import_tasks(tasks)
        print(f"Created {len(tasks)} embedded task(s) in project {args.project_id}.")
        return 0

    existing = ls.uploaded()
    todo = [p for p in images if p.name not in existing]
    print(
        f"{len(images) - len(todo)} already in project {args.project_id}, uploading {len(todo)}"
    )
    for i, path in enumerate(todo, 1):
        ls.upload(path)
        if i % 20 == 0 or i == len(todo):
            print(f"  uploaded {i}/{len(todo)}")

    # LS may register uploads asynchronously; retry briefly until all are visible
    import time

    paths = ls.uploaded()
    for _ in range(5):
        if all(p.name in paths for p in images):
            break
        time.sleep(1.5)
        paths = ls.uploaded()

    tasks = [{"data": {"image": paths[p.name]}} for p in images if p.name in paths]
    lost = [p.name for p in images if p.name not in paths]
    if lost:
        print(
            f"WARNING: {len(lost)} upload(s) not visible in Label Studio: {lost[:10]}"
        )

    # Tasks for already-uploaded files may already exist; only create tasks for files uploaded now.
    new_names = {p.name for p in todo}
    tasks = [
        t
        for t, p in zip(tasks, [p for p in images if p.name in paths], strict=False)
        if p.name in new_names
    ]
    if tasks:
        ls.import_tasks(tasks)
    print(f"Created {len(tasks)} task(s) in project {args.project_id}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
