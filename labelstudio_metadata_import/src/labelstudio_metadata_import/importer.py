"""Label Studio API client: upload images and import them as tasks, without
mounting any shared storage. Everything goes over HTTPS to the LS API.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
_SERVER_PATHS_RETRY_ATTEMPTS = 5
_SERVER_PATHS_RETRY_DELAY_SECONDS = 1.5


@dataclass
class HealthCheckResult:
    label: str
    ok: bool
    detail: str = ""


class LabelStudioImporter:
    def __init__(self, base_url: str, api_key: str, project_id: int):
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.headers = {"Authorization": f"Token {api_key}"}

    def upload_images(
        self,
        image_path: str | Path | list[str | Path],
        filenames: set[str] | None = None,
    ) -> dict[str, str]:
        """Upload images to Label Studio's file storage, without creating
        tasks (commit_to_project=false).

        IMAGE_PATH is either a folder, a single file, or an explicit list of
        file paths (the last avoids re-scanning a large/slow — e.g. network
        mounted — folder for every call, which matters when this is called
        once per checkpoint chunk).

        If FILENAMES is given, only files whose name is in FILENAMES are
        uploaded (checked against every form of IMAGE_PATH, including an
        explicit list) — use this to skip images that were filtered out
        (e.g. rated Safe by every model) instead of uploading everything
        passed in.

        One file per request: Label Studio's import endpoint only registers
        the first file when several are sent under the repeated "file" field
        of a single multipart request (confirmed against a live instance —
        sending 3 files in one POST only produced 1 file_upload_id) — batching
        here would silently drop most of the batch.

        Returns {original_filename: "/data/upload/{project}/{stored_name}"}.
        """
        if isinstance(image_path, list):
            files = [Path(p) for p in image_path]
        else:
            image_path = Path(image_path)
            if image_path.is_dir():
                files = sorted(
                    p for p in image_path.iterdir() if p.suffix.lower() in EXTENSIONS
                )
            elif image_path.is_file():
                files = [image_path]
            else:
                raise ValueError(f"Invalid image path: {image_path}")

        if filenames is not None:
            files = [p for p in files if p.name in filenames]

        if not files:
            raise ValueError(f"No images found in {image_path}")
        logger.info("Uploading %d image(s) from %s", len(files), image_path)

        for i, p in enumerate(files, start=1):
            with open(p, "rb") as fh:
                resp = requests.post(
                    f"{self.base_url}/api/projects/{self.project_id}/import",
                    headers=self.headers,
                    params={"commit_to_project": "false"},
                    files={"file": (p.name, fh, "application/octet-stream")},
                    timeout=300,
                )
            resp.raise_for_status()
            if i % 20 == 0 or i == len(files):
                logger.info("  uploaded %d/%d", i, len(files))

        expected = {p.name for p in files}
        return self._server_paths_covering(expected)

    def _server_paths_covering(self, expected: set[str]) -> dict[str, str]:
        """server_paths(), retried briefly if EXPECTED filenames we just
        uploaded haven't shown up yet — Label Studio can register a batch
        asynchronously, so an immediate GET can lag behind the POST that just
        succeeded.
        """
        paths = self.server_paths()
        for attempt in range(1, _SERVER_PATHS_RETRY_ATTEMPTS):
            if expected <= paths.keys():
                break
            missing = len(expected - paths.keys())
            logger.info(
                "%d of %d just-uploaded file(s) not visible yet, retrying (%d/%d)...",
                missing,
                len(expected),
                attempt,
                _SERVER_PATHS_RETRY_ATTEMPTS - 1,
            )
            time.sleep(_SERVER_PATHS_RETRY_DELAY_SECONDS)
            paths = self.server_paths()
        return paths

    def server_paths(self) -> dict[str, str]:
        """Fetch the stored filenames and build /data/upload/{project}/... paths.

        Label Studio prefixes uploads with a uuid, e.g.
        'upload/1/a1b2c3-plaat_0001.jpg'; that prefix is stripped so results can
        be matched back to the original filename. Follows pagination if the
        endpoint returns a paginated {"results": [...], "next": ...} envelope
        instead of a bare list.
        """
        paths: dict[str, str] = {}
        url = f"{self.base_url}/api/projects/{self.project_id}/file-uploads"
        params = {"all": "true", "page_size": 10000}

        while url:
            resp = requests.get(url, headers=self.headers, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            items = data if isinstance(data, list) else data.get("results", [])

            for item in items:
                stored = Path(item["file"]).name
                original = stored.split("-", 1)[-1]
                paths[original] = f"/data/upload/{self.project_id}/{stored}"

            url = data.get("next") if isinstance(data, dict) else None
            params = None  # 'next' already carries the full query string

        return paths

    def import_tasks(self, tasks: list[dict]) -> dict:
        resp = requests.post(
            f"{self.base_url}/api/projects/{self.project_id}/import",
            headers={**self.headers, "Content-Type": "application/json"},
            data=json.dumps(tasks, ensure_ascii=False).encode("utf-8"),
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> list[HealthCheckResult]:
        """Verify LS_URL is reachable, the token is valid, and the project is
        accessible — the same three calls the real import flow depends on.
        """
        results = []

        def get(label: str, path: str, **kwargs) -> requests.Response | None:
            try:
                resp = requests.get(
                    f"{self.base_url}{path}", headers=self.headers, timeout=15, **kwargs
                )
            except requests.exceptions.RequestException as exc:
                results.append(
                    HealthCheckResult(label, False, f"{type(exc).__name__}: {exc}")
                )
                return None
            if not resp.ok:
                results.append(
                    HealthCheckResult(
                        label, False, f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                )
                return None
            return resp

        who_resp = get("auth (whoami)", "/api/current-user/whoami")
        if who_resp is None:
            # server unreachable or token invalid — the remaining checks would
            # only fail the same way, so stop here.
            return results
        who = who_resp.json()
        results.append(
            HealthCheckResult(
                "auth (whoami)", True, who.get("email") or who.get("username") or ""
            )
        )

        project_resp = get(
            f"project {self.project_id}", f"/api/projects/{self.project_id}/"
        )
        if project_resp is not None:
            data = project_resp.json()
            results.append(
                HealthCheckResult(
                    f"project {self.project_id}",
                    True,
                    f"'{data.get('title')}', {data.get('task_number')} task(s)",
                )
            )

        uploads_resp = get(
            "file-uploads endpoint",
            f"/api/projects/{self.project_id}/file-uploads",
            params={"all": "true"},
        )
        if uploads_resp is not None:
            results.append(HealthCheckResult("file-uploads endpoint", True))

        return results


def csv_image_filenames(csv_path: str | Path, image_field: str = "image") -> set[str]:
    """Read just the IMAGE_FIELD column of a metadata CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row[image_field] for row in csv.DictReader(f)}


def build_tasks_from_rows(
    rows: list[dict], server_paths: dict[str, str], image_field: str = "image"
) -> tuple[list[dict], list[str]]:
    """Build Label Studio tasks from in-memory metadata rows, matching each
    row to an uploaded file by filename (the row's IMAGE_FIELD value).

    Rows are not mutated. Returns (tasks, filenames_missing_from_server_paths).
    """
    tasks = []
    missing = []

    for row in rows:
        data = dict(row)
        filename = data.pop(image_field)
        if filename not in server_paths:
            missing.append(filename)
            continue
        tasks.append({"data": {image_field: server_paths[filename], **data}})

    if missing:
        logger.warning(
            "%d row(s) had no matching uploaded file: %s", len(missing), missing[:10]
        )

    return tasks, missing


def build_tasks(
    csv_path: str | Path, server_paths: dict[str, str], image_field: str = "image"
) -> tuple[list[dict], list[str]]:
    """Build Label Studio tasks from a metadata CSV, matching each row to an
    uploaded file by filename (the CSV's IMAGE_FIELD column).

    Returns (tasks, filenames_missing_from_server_paths).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return build_tasks_from_rows(rows, server_paths, image_field)
