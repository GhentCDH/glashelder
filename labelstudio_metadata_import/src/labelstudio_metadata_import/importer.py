"""Label Studio API client: upload images and import them as tasks, without
mounting any shared storage. Everything goes over HTTPS to the LS API.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
_SERVER_PATHS_RETRY_ATTEMPTS = 5
_SERVER_PATHS_RETRY_DELAY_SECONDS = 1.5
_TASKS_PAGE_SIZE = 50

FILENAME_FIELD = "image_filename"


@dataclass
class HealthCheckResult:
    label: str
    ok: bool
    detail: str = ""


def resolve_image_files(
    image_path: str | Path | list[str | Path],
    filenames: set[str] | None = None,
) -> list[Path]:
    """Resolve IMAGE_PATH — a folder, a single file, or an explicit list of
    file paths (the last avoids re-scanning a large/slow, e.g. network
    mounted, folder once per checkpoint chunk) — to a list of Path objects.

    If FILENAMES is given, only files whose name is in FILENAMES are kept
    (checked against every form of IMAGE_PATH, including an explicit list).
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
    return files


def resolve_image_files_by_name(
    image_path: str | Path | list[str | Path],
    filenames: set[str] | None = None,
) -> dict[str, Path]:
    """resolve_image_files, keyed by filename for row-by-row lookup."""
    return {p.name: p for p in resolve_image_files(image_path, filenames)}


def image_to_data_uri(path: Path) -> str:
    """Read an image file and encode it as a data: URI, for embedding
    directly in a task's data instead of uploading it to Label Studio's file
    storage first. See build_tasks_from_paths for why.
    """
    mime_type, _ = mimetypes.guess_type(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


def build_tasks_from_paths(
    rows: list[dict],
    image_path: str | Path | list[str | Path],
    image_field: str = "image",
) -> tuple[list[dict], list[str]]:
    """Build Label Studio tasks from in-memory metadata ROWS, embedding each
    row's image (found under IMAGE_PATH, matched by the row's IMAGE_FIELD
    filename) as a base64 data: URI rather than referencing an uploaded
    file's storage path.

    This sidesteps a real Label Studio restriction: its "direct file upload"
    endpoint (commit_to_project=false, used to pre-upload a file so a task
    can later reference its storage path) outright refuses any project whose
    labeling config has more than one data key — which any project showing
    per-model ratings alongside the image necessarily has. Embedding the
    image inline instead never touches that endpoint, and works for any
    label config, at the cost of a larger request payload (base64 inflates
    size by ~33%) and every task's data permanently carrying its own image.

    Rows are not mutated. Each task's data also carries the original filename
    under FILENAME_FIELD — the data: URI itself has no filename in it, and
    without this a project's existing tasks are unmatchable against a local
    folder, making "skip what's already uploaded" impossible for anything
    imported before this field existed. See LabelStudioImporter.uploaded_filenames.

    Returns (tasks, filenames_missing_locally).
    """
    filenames = {row[image_field] for row in rows}
    resolved = resolve_image_files_by_name(image_path, filenames)

    tasks = []
    missing = []
    for row in rows:
        data = dict(row)
        filename = data.pop(image_field)
        path = resolved.get(filename)
        if path is None:
            missing.append(filename)
            continue
        tasks.append(
            {
                "data": {
                    image_field: image_to_data_uri(path),
                    FILENAME_FIELD: filename,
                    **data,
                }
            }
        )

    if missing:
        logger.warning(
            "%d row(s) had no matching local file: %s", len(missing), missing[:10]
        )

    return tasks, missing


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

        Only usable for a project whose labeling config has a single data
        key (e.g. just $image) — Label Studio rejects this endpoint outright
        for any project with more than one, such as anything showing
        per-model ratings alongside the image. Use build_tasks_from_paths
        (base64-embedded images) instead for those; this remains the more
        efficient path (no ~33% base64 inflation) for single-key projects
        like the `sample` command's.

        One file per request: Label Studio's import endpoint only registers
        the first file when several are sent under the repeated "file" field
        of a single multipart request (confirmed against a live instance —
        sending 3 files in one POST only produced 1 file_upload_id) — batching
        here would silently drop most of the batch.

        Returns {original_filename: "/data/upload/{project}/{stored_name}"}.
        """
        files = resolve_image_files(image_path, filenames)
        if not files:
            raise ValueError(f"No images found in {image_path}")
        logger.info("Uploading %d image(s): %s", len(files), [p.name for p in files])

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

    def task_count(self) -> int:
        """Total number of tasks in this project — one cheap request, used
        to size a progress bar around iter_tasks() without paginating
        through everything first.
        """
        resp = requests.get(
            f"{self.base_url}/api/projects/{self.project_id}/",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("task_number", 0)

    def iter_tasks(self, page_size: int = _TASKS_PAGE_SIZE):
        """Yield every task in this project, one at a time, paginated.

        Each task's data carries its full embedded image (there's no LS
        endpoint that lists tasks without it), so this is inherently
        expensive for a project of any size — see dedupe.py, which is the
        one place that should call this, and only deliberately (caching the
        result) rather than as a side effect of a normal upload run.

        PAGE_SIZE defaults small: a page full of base64-embedded images is a
        large response, and a too-large page_size has been observed to
        502 through Label Studio's proxy.
        """
        url = f"{self.base_url}/api/tasks"
        params = {"project": self.project_id, "page": 1, "page_size": page_size}

        while True:
            for attempt in range(1, _SERVER_PATHS_RETRY_ATTEMPTS + 1):
                try:
                    resp = requests.get(
                        url, headers=self.headers, params=params, timeout=120
                    )
                    resp.raise_for_status()
                    break
                except requests.exceptions.RequestException as exc:
                    if attempt == _SERVER_PATHS_RETRY_ATTEMPTS:
                        raise
                    logger.warning(
                        "Fetching page %d failed (%s), retrying (%d/%d)...",
                        params["page"],
                        exc,
                        attempt,
                        _SERVER_PATHS_RETRY_ATTEMPTS,
                    )
                    time.sleep(_SERVER_PATHS_RETRY_DELAY_SECONDS)

            data = resp.json()
            tasks = data.get("tasks", [])
            yield from tasks

            if len(tasks) < page_size:
                return
            params["page"] += 1

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
