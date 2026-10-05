"""A PDF report of a clustering result: stats, plots and sample images per cluster.

Everything typst reads is made in a temporary directory; only the PDF is kept,
as `report.pdf` in the result.
"""

import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from embed import rebase_paths

from . import result
from .cluster import (
    calculate_cluster_statistics,
    create_2d_visualization,
    create_3d_visualization,
)
from .thumbs import samples, thumbnail

TEMPLATE = Path(__file__).parent / "templates" / "report.typ"
SAMPLE_SIZE = 512
READERS = 8

logger = logging.getLogger(__name__)


def build(path: Path, image_dir: str | None, *, count: int = 4) -> Path:
    """Write `report.pdf` into the clustering result at `path`, and return its path."""
    import typst

    clustering = result.load(path)
    meta = clustering.meta
    annotations = result.read_annotations(path)
    names = annotations.names
    images = rebase_paths(
        clustering.refs, image_dir if image_dir is not None else meta["root"]
    )
    persistence = meta.get("persistence", [])
    # images each cluster gets from noise with the thresholds saved in the viewer
    assigned = result.assign(clustering, annotations.soft)
    joined = Counter(assigned[(clustering.labels == -1) & (assigned >= 0)].tolist())

    logger.info("Calculating cluster statistics")
    stats = calculate_cluster_statistics(clustering.reduced, clustering.labels, None)

    with tempfile.TemporaryDirectory(prefix="cluster-report-") as directory:
        work = Path(directory)
        logger.info("Plotting")
        create_2d_visualization(
            clustering.coords2d, clustering.labels, str(work / "clusters_2d.png")
        )
        create_3d_visualization(
            clustering.coords3d, clustering.labels, str(work / "clusters_3d.png")
        )

        rows = {
            int(c["cluster_id"]): samples(clustering, int(c["cluster_id"]), count)
            for c in stats["clusters"]
        }
        wanted = [(label, row) for label, chosen in rows.items() for row in chosen]
        logger.info(f"Reading {len(wanted):,} sample images")

        def write(sample: tuple[int, int]) -> str | None:
            label, row = sample
            target = Path("samples") / str(label) / f"{row}.jpg"
            try:
                jpeg = thumbnail(images[row], SAMPLE_SIZE)
            except OSError as e:
                logger.warning(f"Could not read {images[row]}: {e}")
                return None
            (work / target).parent.mkdir(parents=True, exist_ok=True)
            (work / target).write_bytes(jpeg)
            return str(target)

        with ThreadPoolExecutor(READERS) as pool:
            written = dict(zip(wanted, pool.map(write, wanted), strict=True))

        def entry(cluster: dict) -> dict:
            label = int(cluster["cluster_id"])
            files = (written[(label, row)] for row in rows[label])
            return {
                **{k: v for k, v in cluster.items() if k != "cluster_id"},
                "id": label,
                "name": names.get(label, ""),
                "persistence": persistence[label] if label < len(persistence) else None,
                "soft": joined.get(label, 0),
                "samples": [file for file in files if file is not None],
            }

        clusters = [entry(c) for c in stats["clusters"]]
        size = {c["id"]: c["size"] for c in clusters}
        groups = [
            {
                "name": name,
                "clusters": [(m, names.get(m, "")) for m in members],
                "size": sum(size.get(m, 0) for m in members),
            }
            for name, members in sorted(annotations.groups.items())
        ]
        (work / "data.json").write_text(
            json.dumps(
                {
                    "meta": meta,
                    "overall": stats["overall"],
                    "clusters": clusters,
                    "groups": groups,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        shutil.copyfile(TEMPLATE, work / "report.typ")

        logger.info("Compiling the report")
        typst.compile(
            str(work / "report.typ"), output=str(work / "report.pdf"), root=str(work)
        )
        # through a temporary name next to it: the result is often on another disk
        target = Path(path) / result.REPORT
        temporary = target.with_name(f"{result.REPORT}.tmp")
        shutil.copyfile(work / "report.pdf", temporary)
        os.replace(temporary, target)

    return target
