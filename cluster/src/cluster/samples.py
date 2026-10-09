import logging
import shutil
from pathlib import Path

from common.images import save_resized

logger = logging.getLogger(__name__)


def export_samples(
    cluster_id,
    indices,
    image_paths,
    output_dir: Path,
    count: int,
    *,
    clean: bool = True,
) -> int:
    """Save up to `count` resized sample images of a cluster into `output_dir`.

    Returns the number of samples written. `output_dir` is the directory for this
    one cluster (or "noise"); it is cleared first when `clean` is set.
    """
    if clean:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = list(indices)[:count]
    for idx in sample_indices:
        src_path = image_paths[idx]
        dst_path = output_dir / Path(src_path).name
        save_resized(src_path, dst_path, 512)

    logger.info(
        f"Cluster {cluster_id}: {len(indices)} total images, "
        f"saved {len(sample_indices)} samples"
    )
    return len(sample_indices)
