#!/usr/bin/env python
import json
from pathlib import Path

with open("./data/cluster_stats.json") as f:
    data = json.load(f)

image_paths: dict[str, list[str]] = {}
for cluster in data["clusters"]:
    key = str(cluster["cluster_id"])
    cluster_dir = Path("data").resolve() / "cluster_samples" / f"cluster_{key}"

    # Get all image files, sorted for consistency
    files_in_dir = sorted(
        [
            str(Path("data") / "cluster_samples" / f"cluster_{key}" / file.name)
            for file in cluster_dir.glob("*")
            if file.is_file()
        ]
    )

    image_paths[key] = files_in_dir

with open("cluster_images.json", "w") as f:
    json.dump(image_paths, f, indent=2)
