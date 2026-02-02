"""
Script to create YOLO training and validation dataset split.
Assumes images and labels are already in YOLO format.
"""

import logging
import random
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def create_yolo_dataset(ground_truth_folder, output_folder, train_split, seed):
    random.seed(seed)
    """Create YOLO dataset structure with train/val split."""
    # Ensure inputs are Path objects so Path methods work
    ground_truth_folder = Path(ground_truth_folder)
    output_folder = Path(output_folder)

    # Define paths
    images_dir = ground_truth_folder / "images"
    labels_dir = ground_truth_folder / "labels"
    classes_file = ground_truth_folder / "classes.txt"

    # Create output directories
    train_images_dir = output_folder / "images" / "train"
    val_images_dir = output_folder / "images" / "val"
    train_labels_dir = output_folder / "labels" / "train"
    val_labels_dir = output_folder / "labels" / "val"
    val_labels_dir = output_folder / "labels" / "val"

    for dir_path in [
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(images_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images")

    # Verify corresponding label files exist
    valid_pairs = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            logger.warning(f"No label file for {img_path.name}")

    logger.info(f"Found {len(valid_pairs)} valid image-label pairs")

    # Shuffle and split
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_split)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    logger.info("\nSplitting dataset:")
    logger.info(f"  Training: {len(train_pairs)} samples ({train_split * 100:.0f}%)")
    logger.info(
        f"  Validation: {len(val_pairs)} samples ({(1 - train_split) * 100:.0f}%)"
    )

    # Copy training files
    logger.info("\nCopying training files...")
    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, train_images_dir / img_path.name)
        shutil.copy2(label_path, train_labels_dir / label_path.name)

    # Copy validation files
    logger.info("Copying validation files...")
    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, val_images_dir / img_path.name)
        shutil.copy2(label_path, val_labels_dir / label_path.name)

    # Read class names
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    logger.info(f"\nClass names: {class_names}")

    # Create data.yaml for YOLO
    data_yaml = {
        "path": str(output_folder.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output_folder / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return data_yaml
