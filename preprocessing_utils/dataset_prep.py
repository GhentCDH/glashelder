"""
Script to create YOLO training and validation dataset split.
Assumes images and labels are already in YOLO format.
"""

from pathlib import Path
import shutil
import random
import yaml

# Configuration
GT_DIR = Path("preprocessing_utils/data/GT")
OUTPUT_DIR = Path("preprocessing_utils/data/training_data")
TRAIN_SPLIT = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

def create_yolo_dataset():
    """Create YOLO dataset structure with train/val split."""
    
    # Define paths
    images_dir = GT_DIR / "images"
    labels_dir = GT_DIR / "labels"
    classes_file = GT_DIR / "classes.txt"
    
    # Create output directories
    train_images_dir = OUTPUT_DIR / "images" / "train"
    val_images_dir = OUTPUT_DIR / "images" / "val"
    train_labels_dir = OUTPUT_DIR / "labels" / "train"
    val_labels_dir = OUTPUT_DIR / "labels" / "val"
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list(images_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} images")
    
    # Verify corresponding label files exist
    valid_pairs = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            print(f"Warning: No label file for {img_path.name}")
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    
    # Shuffle and split
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * TRAIN_SPLIT)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"\nSplitting dataset:")
    print(f"  Training: {len(train_pairs)} samples ({TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation: {len(val_pairs)} samples ({(1-TRAIN_SPLIT)*100:.0f}%)")
    
    # Copy training files
    print("\nCopying training files...")
    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, train_images_dir / img_path.name)
        shutil.copy2(label_path, train_labels_dir / label_path.name)
    
    # Copy validation files
    print("Copying validation files...")
    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, val_images_dir / img_path.name)
        shutil.copy2(label_path, val_labels_dir / label_path.name)
    
    # Read class names
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"\nClass names: {class_names}")
    
    # Create data.yaml for YOLO
    data_yaml = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return data_yaml

if __name__ == "__main__":
    create_yolo_dataset()
