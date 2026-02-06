#!/usr/bin/env python3
"""
Script to create a copy of contro-set using the cropped versions from assets-crop.
This will copy the cropped variants of files from assets-crop that exist in contro-set.
"""

import os
import shutil
from pathlib import Path

# Define paths
CONTRO_SET_PATH = Path("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set")
ASSETS_CROP_PATH = Path("/mnt/UGent_Share/glashelder/assets-crop")
OUTPUT_PATH = Path("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop")

def main():
    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {OUTPUT_PATH}")

    # Get all files from contro-set
    contro_files = [f for f in CONTRO_SET_PATH.iterdir() if f.is_file()]
    print(f"\nFound {len(contro_files)} files in contro-set")

    # Counters
    copied = 0
    not_found = 0
    skipped = 0

    # Copy each file from assets-crop to the new directory
    for file_path in contro_files:
        filename = file_path.name

        # Skip system files
        if filename in ['Thumbs.db', 'image_id.txt']:
            print(f"Skipping system file: {filename}")
            skipped += 1
            continue

        source_file = ASSETS_CROP_PATH / filename
        dest_file = OUTPUT_PATH / filename

        # Check if the cropped version exists
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"✓ Copied: {filename}")
            copied += 1
        else:
            print(f"✗ Not found in assets-crop: {filename}")
            not_found += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files copied: {copied}")
    print(f"  Files not found: {not_found}")
    print(f"  Files skipped: {skipped}")
    print(f"  Output directory: {OUTPUT_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
