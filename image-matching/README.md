# image-matching

Image matching and artwork recognition using SIFT features and FLANN matching.

## Features

- **Image Stitching**: Create panoramas from multiple overlapping images
- **Artwork Recognition**: Identify paintings in photos by matching against a database

## Installation

```bash
uv sync
```

## Usage

### Artwork Recognition

Build a database of known paintings:

```bash
uv run --package image-matching image-matching build-db ./known_paintings ./paintings.db.pkl
```

Identify a painting in a photo:

```bash
uv run --package image-matching image-matching identify photo.jpg paintings.db.pkl --show-matches
```

Options:
- `--min-matches`: Minimum feature matches required (default: 10)
- `--min-inliers`: Minimum inliers after geometric verification (default: 4)
- `--max-results`: Maximum results to show (default: 5)
- `--no-verify`: Skip geometric verification (faster but less accurate)
- `--show-matches`: Visualize the feature matches

### Image Stitching

Stitch multiple images into a panorama:

```bash
uv run --package image-matching image-matching stitch img1.jpg img2.jpg img3.jpg
```

Options:
- `--size`: Max dimension for feature extraction (speeds up processing)
- `--max-canvas`: Max canvas size in pixels (default: 8000)
- `--max-stretch`: Max allowed geometric distortion (default: 3.0)

### Feature Visualization

View SIFT keypoints in an image:

```bash
uv run --package image-matching image-matching keypoints image.jpg
```

Match features between two images:

```bash
uv run --package image-matching image-matching match img1.jpg img2.jpg
```

## How It Works

### SIFT + FLANN for Artwork Recognition

The recognition system uses:

1. **SIFT (Scale-Invariant Feature Transform)**: Extracts robust local features that are:
   - Scale-invariant (works at different distances)
   - Rotation-invariant (handles tilted photos)
   - Illumination-robust (different lighting conditions)
   - Occlusion-resistant (partial views)

2. **FLANN (Fast Library for Approximate Nearest Neighbors)**: Efficiently matches features against large databases

3. **Homography Verification**: Uses RANSAC to verify geometric consistency and reject false matches

This approach is ideal for:
- Photos taken at angles or different scales
- Paintings behind glass (handles reflections)
- Partial views or crops
- Different lighting conditions
- Photos with people or objects in front

## Technical Details

- Uses OpenCV's SIFT implementation
- FLANN with KD-tree for fast matching
- Ratio test (0.7 threshold) to filter matches
- RANSAC-based homography verification
- Confidence scoring based on inlier ratio
