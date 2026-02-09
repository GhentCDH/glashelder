# Clipcluster - Image Matching and Clustering

Image clustering, search, and matching using CLIP embeddings with UMAP dimensionality reduction and HDBSCAN clustering.

## Features

- **Generate Embeddings**: Create CLIP, DINOv2, or Qwen3VL embeddings for image collections
- **Text Search**: Find images matching text descriptions
- **Image Matching**: Check if new images are already in your dataset
- **Clustering**: Group similar images automatically
- **Web Interface**: Interactive search UI

## Quick Start

### 1. Generate Embeddings

First, generate embeddings for your image collection:

```bash
uv run glashelder clipcluster generate-embeddings \
  --image-dir ./images \
  --save-path embeddings.pkl \
  --model clip
```

### 2. Match Images

Check if a new image is already in your dataset:

```bash
# Basic matching
uv run glashelder clipcluster match photo.jpg

# Strict duplicate detection
uv run glashelder clipcluster match photo.jpg --check-duplicate

# Find similar images with custom threshold
uv run glashelder clipcluster match photo.jpg --threshold 0.7 --top-k 10

# Save matches to directory
uv run glashelder clipcluster match photo.jpg --output-dir ./matches
```

### 3. Text Search

Search for images using text descriptions:

```bash
uv run glashelder clipcluster search "a dog in the park" --top-k 10
```

### 4. Start Web Interface

Launch the interactive search UI:

```bash
uv run glashelder clipcluster serve
```

Then open http://localhost:5000 in your browser.

