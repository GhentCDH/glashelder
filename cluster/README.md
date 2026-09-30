# Cluster - Image Clustering

Image clustering, search using embeddings with UMAP dimensionality reduction and HDBSCAN clustering.

## Features

- **Text Search**: Find images matching text descriptions
- **Clustering**: Group similar images automatically
- **Web Interface**: Interactive search UI

## Quick Start

### 1. Generate Embeddings

First, generate embeddings for your image collection:

```bash
uv run glashelder embed images ./images --output embeddings.store --model clip
```


### 2. Text Search

Search for images using text descriptions:

```bash
uv run glashelder cluster search "a dog in the park" --top-k 10
```

### 3. Start Web Interface

Launch the interactive search UI:

```bash
uv run glashelder cluster serve
```

Then open http://localhost:5000 in your browser.

