# Cluster - Image Clustering

Image clustering using embeddings with UMAP dimensionality reduction and HDBSCAN clustering.
Text search is `glashelder embed search` and `glashelder embed serve`.

## Features

- **Clustering**: Group similar images automatically
- **Web Interface**: Interactive search UI

## Quick Start

### 1. Generate Embeddings

First, generate embeddings for your image collection:

```bash
uv run glashelder embed images ./images --output embeddings.store --model clip
```


### 2. Start Web Interface

Launch the interactive search UI:

```bash
uv run glashelder cluster serve
```

Then open http://localhost:5000 in your browser.

