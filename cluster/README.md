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

It opens the link it logs (`http://localhost:5002/?secret=...`) in your browser.

