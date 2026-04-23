# Embed

Generates image embedding vectors and provides a web interface for searching through them.

Supports CLIP (default), DINOv2 (no text support), OpenCLIP, and Qwen3-VL (~16 GB VRAM).

## Commands

```bash
# generate embeddings for a folder of images
uv run glashelder embed generate-embeddings --image-dir ./images --save-path embeddings.pkl

# start the search web interface
uv run glashelder embed serve
```

The web interface supports text-to-image search, image-to-image search, and similarity browsing.
