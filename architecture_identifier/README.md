# architecture_identifier

Segments buildings in images with [SAM3](https://huggingface.co/facebook/sam3)
(text-prompted "promptable concept segmentation").

`facebook/sam3` is a gated model: request access on Hugging Face and run
`hf auth login` (or set `HF_TOKEN`) before the first run.

```sh
uv run glashelder architecture_identifier segment IMAGES_DIR -o OUTPUT_DIR
# options: -p building -p church --threshold 0.5 --device cuda
```

For each image it writes to `OUTPUT_DIR`:

- `<name>_overlay.png`: the image with coloured masks, boxes and scores
- `<name>_masks/<i>.png`: one binary mask per detected building
- `<name>.json`: box, score, prompt and mask area per building
