# OCR

Extracts and scores text from the photos and labels on the glass plates.

## Commands

```bash
# run OCR on a single image
uv run glashelder ocr one --path ./images/plate.jpg

# run OCR on a folder of images
uv run glashelder ocr many --path ./images

# hand-transcribe images in an editor, pre-filled by the VLM
uv run glashelder ocr annotate

# check transcriptions against stored annotations
uv run glashelder ocr validate

# recompute a report at a new --tau from stored similarities
uv run glashelder ocr rescore
```
