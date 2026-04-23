# Preprocessing - YOLO Dataset Creation & Training

Prepares datasets and trains YOLO object detection models for cropping glass plate photographs.

## Commands

### Create a YOLO dataset

Split LabelStudio ground truth annotations into train/validation sets:

```bash
uv run glashelder preprocessing create-yolo-dataset
```

### Train a YOLO model

Fine-tune a YOLO model on your custom dataset:

```bash
uv run glashelder preprocessing train-yolo-model
```

### Run preprocessing on images

Detect and crop objects from images using a trained YOLO model:

```bash
uv run glashelder preprocessing preprocessing-on-image
```
