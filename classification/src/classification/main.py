import shutil
import json
import logging
from pathlib import Path
import pickle

from common.clipmodel import get_model, get_processor, get_device
from common.dataloader import get_dataloader
from common import PROJECT_ROOT

logger = logging.getLogger(__name__)

def classify(image_dir, model="openai/clip-vit-large-patch14", max_batches=10, prompts : list = ['a photo of a human'], labels : list = ['human'], confidence=0.4 , output_folder=None):
    import torch

    device = get_device()
    model = get_model(model).to(device)
    processor = get_processor(model)

    if len(prompts) != len(labels):
        logger.error("Invalid configuration: number of prompts must match number of labels!")
        raise ValueError("Number of prompts must match number of labels!")

    # Encode text once
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Load images
    image_paths = list(Path(image_dir).glob("*.jpg"))
    dataloader = get_dataloader(image_paths, processor, batch_size=256, dataset_type="imagepath")
    results = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            logger.info(
                f"Processing batch {batch_idx + 1} of {min(len(dataloader), max_batches)}"
            )

            paths = batch.pop("path")
            pixel_values = batch["pixel_values"].to(device)

            # Image embeddings
            image_embeds = model.get_image_features(pixel_values=pixel_values)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * (image_embeds @ text_embeds.T)
            probs = logits_per_image.softmax(dim=-1)

            for path, p in zip(paths, probs):
                p = p.detach().cpu()
                
                # Debug: log the probabilities
                logger.debug(f"Image: {Path(path).name}, Probs: {[float(p[i]) for i in range(len(labels))]}")

                # Get all labels above threshold
                above_threshold = [
                    (labels[i], float(p[i]))
                    for i in range(len(labels))
                    if float(p[i]) >= confidence
                ]
                
                logger.debug(f"Above threshold ({confidence}): {above_threshold}")

                if above_threshold:
                    # Sort by weight (highest first)
                    above_threshold.sort(key=lambda x: x[1], reverse=True)
                    assigned_labels = [label for label, weight in above_threshold]
                else:
                    assigned_labels = ["not_classified"]
                
                logger.debug(f"Assigned labels: {assigned_labels}")

                results.append(
                    {
                        "path": path,
                        "predicted": {
                            "labels": assigned_labels,
                            "weights": {
                                label: weight for label, weight in above_threshold
                            }
                            if above_threshold
                            else {},
                        },
                    }
                )

    # Create JSON output
    json_output = {}
    for idx, result in enumerate(results, 1):
        json_output[f"id_{idx}"] = {
            "path": result["path"],
            "classes": result["predicted"]["weights"],
        }

    logger.info("Now moving images to correct folders based on classification.")

    # Ensure output directories exist and collect their paths
    output_dirs = {}
    for label in list(labels) + ["not_classified"]:
        if not output_folder:
            dest_dir = PROJECT_ROOT / "data" / "classification" / label
        else:
            dest_dir = Path(output_folder) / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[label] = dest_dir

    for result in results:
        src_path = Path(result["path"])
        assigned_labels = result["predicted"]["labels"]

        # If nothing was assigned, treat as not_classified
        if not assigned_labels:
            assigned_labels = ["not_classified"]

        # Copy image to all matching label folders
        for label in assigned_labels:
            dest_path = output_dirs.get(label, output_dirs["not_classified"]) / src_path.name
            logger.info(f"Moving {src_path.name} to {label}")
            shutil.copy2(src_path, dest_path)

    logger.info("Images moved to respective folders.")
