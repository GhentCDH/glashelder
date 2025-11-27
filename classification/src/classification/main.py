import shutil
import json
import logging
from pathlib import Path
import pickle

import torch

from common.clipmodel import get_model, get_processor, get_device
from common.dataloader import get_dataloader
from common import PROJECT_ROOT

logger = logging.getLogger(__name__)

MAX_BATCHES = 50
DEV = True
CONFIDENCE_THRESHOLD = 0.4
MODEL = "openai/clip-vit-large-patch14"
CLASSIFICATION_CACHE_FILE = PROJECT_ROOT / "classification" / "data" / "classification_results.pkl"

def classify(image_dir):
    device = get_device()
    model = get_model(MODEL).to(device)
    processor = get_processor(MODEL)
    
    prompts = [
        "a photo of a human",
        "a photo of an animal",
        "a photo of a plant",
        "a photo of an object",
        "a photo of a landscape",
        "a photo of architecture or buildings",
        "a photo of an artwork"
    ]

    # Encode text once
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Load images
    image_paths = list(Path(image_dir).glob("*.jpg"))
    dataloader = get_dataloader(image_paths, processor, batch_size=256)
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
    
            if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
                break
    
            logger.info(f"Processing batch {batch_idx + 1} of {min(len(dataloader), MAX_BATCHES)}")
    
            paths = batch.pop("path")
            pixel_values = batch["pixel_values"].to(device)
    
            # Image embeddings
            image_embeds = model.get_image_features(pixel_values=pixel_values)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * (image_embeds @ text_embeds.T)
            probs = logits_per_image.softmax(dim=-1)
    
    
            labels = ["human", "animal", "plant", "object", "landscape", "architecture", "artwork", "ugly"]
            for path, p in zip(paths, probs):
                p = p.detach().cpu()
                
                # Get all labels above threshold
                above_threshold = [(labels[i], float(p[i])) for i in range(len(labels)) if float(p[i]) >= CONFIDENCE_THRESHOLD]
                
                if above_threshold:
                    # Sort by weight (highest first)
                    above_threshold.sort(key=lambda x: x[1], reverse=True)
                    assigned_labels = [label for label, weight in above_threshold]
                else:
                    assigned_labels = ["unsure"]
    
                results.append({
                    "path": path,
                    "predicted": {"labels": assigned_labels, "weights": {label: weight for label, weight in above_threshold} if above_threshold else {}},
                })
    
    with open(CLASSIFICATION_CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    
    # Create JSON output
    json_output = {}
    for idx, result in enumerate(results, 1):
        json_output[f"id_{idx}"] = {
            "path": result["path"],
            "classes": result["predicted"]["weights"]
        }
    
    with open(CLASSIFICATION_CACHE_FILE.with_suffix(".json"), "w") as f:
        json.dump(json_output, f, indent=2)
    
    logger.info("Classification complete. Results saved to classification_results.pkl and classification_results.json")
    logger.info("Now moving images to correct folders based on classification.")
    
    output_dirs = {
            "human": PROJECT_ROOT / "data" / "classification" / "human",
            "animal": PROJECT_ROOT / "data" / "classification" / "animal",
            "plant": PROJECT_ROOT / "data" / "classification" / "plant",
            "object": PROJECT_ROOT / "data" / "classification" / "object",
            "landscape": PROJECT_ROOT / "data" / "classification" / "landscape",
            "architecture": PROJECT_ROOT / "data" / "classification" / "architecture",
            "artwork": PROJECT_ROOT / "data" / "classification" / "artwork",
            "unsure": PROJECT_ROOT / "data" / "classification" / "unsure",
    }
    
    if DEV:
        for dir_path in output_dirs.values():
            if dir_path.exists():
                if dir_path.is_dir():
                    shutil.rmtree(dir_path)
                else:
                    dir_path.unlink()
            dir_path.mkdir(parents=True, exist_ok=True)
    else:
        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    for result in results:
        src_path = Path(result["path"])
        assigned_labels = result["predicted"]["labels"]
        
        # Copy image to all matching label folders
        for label in assigned_labels:
            dest_path = output_dirs[label] / src_path.name
            logger.info(f"Moving {src_path.name} to {label}")
            shutil.copy2(src_path, dest_path)
    
    logger.info("Images moved to respective folders.")
