from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from pathlib import Path
from dataloader import get_dataloader
import pickle

MAX_BATCHES = 25
DEV = True
CONFIDENCE_THRESHOLD = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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
image_paths = list(Path("/mnt/UGent_Share/glashelder/assets").glob("*.jpg"))
dataloader = get_dataloader(image_paths, processor, batch_size=256)
results = []


model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):

        if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
            break

        print(f"Processing batch {batch_idx + 1} of {min(len(dataloader), MAX_BATCHES)}")

        paths = batch.pop("path")
        pixel_values = batch["pixel_values"].to(device)

        # Image embeddings
        image_embeds = model.get_image_features(pixel_values=pixel_values)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * (image_embeds @ text_embeds.T)
        probs = logits_per_image.softmax(dim=-1)


        labels = ["human", "animal", "plant", "object", "landscape", "architecture", "artwork"]
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

pickle.dump(results, open("classificationcluster/data/classification_results.pkl", "wb"))

# Create JSON output
import json
json_output = {}
for idx, result in enumerate(results, 1):
    image_name = Path(result["path"]).name
    json_output[f"id_{idx}"] = {
        "path": result["path"],
        "classes": result["predicted"]["weights"]
    }

with open("classificationcluster/data/classification_results.json", "w") as f:
    json.dump(json_output, f, indent=2)

print("Classification complete. Results saved to classification_results.pkl and classification_results.json")
print("Now moving images to correct folders based on classification.")

import shutil
from collections import defaultdict

output_dirs = {"human": Path("classificationcluster/data/mens"),
               "animal": Path("classificationcluster/data/dier"),
               "plant": Path("classificationcluster/data/plant"),
               "object": Path("classificationcluster/data/object"),
               "landscape": Path("classificationcluster/data/landschap"),
               "architecture": Path("classificationcluster/data/architectuur"),
               "artwork": Path("classificationcluster/data/kunstwerk"),
               "unsure": Path("classificationcluster/data/unsure")}

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
        print(f"Moving {src_path.name} to {label}")
        shutil.copy2(src_path, dest_path)

print("Images moved to respective folders.")



