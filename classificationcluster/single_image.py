from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

image = Image.open("classificationcluster/data/0z70w1dc7f_tiff.jpg").convert("RGB")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

prompts = [
    "a photo of a person or a scene",
    "a photo of an animal, plant or object",
    "a photo of a landscape or topography"
]

inputs = processor(
    text=prompts,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

# Image-text similarity logits
logits_per_image = outputs.logits_per_image  # shape: (1, 3)
probs = logits_per_image.softmax(dim=-1)[0]

print("Person prob:   ", float(probs[0]))
print("Animal prob:   ", float(probs[1]))
print("Landscape prob:", float(probs[2]))
