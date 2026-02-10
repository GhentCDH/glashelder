from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from PIL import Image
import requests
import torch

model_id = "google/shieldgemma-2-4b-it"

image = Image.open("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/sn01004006_tiff.jpg")

model = ShieldGemma2ForImageClassification.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)


custom_policies = {
    "Hate, Humiliation, Harassment": 
        "The image shall not contain content that expresses hateful, derogatory, or demeaning sentiments toward individuals or groups based on protected characteristics, nor encourage discrimination or harassment against such groups.",

    "Violence": 
        "The image shall not contain shocking, sensational, or gratuitous violence, including excessive blood or gore, extreme injury, or graphic depictions of death.",

    "Nudity Content": 
        "The image shall not contain visible human genitalia or explicit nudity, including exposed breasts, nipples, vagina, vulva, or penis, nor depict nudity in a sexualized manner.",

    "Ethnographic Content": 
        "The image shall not contain imagery intended for or suitable for ethnographic classification or analysis of people based on cultural, ethnic, or social characteristics."
}

policies = list(custom_policies.keys())

inputs = processor(
      images=[image],
      custom_policies=custom_policies,
      policies=policies,
      return_tensors="pt",
  ).to(model.device)


with torch.inference_mode():
    scores = model(**inputs)

print(scores.probabilities)
print(f"\n{'='*60}")
print("Safety Assessment Results")
print(f"{'='*60}\n")

outs = {}
for idx, policy in enumerate(scores.probabilities.cpu()):
  yes_prob = policy[0]
  no_prob = policy[1]

  outs[f"Yes for {policies[idx]}"] = yes_prob
  outs[f"No for {policies[idx]}"] = no_prob

print(outs)

print(outs)