from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from PIL import Image
import torch
import random
import numpy as np
import os
import csv


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model():
    set_seed(42)
    model_id = "google/shieldgemma-2-4b-it"
    model = ShieldGemma2ForImageClassification.from_pretrained(model_id).to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def assess_risk_of_image_folder(folder_path, output_csv="risk_assessment_results.csv"):
    model, processor = load_model()
    
    # Policy names for the 3 outputs
    policy_names = ["Sexually Explicit", "Dangerous Content", "Violence/Gore"]
    
    # Create CSV header
    header = ["filename"]
    for policy in policy_names:
        header.extend([
            f"{policy}_yes_probability",
            f"{policy}_no_probability",
            f"{policy}_decision"
        ])
    
    results = []
    
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                
                inputs = processor(
                    images=[image],
                    return_tensors="pt",
                ).to(model.device)
                
                with torch.inference_mode():
                    output = model(**inputs)
                
                # Build row for this image
                row = [image_file]
                
                for idx, policy_probs in enumerate(output.probabilities.cpu()):
                    yes_prob = policy_probs[0].item()  # Index 0 = "Yes"
                    no_prob = policy_probs[1].item()   # Index 1 = "No"
                    
                    # Determine Yes/No based on threshold
                    decision = "Yes" if yes_prob > 0.80 else "No"
                    
                    row.extend([
                        f"{yes_prob:.4f}",
                        f"{no_prob:.4f}",
                        decision
                    ])
                
                results.append(row)
                print(f"Processed: {image_file}")
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(results)
    
    print(f"\nResults saved to {output_csv}")
    print(f"Total images processed: {len(results)}")

def assess_risk_of_image(image_path):
    model, processor = load_model()
    image = Image.open(image_path)
    process_image(image, model, processor, os.path.basename(image_path))

def process_image(image, model, processor, image_name=""):
    inputs = processor(
        images=[image],
        return_tensors="pt",
    ).to(model.device)
    
    with torch.inference_mode():
        output = model(**inputs)
    
    print(f"Image: {image_name}")
    print(f"Raw probabilities: {output.probabilities}")
    
    # Policy names for the 3 outputs
    policy_names = ["No Sexually Explicit", "No Dangerous Content", "No Violence/Gore"]
    
    outs = {}
    for idx, policy_probs in enumerate(output.probabilities.cpu()):
        yes_prob = policy_probs[0].item()  # Index 0 = "Yes"
        no_prob = policy_probs[1].item()   # Index 1 = "No"
        
        # Determine Yes/No based on threshold
        decision = "Yes" if yes_prob > 0.80 else "No"
        
        policy_name = policy_names[idx] if idx < len(policy_names) else f"Policy {idx}"
        outs[policy_name] = {
            "decision": decision,
            "yes_probability": f"{yes_prob:.4f}",
            "no_probability": f"{no_prob:.4f}"
        }
    
    print(outs)
    print("-" * 50)
    return outs


if __name__ == "__main__":
    assess_risk_of_image_folder("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set")