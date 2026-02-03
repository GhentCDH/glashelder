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
    
    # Policy names for the 3 outputs (should match how the model is configured)
    policy_names = ["Sexually Explicit content", "Dangerous content", "Violent content"]
    
    # Create CSV header
    header = ["filename"]
    for policy in policy_names:
        header.extend([
            f"{policy}_safe_probability",
            f"{policy}_violates_probability",
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
                    # Per HF docs: Index 0 = "Yes" (violates policy/UNSAFE), Index 1 = "No" (doesn't violate/SAFE)
                    violates_prob = policy_probs[0].item()  # Index 0 = Violates policy (unsafe)
                    safe_prob = policy_probs[1].item()      # Index 1 = Doesn't violate (safe)

                    # Decision: "Safe" if safe_prob > threshold, else "Unsafe"
                    decision = "Safe" if safe_prob > 0.80 else "Unsafe"

                    row.extend([
                        f"{safe_prob:.4f}",
                        f"{violates_prob:.4f}",
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

def assess_risk_of_image(image_path, use_default_policies=True):
    model, processor = load_model()
    image = Image.open(image_path)
    process_image(image, model, processor, os.path.basename(image_path), use_default_policies)

def process_image(image, model, processor, image_name="", use_default_policies=True):

    if use_default_policies:
        # Use default ShieldGemma policies
        policies = ["sexual", "dangerous", "violence"]
        inputs = processor(
            images=[image],
            policies=policies,
            return_tensors="pt",
        ).to(model.device)
        policy_names = ["Sexually Explicit content", "Dangerous content", "Violent content"]
    else:
        # Use custom policies
        custom_policies = {"Sexually Explicit content": "No Sexually Explicit content: The image shall not contain content that depicts explicit or graphic sexual acts (e.g., pornography, erotic nudity, depictions of rape or sexual assault).",
                         "Dangerous content": "No Dangerous Content: The image shall not contain content that facilitates or encourages activities that could cause real-world harm (e.g., building firearms and explosive devices, promotion of terrorism, instructions for suicide).",
                         "Violent content": "No Violence/Gore content: The image shall not contain content that depicts shocking, sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against animals, extreme injury or moment of death)."}
        policies = list(custom_policies.keys())
        inputs = processor(
            images=[image],
            custom_policies=custom_policies,
            policies=policies,
            return_tensors="pt",
        ).to(model.device)
        policy_names = list(custom_policies.keys())

    print(processor.batch_decode(inputs['input_ids'])[0])
    
    with torch.inference_mode():
        output = model(**inputs)
    
    print(f"Image: {image_name}")
    print(f"Raw probabilities: {output.probabilities}")

    outs = {}
    for idx, policy_probs in enumerate(output.probabilities.cpu()):
        # Per HF docs: Index 0 = "Yes" (violates policy/UNSAFE), Index 1 = "No" (doesn't violate/SAFE)
        violates_prob = policy_probs[0].item()  # Index 0 = Violates policy (unsafe)
        safe_prob = policy_probs[1].item()      # Index 1 = Doesn't violate (safe)

        # Decision: "Safe" if safe_prob > threshold, else "Unsafe"
        decision = "Safe" if safe_prob > 0.80 else "Unsafe"

        policy_name = policy_names[idx] if idx < len(policy_names) else f"Policy {idx}"
        outs[policy_name] = {
            "decision": decision,
            "safe_probability": f"{safe_prob:.4f}",
            "violates_probability": f"{violates_prob:.4f}"
        }
    
    print(outs)
    print("-" * 50)
    return outs


if __name__ == "__main__":
    #assess_risk_of_image_folder("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set")
    # Test with default policies first
    assess_risk_of_image("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set/test.png")