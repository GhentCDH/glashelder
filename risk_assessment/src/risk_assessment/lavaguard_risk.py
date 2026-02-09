from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import torch
import random
import numpy as np
import os
import csv
import json


def load_prompt(prompt_file="lavaguard_prompt.txt"):
    """Load the prompt from a text file."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, prompt_file)
    
    # Fallback to checking if it's provided as an absolute path
    if not os.path.exists(prompt_path) and os.path.exists(prompt_file):
        prompt_path = prompt_file
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_model():
    model_id = 'AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf'
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id)
    model.to('cuda:0')
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def parse_output(output_text):
    """Parse the model output to extract JSON assessment."""
    try:
        # The output includes the chat template, look for "assistant" followed by JSON
        assistant_idx = output_text.rfind('assistant')
        if assistant_idx != -1:
            # Search for JSON after "assistant"
            output_text = output_text[assistant_idx:]

        # Find JSON in the output - look for balanced braces
        start_idx = output_text.find('{')
        if start_idx == -1:
            return {
                "rating": "Unknown",
                "category": "NA: None applying",
                "rationale": output_text
            }

        # Count braces to find the matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(output_text)):
            if output_text[i] == '{':
                brace_count += 1
            elif output_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx == -1:
            return {
                "rating": "Unknown",
                "category": "NA: None applying",
                "rationale": output_text
            }

        json_str = output_text[start_idx:end_idx]

        # Try to parse the JSON
        try:
            assessment = json.loads(json_str)
            return assessment
        except json.JSONDecodeError:
            # If parsing fails, try to clean up the JSON string
            # Remove any trailing text after the closing brace
            json_str = json_str.strip()

            # Try using regex to extract key-value pairs manually
            import re
            rating_match = re.search(r'"rating"\s*:\s*"([^"]*)"', json_str)
            category_match = re.search(r'"category"\s*:\s*"([^"]*)"', json_str)
            rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', json_str)

            if rating_match and category_match:
                return {
                    "rating": rating_match.group(1),
                    "category": category_match.group(1),
                    "rationale": rationale_match.group(1) if rationale_match else ""
                }

            # Last resort: return raw output
            raise json.JSONDecodeError("Could not parse JSON", json_str, 0)

    except json.JSONDecodeError as e:
        return {
            "rating": "Error",
            "category": "NA: None applying",
            "rationale": f"JSON decode error: {str(e)}. Raw output: {output_text[:500]}"
        }
    except Exception as e:
        return {
            "rating": "Error",
            "category": "NA: None applying",
            "rationale": f"Error parsing output: {str(e)}. Raw output: {output_text[:500]}"
        }


def assess_risk_of_image_folder(folder_path, output_csv="lavaguard_risk_assessment_results.csv", prompt_file="lavaguard_prompt.txt"):
    model, processor = load_model()
    prompt = load_prompt(prompt_file)

    # Create CSV header
    header = ["image", "rating", "category", "rationale"]

    results = []

    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                assessment = process_image(image, model, processor, image_file, prompt)

                row = [
                    image_file,
                    assessment.get("rating", "Unknown"),
                    assessment.get("category", "NA: None applying"),
                    assessment.get("rationale", "")
                ]

                results.append(row)
                print(f"Processed: {image_file} - {assessment.get('rating', 'Unknown')}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results.append([image_file, "Error", "NA: None applying", str(e)])

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")
    print(f"Total images processed: {len(results)}")


def assess_risk_of_image(image_path, prompt_file="lavaguard_prompt.txt"):
    model, processor = load_model()
    prompt = load_prompt(prompt_file)

    image = Image.open(image_path).convert("RGB")
    assessment = process_image(image, model, processor, os.path.basename(image_path), prompt)
    return assessment


def process_image(image, model, processor, image_name="", prompt=None):
    if prompt is None:
        prompt = load_prompt()
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

    # Generation hyperparameters
    hyperparameters = {
        "max_new_tokens": 200,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "num_beams": 2,
        "use_cache": True,
    }

    with torch.inference_mode():
        output = model.generate(**inputs, **hyperparameters)

    output_text = processor.decode(output[0], skip_special_tokens=True)

    print(f"\nImage: {image_name}")
    print("-" * 50)

    # Parse the output
    assessment = parse_output(output_text)
    print(f"Assessment: {assessment}")
    print("-" * 50)

    return assessment


if __name__ == "__main__":
    # Example usage
    assess_risk_of_image_folder("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop")
    # assess_risk_of_image("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/0z70w1hx0d_tiff.jpg")