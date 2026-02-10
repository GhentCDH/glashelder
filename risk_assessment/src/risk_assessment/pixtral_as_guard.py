from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
import torch
import os
import csv
import json
import time


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
    """Load the Pixtral model and processor with 8-bit quantization."""
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    model_id = "mistral-community/pixtral-12b"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='cuda',
        quantization_config=quantization_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def parse_output(output_text):
    """Parse the model output to extract JSON assessment."""
    try:
        # Extract JSON from the output - look for ```json blocks
        if "```json" in output_text:
            json_str = output_text.split("```json")[1].replace("```", "").strip()
        else:
            # Try to find JSON in the output - look for balanced braces
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
            # If parsing fails, try to extract key-value pairs manually
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


def assess_risk_of_image_folder(folder_path, output_csv="pixtral_risk_assessment_results.csv", prompt_file="lavaguard_prompt.txt"):
    """Process all images in a folder and save results to CSV."""
    model, processor = load_model()
    prompt = load_prompt(prompt_file)

    # Create CSV header
    header = ["image", "rating", "category", "rationale", "generation_time", "tokens_per_second"]

    results = []

    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                assessment, gen_time, tok_per_sec = process_image(image, model, processor, image_file, prompt)

                row = [
                    image_file,
                    assessment.get("rating", "Unknown"),
                    assessment.get("category", "NA: None applying"),
                    assessment.get("rationale", ""),
                    f"{gen_time:.3f}",
                    f"{tok_per_sec:.3f}"
                ]

                results.append(row)
                print(f"Processed: {image_file} - {assessment.get('rating', 'Unknown')} ({tok_per_sec:.2f} tok/s)")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results.append([image_file, "Error", "NA: None applying", str(e), "0", "0"])

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")
    print(f"Total images processed: {len(results)}")


def assess_risk_of_image(image_path, prompt_file="lavaguard_prompt.txt"):
    """Assess risk of a single image."""
    model, processor = load_model()
    prompt = load_prompt(prompt_file)

    image = Image.open(image_path).convert("RGB")
    assessment, gen_time, tok_per_sec = process_image(image, model, processor, os.path.basename(image_path), prompt)
    return assessment


def process_image(image, model, processor, image_name="", prompt=None):
    """Process a single image and return the risk assessment."""
    if prompt is None:
        prompt = load_prompt()
    
    # Prepare the prompt in Pixtral format
    PROMPT = f"<s>[INST]{prompt}\n[IMG][/INST]"
    
    inputs = processor(images=[image], text=PROMPT, return_tensors="pt").to("cuda")
    prompt_tokens = len(inputs['input_ids'][0])

    # Generate with timing
    t0 = time.time()
    generate_ids = model.generate(**inputs, max_new_tokens=512)
    t1 = time.time()
    
    total_time = t1 - t0
    generated_tokens = len(generate_ids[0]) - prompt_tokens
    time_per_token = generated_tokens / total_time if total_time > 0 else 0

    output_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(f"\nImage: {image_name}")
    print("-" * 50)
    print(f"Generated {generated_tokens} tokens in {total_time:.3f} s ({time_per_token:.3f} tok/s)")

    # Parse the output
    assessment = parse_output(output_text)
    print(f"Assessment: {assessment}")
    print("-" * 50)

    return assessment, total_time, time_per_token


if __name__ == "__main__":
    # Example usage
    assess_risk_of_image_folder("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop")
    # assess_risk_of_image("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/1c1tf36b1k_tiff.jpg")

