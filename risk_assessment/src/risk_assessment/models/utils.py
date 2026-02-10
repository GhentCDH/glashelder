import json
import re
import os

def get_prompt(prompt_file="models/taxonomy.txt"):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def show_prompt(prompt_file="models/taxonomy.txt"):
    prompt = get_prompt(prompt_file)
    print(prompt)

def get_images(image_path):
    """Get a list of images from a folder or a single image path."""
    if os.path.isdir(image_path):
        images = []
        for file in os.listdir(image_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                images.append(os.path.join(image_path, file))
        return images
    elif os.path.isfile(image_path):
        return [image_path]
    else:
        raise ValueError(f"Invalid image path: {image_path}")


def parse_output(output_text, model_type="lava"):
    """Parse the model output to extract JSON assessment.
    
    Args:
        output_text: The output text to parse
        model_type: The model type - "lava", "pixtral", or "qwen"
    """
    try:
        json_str = None
        
        # Model-specific extraction logic
        if model_type.lower() == "pixtral":
            # Pixtral: Look for ```json blocks first
            if "```json" in output_text:
                json_str = output_text.split("```json")[1].replace("```", "").strip()
        
        elif model_type.lower() == "lava":
            # Lava: Search for JSON after "assistant" marker
            assistant_idx = output_text.rfind('assistant')
            if assistant_idx != -1:
                output_text = output_text[assistant_idx:]
        
        # Default extraction: find balanced JSON braces (used by lava, qwen, or fallback for pixtral)
        if not json_str:
            start_idx = output_text.find('{')
            if start_idx == -1:
                return {
                    "rating": "Unknown",
                    "category": "NA: None applying",
                    "rationale": output_text
                }

            # Count braces to find matching closing brace
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
            # Fallback: extract using regex
            rating_match = re.search(r'"rating"\s*:\s*"([^"]*)"', json_str)
            category_match = re.search(r'"category"\s*:\s*"([^"]*)"', json_str)
            rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', json_str)

            if rating_match and category_match:
                return {
                    "rating": rating_match.group(1),
                    "category": category_match.group(1),
                    "rationale": rationale_match.group(1) if rationale_match else ""
                }

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
