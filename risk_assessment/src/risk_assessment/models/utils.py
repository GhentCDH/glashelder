import json
import re
import os
import csv
from collections import Counter

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


def write_combined_csv(assessments_lavaguard, assessments_qwen, assessments_pixtral, output_csv="combined_risk_assessment_results.csv"):
    """Write combined assessment results from all three models to a single CSV.
    
    Args:
        assessments_lavaguard: List of assessment dictionaries from LavaGuard
        assessments_qwen: List of assessment dictionaries from Qwen3
        assessments_pixtral: List of assessment dictionaries from Pixtral
        output_csv: Path to the output CSV file
    """
    # Create a dictionary to group assessments by image name
    combined = {}
    
    for assessment in assessments_lavaguard:
        image_name = assessment.get("image", "")
        if image_name not in combined:
            combined[image_name] = {"image": image_name}
        combined[image_name]["lavaguard_rating"] = assessment.get("rating", "Unknown")
        combined[image_name]["lavaguard_category"] = assessment.get("category", "NA: None applying")
        combined[image_name]["lavaguard_rationale"] = assessment.get("rationale", "")
    
    for assessment in assessments_qwen:
        image_name = assessment.get("image", "")
        if image_name not in combined:
            combined[image_name] = {"image": image_name}
        combined[image_name]["qwen_rating"] = assessment.get("rating", "Unknown")
        combined[image_name]["qwen_category"] = assessment.get("category", "NA: None applying")
        combined[image_name]["qwen_rationale"] = assessment.get("rationale", "")
    
    for assessment in assessments_pixtral:
        image_name = assessment.get("image", "")
        if image_name not in combined:
            combined[image_name] = {"image": image_name}
        combined[image_name]["pixtral_rating"] = assessment.get("rating", "Unknown")
        combined[image_name]["pixtral_category"] = assessment.get("category", "NA: None applying")
        combined[image_name]["pixtral_rationale"] = assessment.get("rationale", "")
    
    # Calculate majority voting for each image
    for image_name, row in combined.items():
        ratings = [
            row.get("lavaguard_rating", "Unknown"),
            row.get("qwen_rating", "Unknown"),
            row.get("pixtral_rating", "Unknown")
        ]
        
        # Count occurrences of each rating
        rating_counts = Counter(ratings)
        most_common = rating_counts.most_common()
        
        # Get the majority vote (most common rating)
        if most_common:
            majority_rating, count = most_common[0]
            # If there's a tie (all different or 2-way tie), note it
            if len(most_common) == 3:  # All three different
                row["majority_vote"] = "No Consensus (all different)"
            elif len(most_common) > 1 and most_common[0][1] == most_common[1][1]:  # Tie between top ratings
                row["majority_vote"] = f"Tie: {most_common[0][0]} / {most_common[1][0]}"
            else:
                row["majority_vote"] = majority_rating
        else:
            row["majority_vote"] = "Unknown"
    
    # Define CSV header
    header = [
        "image",
        "majority_vote",
        "lavaguard_rating", "lavaguard_category", "lavaguard_rationale",
        "qwen_rating", "qwen_category", "qwen_rationale",
        "pixtral_rating", "pixtral_category", "pixtral_rationale"
    ]
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        for row in combined.values():
            writer.writerow(row)
    
    print(f"\nCombined results saved to {output_csv}")
    print(f"Total images processed: {len(combined)}")
