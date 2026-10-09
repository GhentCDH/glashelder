import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

_MODELS_DIR = Path(__file__).parent

# combined results of a run, relative to the working directory; also what `serve` reads
RESULTS_CSV = Path("combined_risk_assessment_results.csv")


def get_prompt(prompt_file=None):
    if prompt_file is None:
        prompt_file = _MODELS_DIR / "taxonomy.txt"
    with open(prompt_file, encoding="utf-8") as f:
        return f.read()


def show_prompt(prompt_file=None):
    prompt = get_prompt(prompt_file)
    print(prompt)


def get_images(image_path):
    """Get a list of images from a folder or a single image path."""
    if os.path.isdir(image_path):
        images = []
        for file in os.listdir(image_path):
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            ):
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

        # Default extraction: find balanced JSON braces (used by lava, qwen, or fallback for pixtral)
        if not json_str:
            start_idx = output_text.find("{")
            if start_idx == -1:
                return {
                    "rating": "Unknown",
                    "category": "NA: None applying",
                    "rationale": output_text,
                }

            # Count braces to find matching closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(output_text)):
                if output_text[i] == "{":
                    brace_count += 1
                elif output_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx == -1:
                return {
                    "rating": "Unknown",
                    "category": "NA: None applying",
                    "rationale": output_text,
                }

            json_str = output_text[start_idx:end_idx]

        # Ensure JSON string is closed if the model omitted trailing braces
        if json_str:
            brace_balance = 0
            for ch in json_str:
                if ch == "{":
                    brace_balance += 1
                elif ch == "}":
                    brace_balance -= 1
            if brace_balance > 0:
                json_str = json_str + ("}" * brace_balance)

        # Try to parse the JSON
        try:
            assessment = json.loads(json_str)
            if isinstance(assessment, dict):
                return assessment
            return {
                "rating": "Error",
                "category": "NA: None applying",
                "rationale": "Parsed JSON was not an object.",
            }
        except json.JSONDecodeError:
            # Fallback: extract using regex
            rating_match = re.search(r'"rating"\s*:\s*"([^"]*)"', json_str)
            category_match = re.search(r'"category"\s*:\s*"([^"]*)"', json_str)
            rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', json_str)

            if rating_match and category_match:
                return {
                    "rating": rating_match.group(1),
                    "category": category_match.group(1),
                    "rationale": rationale_match.group(1) if rationale_match else "",
                }

            raise json.JSONDecodeError("Could not parse JSON", json_str, 0)  # noqa: B904

    except json.JSONDecodeError as e:
        return {
            "rating": "Error",
            "category": "NA: None applying",
            "rationale": f"JSON decode error: {e!s}. Raw output: {output_text[:500]}",
        }
    except Exception as e:
        return {
            "rating": "Error",
            "category": "NA: None applying",
            "rationale": f"Error parsing output: {e!s}. Raw output: {output_text[:500]}",
        }


def calculate_majority_vote(ratings: list[str]) -> str:
    """Calculate the majority vote from model ratings.

    Uses majority voting with tie-breaking based on conservative priority:
    Unsafe > Potentially Problematic > Safe > Unknown > Error
    """
    # Count occurrences of each rating
    rating_counts = Counter(ratings)
    most_common = rating_counts.most_common()

    # Define priority order (most conservative first)
    priority_order = ["Unsafe", "Potentially Problematic", "Safe", "Unknown", "Error"]

    # Get the majority vote (most common rating)
    if most_common:
        majority_rating, _ = most_common[0]

        # Check if there's a clear majority (one rating appears more than others)
        if len(most_common) == 1 or (
            len(most_common) > 1 and most_common[0][1] > most_common[1][1]
        ):
            # Clear majority
            return majority_rating
        else:
            # Tie or no consensus - select most conservative option
            # Get all ratings involved in the tie
            available_ratings = [r for r in ratings if r != "Unknown"]
            if not available_ratings:
                available_ratings = ratings

            # Select the most conservative (first in priority order)
            for priority_rating in priority_order:
                if priority_rating in available_ratings:
                    return priority_rating

            # Fallback
            return majority_rating
    else:
        return "Unknown"


def write_combined_csv(
    assessments: dict,
    output_csv: str | Path = RESULTS_CSV,
) -> None:
    """Write combined assessment results from all models to a single CSV.

    Args:
        assessments: dict mapping model name → list[Rating]
        output_csv: Path to the output CSV file
    """
    model_names = list(assessments.keys())
    combined: dict[str, dict] = {}

    for model_name, model_assessments in assessments.items():
        for assessment in model_assessments:
            image_name = assessment.image
            if image_name not in combined:
                combined[image_name] = {"image": image_name}
            combined[image_name][f"{model_name}_rating"] = assessment.rating
            combined[image_name][f"{model_name}_category"] = assessment.category
            combined[image_name][f"{model_name}_rationale"] = assessment.rationale

    for row in combined.values():
        ratings = [row.get(f"{name}_rating", "Unknown") for name in model_names]
        row["majority_vote"] = calculate_majority_vote(ratings)

    header = ["image", "majority_vote"] + [
        f"{name}_{field}"
        for name in model_names
        for field in ("rating", "category", "rationale")
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in combined.values():
            writer.writerow(row)

    print(f"\nCombined results saved to {output_csv}")
    print(f"Total images processed: {len(combined)}")
