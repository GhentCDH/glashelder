from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import os
import csv
import json


def load_prompt():
    """Load the default prompt for Qwen3 safety assessment."""
    return """Provide ONLY a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence: 
Should not:
- Contain blood or gore.
Can:
- Contain fictional content (such as skeletons).

O3: Nudity Content
Should Not:
- Contain visible human genitalia or explicit nudity, including exposed breasts, nipples, vagina, vulva, or penis.
- Contain sexualized depictions of nudity.
Can:
- Contain nudity or visible genitalia in non-sexual artistic contexts, such as statues, sculptures, paintings, or drawings.
- Contain educational, medical, or scientific content providing factual information about the human body or anatomy, without sexual intent.

O4: Ethnographic content
Should not:
- Contain images that could be used for ethnographic purposes
Can:
- Contain images that display people in everyday settings
Category 04 is always rated Potentially Problemtic.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe"|"Potentially Problematic">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence"|"O3: Nudity Content"|"NA: None applying">,
    "rationale": str,
}"""


def load_model():
    """Load the Qwen3-VL model and processor."""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model, processor


def parse_output(output_text):
    """Parse the model output to extract JSON assessment."""
    try:
        # Look for JSON in the output
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


def process_image(image, model, processor, image_name="", prompt=None, max_new_tokens=200):
    """Process a single image and return the safety assessment."""
    if prompt is None:
        prompt = load_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(f"\nImage: {image_name}")
    print("-" * 50)

    # Parse the output
    assessment = parse_output(output_text[0])
    print(f"Assessment: {assessment}")
    print("-" * 50)

    return assessment


def assess_risk_of_image(image_path, max_new_tokens=200):
    """Assess the risk of a single image."""
    model, processor = load_model()
    prompt = load_prompt()

    image = Image.open(image_path).convert("RGB")
    assessment = process_image(image, model, processor, os.path.basename(image_path), prompt, max_new_tokens)
    return assessment


def assess_risk_of_image_folder(folder_path, output_csv="qwen3_risk_assessment_results.csv", max_new_tokens=200):
    """Assess the risk of all images in a folder and save results to CSV."""
    model, processor = load_model()
    prompt = load_prompt()

    # Create CSV header
    header = ["image", "rating", "category", "rationale"]

    results = []

    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                assessment = process_image(image, model, processor, image_file, prompt, max_new_tokens)

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


if __name__ == "__main__":
    # Example usage
    assess_risk_of_image_folder("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop")
    # assess_risk_of_image("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/sn01004006_tiff.jpg")