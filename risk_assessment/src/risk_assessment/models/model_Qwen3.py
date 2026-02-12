from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import torch
import os

from risk_assessment.models.utils import get_prompt, parse_output, get_images
from risk_assessment.orchestrator import ModelStatus, ModelLoadError


class Qwen3:

    def __init__(self):
        self.model = None
        self.processor = None
        self.prompt = None
    
    
    def load_model(self):
        """Load the Qwen3-VL model and processor."""

     
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    
    def message_constructor(self):
        """Construct the message for Qwen3."""
        self.prompt = get_prompt()

    def process_image(self, image, image_name=""):
        """Process a single image and return the safety assessment."""
        if self.model is None or self.processor is None:
            self.load_model()

        if self.prompt is None:
            self.message_constructor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generation hyperparameters
        hyperparameters = {
            "max_new_tokens": 512,
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **hyperparameters)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Parse the output
        assessment = parse_output(output_text[0], model_type="qwen")
        assessment["image"] = image_name
        return assessment


def run_qwen(orchestrator, image_path):
    orchestrator.update_model_state("model2", status=ModelStatus.LOADING, current_task="Loading Qwen3 model...")
    
    model = Qwen3()
    try:
        model.load_model()
    except Exception as exc:
        orchestrator.update_model_state(
            "model2",
            status=ModelStatus.ERROR,
            current_task=f"Model load failed: {exc}",
        )
        raise ModelLoadError(str(exc)) from exc
    
    orchestrator.update_model_state("model2", status=ModelStatus.RUNNING, current_task="Processing images...")
    
    images = get_images(image_path)

    assessments = []

    for i, image_file in enumerate(images):
        image_name = os.path.basename(image_file)
        orchestrator.update_model_state(
            "model2",
            current_task=f"Processing image {i+1}/{len(images)}",
            latest_image=image_name,
        )
        try:
            image = Image.open(image_file).convert("RGB")
            assessment = model.process_image(image, image_name)
        except Exception as exc:
            orchestrator.update_model_state(
                "model2",
                current_task=f"Skipped {image_name}: {exc}",
            )
            continue

        orchestrator.assessments_qwen.append(assessment)
        orchestrator.update_model_state("model2", latest_rating=assessment.get("rating", "Unknown"))
        orchestrator.update_majority_votes_live()
    
    orchestrator.update_model_state("model2", status=ModelStatus.DONE, current_task="Done")
    return orchestrator.assessments_qwen
