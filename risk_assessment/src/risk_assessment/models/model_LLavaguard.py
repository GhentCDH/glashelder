import gc
import os

import torch
from PIL import Image
from risk_assessment.models.utils import get_images, get_prompt, parse_output
from risk_assessment.orchestrator import ModelLoadError, ModelStatus
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
)


class LavaGuard:
    def __init__(self):
        self.model = None
        self.processor = None
        self.message = None

    def load_model(self):
        model_id = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def message_constructor(self):
        taxonomy = get_prompt()

        self.message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": taxonomy},
                ],
            },
        ]

    def unload_model(self):
        self.model = None
        self.processor = None
        self.message = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def process_image(self, image, image_name=""):

        if self.model is None or self.processor is None:
            self.load_model()

        model = self.model
        processor = self.processor
        if model is None or processor is None:
            raise RuntimeError("LavaGuard model or processor failed to initialize")

        if self.message is None:
            self.message_constructor()

        text_prompt = processor.apply_chat_template(
            self.message, add_generation_prompt=True
        )
        inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

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

        assessment = parse_output(output_text)
        assessment["image"] = image_name
        return assessment


def run_lavaguard(orchestrator, image_path):
    orchestrator.update_model_state(
        "model1", status=ModelStatus.LOADING, current_task="Loading LavaGuard model..."
    )

    model = LavaGuard()
    try:
        model.load_model()
    except Exception as exc:
        orchestrator.update_model_state(
            "model1",
            status=ModelStatus.ERROR,
            current_task=f"Model load failed: {exc}",
        )
        raise ModelLoadError(str(exc)) from exc

    try:
        orchestrator.update_model_state(
            "model1", status=ModelStatus.RUNNING, current_task="Processing images..."
        )

        image_files = get_images(image_path)
        for i, image_file in enumerate(image_files):
            image_name = os.path.basename(image_file)
            orchestrator.update_model_state(
                "model1",
                current_task=f"Processing image {i + 1}/{len(image_files)}",
                latest_image=image_name,
            )
            try:
                image = Image.open(image_file).convert("RGB")
                assessment = model.process_image(image, image_name)
            except Exception as exc:
                orchestrator.update_model_state(
                    "model1",
                    current_task=f"Skipped {image_name}: {exc}",
                )
                continue

            orchestrator.assessments_lavaguard.append(assessment)
            orchestrator.update_model_state(
                "model1", latest_rating=assessment.get("rating", "Unknown")
            )
            orchestrator.update_majority_votes_live()

        orchestrator.update_model_state(
            "model1", status=ModelStatus.DONE, current_task="Done"
        )
        return orchestrator.assessments_lavaguard
    finally:
        model.unload_model()
