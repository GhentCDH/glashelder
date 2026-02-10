from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import torch
import os

from risk_assessment.models.utils import get_prompt, parse_output, get_images
from risk_assessment.orchestrator import ModelStatus


class LavaGuard:

    def __init__(self):
        self.model = None
        self.processor = None
        self.message = None
    
    def load_model(self):
        model_id = 'AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf'
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id).to('cuda:0').eval()
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

    def process_image(self, image, image_name=""):

        if self.model is None or self.processor is None:
            self.load_model()

        if self.message is None:
            self.message_constructor()

        text_prompt = self.processor.apply_chat_template(self.message, add_generation_prompt=True)  # ty:ignore[possibly-missing-attribute]
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")  # ty:ignore[call-non-callable]
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
            output = self.model.generate(**inputs, **hyperparameters)

        output_text = self.processor.decode(output[0], skip_special_tokens=True)

        assessment = parse_output(output_text)
        assessment["image"] = image_name
        return assessment


def run_lavaguard(orchestrator, image_path):
    orchestrator.update_model_state("model1", status=ModelStatus.LOADING, current_task="Loading LavaGuard model...")
    
    model = LavaGuard()
    model.load_model()
    
    orchestrator.update_model_state("model1", status=ModelStatus.RUNNING, current_task="Processing images...")
    
    image_files = get_images(image_path)
    for i, image_file in enumerate(image_files):
        image_name = os.path.basename(image_file)
        orchestrator.update_model_state("model1", current_task=f"Processing image {i+1}/{len(image_files)}", latest_image=image_name)
        image = Image.open(image_file).convert("RGB")
        assessment = model.process_image(image, image_name)
        orchestrator.assessments_lavaguard.append(assessment)
        orchestrator.update_model_state("model1", latest_rating=assessment.get("rating", "Unknown"))
        orchestrator.update_majority_votes_live()
    
    orchestrator.update_model_state("model1", status=ModelStatus.DONE, current_task="Done")
    return orchestrator.assessments_lavaguard





    
