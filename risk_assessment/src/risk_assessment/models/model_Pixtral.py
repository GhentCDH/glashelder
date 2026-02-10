from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
import torch
import os

from risk_assessment.models.utils import get_prompt, parse_output, get_images
from risk_assessment.orchestrator import ModelStatus


class Pixtral:

    def __init__(self):
        self.model = None
        self.processor = None
        self.prompt = None
    
    def load_model(self):
        """Load the Pixtral model and processor with 8-bit quantization."""
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        model_id = "mistral-community/pixtral-12b"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='cuda',
            quantization_config=quantization_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def message_constructor(self):
        """Construct the prompt in Pixtral format."""
        taxonomy = get_prompt()
        self.prompt = f"<s>[INST]{taxonomy}\n[IMG][/INST]"

    def process_image(self, image, image_name=""):
        """Process a single image and return the risk assessment."""
        if self.model is None or self.processor is None:
            self.load_model()

        if self.prompt is None:
            self.message_constructor()
        
        inputs = self.processor(images=[image], text=self.prompt, return_tensors="pt").to("cuda")

        # Generation hyperparameters
        hyperparameters = {
            "max_new_tokens": 200,
        }

        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **hyperparameters)

        output_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Parse the output
        assessment = parse_output(output_text, model_type="pixtral")
        assessment["image"] = image_name
        return assessment


def run_pixtral(orchestrator, image_path):
    orchestrator.update_model_state("model3", status=ModelStatus.LOADING, current_task="Loading Pixtral model...")
    
    model = Pixtral()
    model.load_model()
    
    orchestrator.update_model_state("model3", status=ModelStatus.RUNNING, current_task="Processing images...")
    
    images = get_images(image_path)
    for i, image_file in enumerate(images):
        image_name = os.path.basename(image_file)
        orchestrator.update_model_state("model3", current_task=f"Processing image {i+1}/{len(images)}", latest_image=image_name)
        image = Image.open(image_file).convert("RGB")
        assessment = model.process_image(image, image_name)
        orchestrator.assessments_pixtral.append(assessment)
        orchestrator.update_model_state("model3", latest_rating=assessment.get("rating", "Unknown"))
        orchestrator.update_majority_votes_live()
    
    orchestrator.update_model_state("model3", status=ModelStatus.DONE, current_task="Done")
    return orchestrator.assessments_pixtral
