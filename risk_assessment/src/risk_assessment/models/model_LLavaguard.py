import gc
import logging

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
)

from risk_assessment.models.safety_model import Rating, SafetyModel
from risk_assessment.models.utils import get_prompt, parse_output

logger = logging.getLogger(__name__)


class LavaGuard(SafetyModel):
    name = "lavaguard"

    def __init__(self):
        self.model = None
        self.processor = None
        self.message = None

    def load_model(self):
        model_id = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
        logger.info("Downloading/loading LavaGuard model: %s", model_id)
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
        logger.info("LavaGuard model loaded successfully")

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
        logger.info("Unloading LavaGuard model")
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

        result = parse_output(output_text)
        return Rating(
            rating=result.get("rating", "Unknown"),
            category=result.get("category", "NA: None applying"),
            rationale=result.get("rationale", ""),
            image=image_name,
        )
