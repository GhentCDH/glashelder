import gc
import logging

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from risk_assessment.models.safety_model import Rating, SafetyModel
from risk_assessment.models.utils import get_prompt, parse_output

logger = logging.getLogger(__name__)


class Pixtral(SafetyModel):
    name = "pixtral"

    def __init__(self):
        self.model = None
        self.processor = None
        self.prompt = None

    def load_model(self) -> None:
        logger.info("Downloading/loading Pixtral model: mistral-community/pixtral-12b")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model_id = "mistral-community/pixtral-12b"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info("Pixtral model loaded successfully")

    def message_constructor(self):
        taxonomy = get_prompt()
        self.prompt = f"<s>[INST]{taxonomy}\n[IMG][/INST]"

    def unload_model(self) -> None:
        logger.info("Unloading Pixtral model")
        loaded_model = self.model
        loaded_processor = self.processor

        self.model = None
        self.processor = None
        self.prompt = None

        if loaded_model is not None:
            del loaded_model
        if loaded_processor is not None:
            del loaded_processor

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def process_image(self, image: Image.Image, image_name: str = "") -> Rating:
        if self.model is None or self.processor is None:
            self.load_model()

        model = self.model
        processor = self.processor
        if model is None or processor is None:
            raise RuntimeError("Pixtral model or processor failed to initialize")

        if self.prompt is None:
            self.message_constructor()

        inputs = processor(images=[image], text=self.prompt, return_tensors="pt").to(
            "cuda"
        )

        hyperparameters = {"max_new_tokens": 200}

        with torch.inference_mode():
            generate_ids = model.generate(**inputs, **hyperparameters)

        output_text = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        result = parse_output(output_text, model_type="pixtral")
        return Rating(
            rating=result.get("rating", "Unknown"),
            category=result.get("category", "NA: None applying"),
            rationale=result.get("rationale", ""),
            image=image_name,
        )
