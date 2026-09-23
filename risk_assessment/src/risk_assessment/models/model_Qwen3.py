import gc
import logging

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from risk_assessment.models.safety_model import Rating, SafetyModel
from risk_assessment.models.utils import get_prompt, parse_output

logger = logging.getLogger(__name__)


class Qwen3(SafetyModel):
    name = "qwen3"

    def __init__(self):
        self.model = None
        self.processor = None
        self.prompt = None

    def load_model(self):
        logger.info("Downloading/loading Qwen3 model: Qwen/Qwen3-VL-8B-Instruct")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype="auto",
            device_map="cuda:0",
            quantization_config=bnb_config,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        logger.info("Qwen3 model loaded successfully")

    def message_constructor(self):
        self.prompt = get_prompt()

    def unload_model(self):
        logger.info("Unloading Qwen3 model")
        self.model = None
        self.processor = None
        self.prompt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def process_image(self, image: Image.Image, image_name: str = "") -> Rating:
        if self.model is None or self.processor is None:
            self.load_model()

        if self.prompt is None:
            self.message_constructor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        hyperparameters = {"max_new_tokens": 512}

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

        result = parse_output(output_text[0], model_type="qwen")
        return Rating(
            rating=result.get("rating", "Unknown"),
            category=result.get("category", "NA: None applying"),
            rationale=result.get("rationale", ""),
            image=image_name,
        )
