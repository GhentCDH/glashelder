import logging

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)


class Gemma3(TransformersSafetyModel):
    name = "gemma3"
    PARSE_MODEL_TYPE = "gemma"

    MODEL_ID = "google/gemma-3-12b-it"

    def __init__(self):
        super().__init__()
        self.prompt = None

    def load_model(self) -> None:
        logger.info("Downloading/loading Gemma3 model: %s", self.MODEL_ID)
        # Gemma 3 is trained in bfloat16; float16 activations can overflow.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        logger.info("Gemma3 model loaded successfully")

    def message_constructor(self):
        self.prompt = get_prompt()

    def _run_inference(self, image: Image.Image) -> str:
        if self.prompt is None:
            self.message_constructor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.generate(inputs)

        input_len = inputs["input_ids"].shape[-1]
        continuation = self.processor.decode(
            generated_ids[0][input_len:], skip_special_tokens=True
        )
        return continuation
