import logging
from typing import ClassVar

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from glaswacht.models.base import TransformersSafetyModel

logger = logging.getLogger(__name__)


class Qwen3(TransformersSafetyModel):
    name = "qwen3"
    PARSE_MODEL_TYPE = "qwen"
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    QUANTIZATION = "nf4-4bit"

    # Qwen3's generation_config.json ships do_sample=True with temperature/top_p/top_k,
    # so leaving the decode implicit made its ratings irreproducible. None rather than
    # neutral values: a leftover temperature with do_sample=False makes
    # `GenerationConfig.validate()` warn on every call.
    GENERATE_KWARGS: ClassVar[dict] = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    def __init__(self):
        super().__init__()
        self.prompt = None

    def load_model(self):
        self._require_cuda()
        logger.info("Downloading/loading Qwen3 model: %s", self.MODEL_ID)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype="auto",
            device_map="cuda",
            quantization_config=bnb_config,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        logger.info("Qwen3 model loaded successfully")

    def message_constructor(self):
        self.prompt = self._taxonomy_text()

    def _build_inputs(self, image: Image.Image) -> dict:
        if self.prompt is None:
            self.message_constructor()

        prefill = self._prefill_message()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            },
            *([prefill] if prefill else []),
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            continue_final_message=prefill is not None,
            add_generation_prompt=prefill is None,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)
