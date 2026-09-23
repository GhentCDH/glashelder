import logging
from typing import ClassVar

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
)

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)


class LavaGuard(TransformersSafetyModel):
    name = "lavaguard"
    PARSE_MODEL_TYPE = "lava"
    MODEL_ID = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
    QUANTIZATION = "nf4-4bit"

    GENERATE_KWARGS: ClassVar[dict] = {
        "max_new_tokens": 200,
        "do_sample": False,
        "num_beams": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "use_cache": True,
    }

    def __init__(self):
        super().__init__()
        self.message = None

    def load_model(self):
        self._require_cuda()
        model_id = self.MODEL_ID
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
            device_map="cuda",
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

    def _build_inputs(self, image: Image.Image) -> dict:
        if self.message is None:
            self.message_constructor()

        messages = [*self.message, self._prefill_message()]
        text_prompt = self.processor.apply_chat_template(
            messages, continue_final_message=True
        )
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        return inputs.to(model_device)
