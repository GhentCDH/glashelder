import logging

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

    def __init__(self):
        super().__init__()
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

    def _run_inference(self, image: Image.Image) -> str:
        if self.message is None:
            self.message_constructor()

        messages = [*self.message, self._prefill_message()]
        text_prompt = self.processor.apply_chat_template(
            messages, continue_final_message=True
        )
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        model_device = next(self.model.parameters()).device
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
            output = self.model.generate(**inputs, **hyperparameters)

        input_len = inputs["input_ids"].shape[-1]
        continuation = self.processor.decode(
            output[0][input_len:], skip_special_tokens=True
        )
        return self.JSON_PREFILL + continuation
