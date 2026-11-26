import logging

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)


class LavaGuard(TransformersSafetyModel):
    name = "lavaguard"
    PARSE_MODEL_TYPE = "lava"
    enabled_by_default = False

    def __init__(self, full_precision: bool = False):
        super().__init__(full_precision)
        self.message = None

    MODEL_ID = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"

    def load_model(self):
        model_id = self.MODEL_ID
        logger.info("Downloading/loading LavaGuard model: %s", model_id)
        # full precision: 15.0 GiB of bfloat16 weights, which fits the 4090
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            # all on the GPU: "auto" could quietly offload layers to the CPU
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

    def _run_inference(self, image: Image.Image) -> str:
        if self.message is None:
            self.message_constructor()

        text_prompt = self.processor.apply_chat_template(
            self.message, add_generation_prompt=True
        )
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        output = self.generate(inputs)

        input_len = inputs["input_ids"].shape[-1]
        continuation = self.processor.decode(
            output[0][input_len:], skip_special_tokens=True
        )
        return continuation
