import logging

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)


class Qwen3(TransformersSafetyModel):
    name = "qwen3"
    PARSE_MODEL_TYPE = "qwen"

    def __init__(self):
        super().__init__()
        self.prompt = None

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def load_model(self):
        logger.info("Downloading/loading Qwen3 model: %s", self.MODEL_ID)
        # full precision: 16.3 GiB of bfloat16 weights, which fits the 4090
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            # all on the GPU: "auto" could quietly offload layers to the CPU
            device_map="cuda",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        logger.info("Qwen3 model loaded successfully")

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
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.generate(inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]
