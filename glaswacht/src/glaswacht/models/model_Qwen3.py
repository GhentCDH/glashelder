import logging

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)


class Qwen3(TransformersSafetyModel):
    name = "qwen3"
    PARSE_MODEL_TYPE = "qwen"

    def __init__(self):
        super().__init__()
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
            device_map="auto",
            quantization_config=bnb_config,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
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
            self._prefill_message(),
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            continue_final_message=True,
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

        return self.JSON_PREFILL + output_text[0]
