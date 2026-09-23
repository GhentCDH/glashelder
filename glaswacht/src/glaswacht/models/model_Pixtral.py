import logging

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)


class Pixtral(TransformersSafetyModel):
    name = "pixtral"
    PARSE_MODEL_TYPE = "pixtral"

    def __init__(self):
        super().__init__()
        self.prompt = None

    def load_model(self) -> None:
        logger.info("Downloading/loading Pixtral model: mistral-community/pixtral-12b")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0
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
        self.prompt = f"<s>[INST]{taxonomy}\n[IMG][/INST]{self.JSON_PREFILL}"

    def _run_inference(self, image: Image.Image) -> str:
        if self.prompt is None:
            self.message_constructor()

        inputs = self.processor(
            images=[image], text=self.prompt, return_tensors="pt"
        ).to("cuda")

        hyperparameters = {"max_new_tokens": 200}

        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **hyperparameters)

        input_len = inputs["input_ids"].shape[-1]
        continuation = self.processor.batch_decode(
            generate_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self.JSON_PREFILL + continuation
