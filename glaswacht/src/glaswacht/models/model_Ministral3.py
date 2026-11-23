import logging
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)

# the chat template's own default system prompt, which it adds whenever the
# conversation has none
_DEFAULT_SYSTEM_PROMPT = re.compile(
    r"\{%-? set default_system_message = '.*?' -?%\}", flags=re.S
)


def _without_default_system_prompt(chat_template: str) -> str:
    """The chat template without its default system prompt.

    Ministral's template adds Mistral's general assistant prompt (who the
    model is, today's date, how to use tools) to every conversation without
    a system message. That is irrelevant here and only dilutes the policy, so
    it is set empty, which the template takes as no system prompt at all.
    """
    cleared, found = _DEFAULT_SYSTEM_PROMPT.subn(
        "{%- set default_system_message = '' %}", chat_template
    )
    if found != 1:
        raise RuntimeError(
            "Ministral 3 chat template changed: cannot remove its default "
            f"system prompt (found {found} definitions, expected 1)"
        )
    return cleared


class Ministral3(TransformersSafetyModel):
    name = "ministral3"
    PARSE_MODEL_TYPE = "ministral"

    # Mistral's checkpoint stores the language model in FP8 (the 4090's
    # compute capability 8.9 is the minimum for that) and keeps the vision
    # tower, projector and lm_head in bfloat16, so no quantization config is
    # passed here.
    MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"

    def __init__(self):
        super().__init__()
        self.prompt = None

    def load_model(self) -> None:
        logger.info("Downloading/loading Ministral 3 model: %s", self.MODEL_ID)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            # the unquantized parts are stored in bfloat16
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.processor.chat_template = _without_default_system_prompt(
            self.processor.chat_template
        )
        logger.info("Ministral 3 model loaded successfully")

    def message_constructor(self):
        self.prompt = get_prompt()

    def _run_inference(self, image: Image.Image) -> str:
        if self.prompt is None:
            self.message_constructor()

        # text before the image, as in Mistral's own example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image", "image": image},
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
        # the vision tower runs in bfloat16, the processor returns float32
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        generated_ids = self.generate(inputs)

        input_len = inputs["input_ids"].shape[-1]
        return self.processor.decode(
            generated_ids[0][input_len:], skip_special_tokens=True
        )
