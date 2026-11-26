import logging

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

from glaswacht.models.base import TransformersSafetyModel
from glaswacht.models.utils import get_prompt

logger = logging.getLogger(__name__)


def _current_module_names(skip_modules: list[str]) -> list[str]:
    """The checkpoint's unquantized modules, also under the names transformers 5
    gives them.

    The checkpoint was saved with transformers 4, where a Llava model's parts
    were `vision_tower`, `multi_modal_projector` and `language_model.model`.
    Transformers 5 nests them under `model.` (`model.language_model` for the
    last) and matches the list from the start of a module's name. Without
    these names it builds those modules as 4-bit layers, the unquantized
    weights do not fit them, and bitsandbytes fails on the first image.
    """
    names = list(skip_modules)
    for name in skip_modules:
        if name.startswith("language_model.model."):
            names.append(
                "model.language_model." + name.removeprefix("language_model.model.")
            )
        elif not name.startswith(("model.", "lm_head")):
            names.append("model." + name)
    return list(dict.fromkeys(names))


class Pixtral(TransformersSafetyModel):
    name = "pixtral"
    PARSE_MODEL_TYPE = "pixtral"
    # replaced by Ministral 3; still runs with `--enable pixtral`
    enabled_by_default = False

    # Pixtral 12B, already quantized to 4-bit NF4 by Unsloth. The checkpoint
    # carries its own quantization config, so none is passed here. This is the
    # "dynamic" one, which keeps the vision tower, the projector and lm_head
    # unquantized: the plain unsloth/Pixtral-12B-2409-bnb-4bit quantizes the
    # vision tower too and misreads images badly.
    MODEL_ID = "unsloth/Pixtral-12B-2409-unsloth-bnb-4bit"

    def __init__(self, full_precision: bool = False):
        super().__init__(full_precision)
        self.prompt = None

    @property
    def precision(self) -> str:
        return "nf4"

    def load_model(self) -> None:
        logger.info("Downloading/loading Pixtral model: %s", self.MODEL_ID)
        if self.full_precision:
            logger.warning("Pixtral has no full-precision version here, it stays 4-bit")
        config = AutoConfig.from_pretrained(self.MODEL_ID)
        quantization = config.quantization_config
        quantization["llm_int8_skip_modules"] = _current_module_names(
            quantization["llm_int8_skip_modules"]
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            config=config,
            # the checkpoint's 4-bit compute dtype
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        logger.info("Pixtral model loaded successfully")

    def message_constructor(self):
        self.prompt = get_prompt()

    def _run_inference(self, image: Image.Image) -> str:
        if self.prompt is None:
            self.message_constructor()

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

        generate_ids = self.generate(inputs)

        input_len = inputs["input_ids"].shape[-1]
        return self.processor.decode(
            generate_ids[0][input_len:], skip_special_tokens=True
        )
