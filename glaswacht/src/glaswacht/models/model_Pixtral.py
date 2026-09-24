import logging
from typing import ClassVar

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from glaswacht.models.base import TransformersSafetyModel

logger = logging.getLogger(__name__)

# The corrected Mistral pre-tokenizer pattern, copied from
# `TokenizersBackend._patch_mistral_regex` in transformers 5.1.0.
_MISTRAL_SPLIT_PATTERN = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _fix_mistral_pretokenizer(tokenizer) -> None:
    """transformers 5.1.0 ships pixtral with an incorrect pre-tokenizer regex, and its
    own `fix_mistral_regex=True` flag raises TypeError (the flag is passed twice into
    `_patch_mistral_regex`). Apply the corrected Split pattern ourselves. Verified
    byte-identical on glaswacht's prompts — this is correctness-by-construction, not a
    behavior change; `tests/test_tokenizer_invariants.py` pins that.
    """
    from tokenizers import Regex, pre_tokenizers

    current = tokenizer.backend_tokenizer.pre_tokenizer
    split = pre_tokenizers.Split(
        pattern=Regex(_MISTRAL_SPLIT_PATTERN), behavior="isolated"
    )
    if isinstance(current, pre_tokenizers.Sequence):
        tokenizer.backend_tokenizer.pre_tokenizer[0] = split
    else:
        tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [split, current]
        )
    tokenizer.fix_mistral_regex = True


class Pixtral(TransformersSafetyModel):
    name = "pixtral"
    PARSE_MODEL_TYPE = "pixtral"
    MODEL_ID = "mistral-community/pixtral-12b"
    QUANTIZATION = "int8"

    # Pixtral's generation_config.json is bare, so this only spells out the greedy
    # defaults it already used — it keeps the trace's production decode block
    # comparable with the other two models.
    GENERATE_KWARGS: ClassVar[dict] = {
        "max_new_tokens": 200,
        "do_sample": False,
        "num_beams": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    def __init__(self):
        super().__init__()
        self.prompt = None

    def load_model(self) -> None:
        self._require_cuda()
        logger.info("Downloading/loading Pixtral model: %s", self.MODEL_ID)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0
        )

        model_id = self.MODEL_ID
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        _fix_mistral_pretokenizer(self.processor.tokenizer)
        logger.info("Pixtral model loaded successfully")

    def message_constructor(self):
        taxonomy = self._taxonomy_text()
        self.prompt = f"<s>[INST]{taxonomy}\n[IMG][/INST]{self._active_prefill()}"

    def _build_inputs(self, image: Image.Image) -> dict:
        if self.prompt is None:
            self.message_constructor()

        return self.processor(images=[image], text=self.prompt, return_tensors="pt").to(
            self.model.device
        )
