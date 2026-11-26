import gc
import logging
import os
from abc import abstractmethod
from typing import ClassVar

import torch
import xgrammar as xgr
from PIL import Image
from transformers import LogitsProcessor, set_seed

from glaswacht.models.safety_model import (
    ASSESSMENT_SCHEMA,
    NO_CATEGORY,
    Rating,
    SafetyModel,
    join_categories,
)
from glaswacht.models.utils import parse_output

logger = logging.getLogger(__name__)

# Transformers otherwise copies weights to the GPU in parallel threads, still
# unquantized, ahead of the loop that quantizes them. With a warm disk cache
# the copies outrun quantization and the full-precision model (~24 GB for
# Gemma 3 12B) piles up on the GPU, so the load runs out of memory. Loading
# sequentially quantizes each weight as soon as it arrives.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

# generation settings only sampling uses, cleared for greedy decoding
_SAMPLING_FLAGS = ("temperature", "top_p", "top_k", "min_p", "typical_p")


class TransformersSafetyModel(SafetyModel):
    """Shared lifecycle for HF-transformers-backed safety models.

    Subclasses implement `load_model()` and `_run_inference()`; everything
    else (lazy load, defensive checks, JSON-constrained decoding, output
    parsing, Rating construction) lives here.
    """

    PARSE_MODEL_TYPE: ClassVar[str]

    # Room for the longest answer ASSESSMENT_SCHEMA allows: 2000 characters of
    # rationale measured at under 600 tokens with every model's tokenizer, plus
    # about 55 for the rest of the JSON
    MAX_NEW_TOKENS: ClassVar[int] = 1024
    # Reset before every image, so an image's result does not depend on
    # which images came before it
    SEED: ClassVar[int] = 0

    def __init__(self, full_precision: bool = False):
        super().__init__(full_precision)
        self.model = None
        self.processor = None
        self._grammar: xgr.CompiledGrammar | None = None

    def json_constraint(self) -> list[LogitsProcessor]:
        """Logits processors for `generate()` that force an answer matching
        ASSESSMENT_SCHEMA. Pass a fresh list to every `generate()` call; the
        processor keeps state and cannot be reused. Needs greedy or sampled
        decoding, not beam search."""
        if self._grammar is None:
            # the logits can be wider than the tokenizer's vocab due to padding
            vocab_size = self.model.get_output_embeddings().out_features
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.processor.tokenizer,
                vocab_size=vocab_size,
                stop_token_ids=self.model.generation_config.eos_token_id,
            )
            self._grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
                ASSESSMENT_SCHEMA, any_whitespace=False
            )
        return [xgr.contrib.hf.LogitsProcessor(self._grammar)]

    def generate(self, inputs) -> torch.Tensor:
        """Run `generate()` reproducibly: seeded, greedy (temperature 0) and
        constrained to ASSESSMENT_SCHEMA. Returns prompt plus answer ids."""
        # Greedy decoding. The model's own generation_config may ask for
        # sampling (Qwen3 and Gemma 3 do); clearing its sampling values, not
        # just turning sampling off, also keeps transformers from warning
        # that they are ignored on every image.
        generation_config = self.model.generation_config
        generation_config.do_sample = False
        for flag in _SAMPLING_FLAGS:
            setattr(generation_config, flag, None)

        set_seed(self.SEED)
        with torch.inference_mode():
            return self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                num_beams=1,
                logits_processor=self.json_constraint(),
            )

    def unload_model(self) -> None:
        logger.info("Unloading %s model", self.name)
        self.model = None
        self.processor = None
        self._grammar = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def process_image(self, image: Image.Image, image_name: str = "") -> Rating:
        if self.model is None or self.processor is None:
            self.load_model()
        if self.model is None or self.processor is None:
            raise RuntimeError(f"{self.name} model or processor failed to initialize")

        output_text = self._run_inference(image)

        result = parse_output(output_text, model_type=self.PARSE_MODEL_TYPE)
        return Rating(
            rating=result.get("rating", "Unknown"),
            category=join_categories(result.get("category", NO_CATEGORY)),
            rationale=result.get("rationale", ""),
            image=image_name,
        )

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def _run_inference(self, image: Image.Image) -> str:
        """Model-specific: build inputs, generate, decode. Returns raw text."""
