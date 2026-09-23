import gc
import logging
from abc import abstractmethod
from typing import ClassVar

import torch
from PIL import Image

from glaswacht.models.safety_model import Rating, SafetyModel
from glaswacht.models.utils import parse_output

logger = logging.getLogger(__name__)


class TransformersSafetyModel(SafetyModel):
    """Shared lifecycle for HF-transformers-backed safety models.

    Subclasses implement `load_model()` and `_run_inference()` exactly as
    their old `process_image()` bodies did; everything else (lazy load,
    defensive checks, output parsing, Rating construction) lives here.
    """

    PARSE_MODEL_TYPE: ClassVar[str]

    # Prefilled into the assistant's turn so the model only has to fill in
    # the values, rather than generate the JSON structure itself. Mirrors
    # the indentation of the template shown in models/taxonomy.txt.
    JSON_PREFILL: ClassVar[str] = '{\n    "rating": "'

    def __init__(self):
        self.model = None
        self.processor = None

    def _prefill_message(self) -> dict:
        """An assistant turn pre-filled with the JSON scaffold, for models
        that build a chat message list (see `apply_chat_template(...,
        continue_final_message=True)` in subclasses using this)."""
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": self.JSON_PREFILL}],
        }

    def unload_model(self) -> None:
        logger.info("Unloading %s model", self.name)
        self.model = None
        self.processor = None
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
            category=result.get("category", "NA: None applying"),
            rationale=result.get("rationale", ""),
            image=image_name,
        )

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def _run_inference(self, image: Image.Image) -> str:
        """Model-specific: build inputs, generate, decode. Returns raw text."""
