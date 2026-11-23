import gc
import logging
import time
from abc import abstractmethod
from typing import ClassVar

import numpy as np
import torch
from PIL import Image

from glaswacht.models.safety_model import (
    Rating,
    SafetyModel,
    SeenImage,
    TokenStep,
    Trace,
)
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

    # first-token candidates for the rating, read right after the prefill
    RATING_LABELS: ClassVar[tuple[str, ...]] = (
        "Safe",
        "Unsafe",
        "Potentially Problematic",
    )

    def inspect(
        self,
        image: Image.Image,
        image_name: str = "",
        top_k: int = 8,
        max_new: int = 200,
    ) -> Trace:
        """Greedy generation that keeps every step's distribution.

        Unlike `process_image` this never samples or beam-searches (LlavaGuard
        normally does both), so each step has one clean distribution to show.
        """
        if self.model is None or self.processor is None:
            self.load_model()
        tok = self.processor.tokenizer

        t0 = time.perf_counter()
        inputs = self._prepare_inputs(image)
        prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                output_scores=True,
                return_dict_in_generate=True,
            )
        latency = time.perf_counter() - t0
        gen_ids = out.sequences[0][prompt_len:].tolist()

        steps: list[TokenStep] = []
        label_probs: dict[str, float] = {}
        shown = (
            ""  # decoding ids one by one loses leading spaces, so diff cumulative text
        )
        for i, (scores, tid) in enumerate(zip(out.scores, gen_ids, strict=False)):
            so_far = tok.decode(
                gen_ids[: i + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            piece = so_far[len(shown) :] if so_far.startswith(shown) else so_far
            shown = so_far
            logp = torch.log_softmax(scores[0].float(), dim=-1)
            p = logp.exp()
            entropy = float(-(p * logp).nansum())
            top_p, top_i = p.topk(top_k)
            steps.append(
                TokenStep(
                    index=i,
                    token=piece,
                    token_id=tid,
                    prob=float(p[tid]),
                    entropy=entropy,
                    top=[
                        (
                            tok.decode([int(j)], clean_up_tokenization_spaces=False),
                            float(q),
                        )
                        for q, j in zip(top_p, top_i, strict=True)
                    ],
                )
            )
            if i == 0:
                for label in self.RATING_LABELS:
                    first = tok.encode(label, add_special_tokens=False)[0]
                    label_probs[label] = float(p[first])

        text = self.JSON_PREFILL + tok.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        parsed = parse_output(text, model_type=self.PARSE_MODEL_TYPE)
        rating = Rating(
            rating=parsed.get("rating", "Unknown"),
            category=parsed.get("category", "NA: None applying"),
            rationale=parsed.get("rationale", ""),
            image=image_name,
        )
        return Trace(
            model=self.name,
            image=image_name,
            prefill=self.JSON_PREFILL,
            steps=steps,
            text=text,
            rating=rating,
            label_probs=label_probs,
            seen=self._seen_images(inputs, image),
            prompt_tokens=prompt_len,
            image_tokens=self._count_image_tokens(inputs),
            latency_s=latency,
            vram_mb=(
                torch.cuda.memory_allocated() / 2**20
                if torch.cuda.is_available()
                else 0.0
            ),
        )

    def _count_image_tokens(self, inputs) -> int:
        cfg = self.model.config
        token_id = getattr(cfg, "image_token_id", None) or getattr(
            cfg, "image_token_index", None
        )
        if token_id is None:
            return 0
        return int((inputs["input_ids"] == token_id).sum())

    def _seen_images(self, inputs, original: Image.Image) -> list[SeenImage]:
        """Undo the processor's normalisation to show what the network gets."""
        try:
            ip = self.processor.image_processor
            mean = np.array(ip.image_mean, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(ip.image_std, dtype=np.float32).reshape(3, 1, 1)
            patch = getattr(ip, "patch_size", None)
            if isinstance(patch, dict):
                patch = patch.get("height")

            def to_rgb(chw) -> np.ndarray:
                a = chw.float().cpu().numpy() * std + mean
                return (a.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)

            pv = inputs["pixel_values"]
            if isinstance(pv, list | tuple):
                pv = pv[0]
            if pv.ndim == 5:  # anyres tiles: [B, tiles, C, H, W]
                return [
                    SeenImage(f"tile {i}", to_rgb(t), patch)
                    for i, t in enumerate(pv[0])
                ]
            if pv.ndim == 4:  # one resized image
                return [SeenImage("resized", to_rgb(pv[0]), patch)]
            if "image_grid_thw" in inputs:  # flattened patches (Qwen)
                _, h, w = (int(x) for x in inputs["image_grid_thw"][0])
                ps = int(getattr(ip, "patch_size", 16))
                return [
                    SeenImage(
                        "resized",
                        np.asarray(original.convert("RGB").resize((w * ps, h * ps))),
                        ps,
                    )
                ]
        except Exception:
            logger.warning("could not reconstruct seen images", exc_info=True)
        return []

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def _prepare_inputs(self, image: Image.Image):
        """Model-specific: build the processor inputs, moved to the device."""

    @abstractmethod
    def _run_inference(self, image: Image.Image) -> str:
        """Model-specific: build inputs, generate, decode. Returns raw text."""
