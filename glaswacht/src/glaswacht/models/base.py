import gc
import logging
import os
import time
from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar

import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

from glaswacht.models.diagnostics import (
    Counterfactual,
    DiagnosticsTrace,
    TokenAlternative,
    TokenStep,
    rating_distribution,
    token_pieces,
    utc_now,
)
from glaswacht.models.safety_model import Rating, SafetyModel
from glaswacht.models.utils import get_prompt, parse_output

logger = logging.getLogger(__name__)

# (phase, tokens generated so far, tokens budgeted) — see `diagnose(progress=...)`.
ProgressCallback = Callable[[str, int, int], None]


class _StepCounter(StoppingCriteria):
    """Reports generation progress without ever stopping.

    `generate()` exposes no per-step callback, so a stopping criterion that
    always returns False is the hook for a live token counter.
    """

    def __init__(self, callback: ProgressCallback, phase: str, total: int):
        self._callback = callback
        self._phase = phase
        self._total = total
        self._steps = 0

    def __call__(self, input_ids, scores, **kwargs):
        self._steps += 1
        try:
            self._callback(self._phase, self._steps, self._total)
        except Exception:
            # A failing progress display must never take generation down.
            logger.debug("progress callback failed", exc_info=True)
        return torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )


class TransformersSafetyModel(SafetyModel):
    """Shared lifecycle for HF-transformers-backed safety models.

    Subclasses implement `load_model()` and `_build_inputs()`; generation,
    decoding, output parsing and Rating construction live here so the
    diagnostics capture path is written once.
    """

    PARSE_MODEL_TYPE: ClassVar[str]
    MODEL_ID: ClassVar[str]
    QUANTIZATION: ClassVar[str]

    # Per-model production decoding hyperparameters, previously inline in
    # each `_run_inference`.
    GENERATE_KWARGS: ClassVar[dict] = {}

    CLEAN_UP_TOKENIZATION_SPACES: ClassVar[bool] = False

    # Applied on top of GENERATE_KWARGS for diagnostics runs. The sampling
    # params are set to None rather than to neutral values because
    # `GenerationConfig.validate()` warns about any sampling parameter that
    # is still set while `do_sample=False`.
    GREEDY_OVERRIDES: ClassVar[dict] = {
        "do_sample": False,
        "num_beams": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "use_cache": True,
    }

    # Prefilled into the assistant's turn so the model only has to fill in
    # the values, rather than generate the JSON structure itself. Mirrors
    # the indentation of the template shown in models/taxonomy.txt.
    JSON_PREFILL: ClassVar[str] = '{\n    "rating": "'

    def __init__(self):
        self.model = None
        self.processor = None
        # Set (e.g. by `diagnose(taxonomy_override=...)`) to swap in an
        # edited taxonomy for this instance's prompt only, without ever
        # touching taxonomy.txt on disk. A fresh model instance is created
        # per diagnostics run, so this never leaks across runs.
        self.taxonomy_override: str | None = None
        # False lets the model respond freely, with no JSON scaffold prefilled
        # into its turn — see `diagnose(enable_json_schema=...)`.
        self.enable_json_schema: bool = True

    @staticmethod
    def _require_cuda() -> None:
        """Fail loudly rather than silently falling back to CPU.

        These models are only practical quantized on a GPU, and a CPU fallback
        is slow enough to look like a hang rather than an error.
        """
        if torch.cuda.is_available():
            return
        raise RuntimeError(
            f"CUDA is required but unavailable. Installed torch is "
            f"{torch.__version__} (cuda build: {torch.version.cuda}). "
            "A '+cpu' build means the workspace is pinned to the CPU wheel "
            "index — check [[tool.uv.index]] in pyproject.toml, then re-run "
            "`uv lock && uv sync`."
        )

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _taxonomy_text(self) -> str:
        """The taxonomy text this instance builds its prompt from: the
        session-only override if one was set, else taxonomy.txt."""
        return (
            self.taxonomy_override
            if self.taxonomy_override is not None
            else get_prompt()
        )

    def _active_prefill(self) -> str:
        """The JSON scaffold prefilled into the assistant's turn, or "" when
        `enable_json_schema` is off and the model is left to respond freely."""
        return self.JSON_PREFILL if self.enable_json_schema else ""

    def _prefill_message(self) -> dict | None:
        """An assistant turn pre-filled with the JSON scaffold, for models
        that build a chat message list (see `apply_chat_template(...,
        continue_final_message=True)` in subclasses using this). `None` when
        `enable_json_schema` is off — there is nothing to continue, the model
        gets a fresh assistant turn instead (`add_generation_prompt=True`)."""
        if not self.enable_json_schema:
            return None
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

    def _decode(self, token_ids) -> str:
        return self.processor.batch_decode(
            [token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=self.CLEAN_UP_TOKENIZATION_SPACES,
        )[0]

    def _run_inference(self, image: Image.Image) -> str:
        """Build inputs, generate, decode the continuation. Returns raw text."""
        inputs = self._build_inputs(image)

        with torch.inference_mode():
            output = self.model.generate(**inputs, **self.GENERATE_KWARGS)

        input_len = inputs["input_ids"].shape[-1]
        return self.JSON_PREFILL + self._decode(output[0][input_len:])

    def _diagnostic_kwargs(self, max_new_tokens: int | None = None) -> dict:
        kwargs = {**self.GENERATE_KWARGS, **self.GREEDY_OVERRIDES}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        return kwargs

    def _image_token_ids(self) -> set[int]:
        config = getattr(self.model, "config", None)
        ids = set()
        for attr in ("image_token_id", "image_token_index", "video_token_id"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                ids.add(value)
        return ids

    def _capture_steps(
        self, logits, generated_ids: list[int], top_k: int
    ) -> tuple[list[TokenStep], torch.Tensor]:
        pieces = token_pieces(self.tokenizer, generated_ids)
        raw_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        steps: list[TokenStep] = []
        step0_probs = None

        for i, token_id in enumerate(generated_ids):
            logprobs = torch.log_softmax(logits[i][0].float(), -1)
            if i == 0:
                step0_probs = logprobs.exp()
            top_logprobs, top_ids = torch.topk(logprobs, top_k)
            alternatives = [
                TokenAlternative(
                    token_id=int(alt_id),
                    token=self.tokenizer.decode([int(alt_id)]),
                    raw_token=str(
                        self.tokenizer.convert_ids_to_tokens([int(alt_id)])[0]
                    ),
                    prob=float(alt_logprob.exp()),
                )
                for alt_logprob, alt_id in zip(top_logprobs, top_ids, strict=True)
            ]
            steps.append(
                TokenStep(
                    i=i,
                    token_id=token_id,
                    token=pieces[i],
                    raw_token=str(raw_tokens[i]),
                    prob=float(logprobs[token_id].exp()),
                    logprob=float(logprobs[token_id]),
                    alternatives=alternatives,
                )
            )

        return steps, step0_probs

    def _progress_kwargs(
        self, progress: ProgressCallback | None, phase: str, gen_kwargs: dict
    ) -> dict:
        if progress is None:
            return {}
        total = int(gen_kwargs.get("max_new_tokens") or 0)
        return {
            "stopping_criteria": StoppingCriteriaList(
                [_StepCounter(progress, phase, total)]
            )
        }

    def diagnose(
        self,
        image: Image.Image,
        image_path: str,
        top_k: int = 5,
        max_new_tokens: int | None = None,
        progress: ProgressCallback | None = None,
        custom_counterfactual: tuple[int, str] | None = None,
        taxonomy_override: str | None = None,
        enable_json_schema: bool = True,
        enable_rating_counterfactuals: bool = True,
    ) -> DiagnosticsTrace:
        """Greedy generation with per-step logits captured, plus the tier-1
        rating counterfactuals. Assumes the model is already loaded.

        `progress` is called as (phase, tokens_done, tokens_budgeted) on every
        generation step; the token budget is an upper bound, so a run that hits
        EOS early finishes before reaching it.

        `custom_counterfactual`, if given, is `(position, forced_text)`: an
        extra ad-hoc counterfactual that forces arbitrary user-supplied text
        into the generation starting at token index `position` of *this
        model's own* real continuation, on top of the automatic rating
        counterfactuals. Works with or without the JSON schema.

        `taxonomy_override`, if given, replaces taxonomy.txt's contents for
        this instance's prompt only — never written to disk, and discarded
        with the instance once this run finishes.

        `enable_json_schema=False` skips prefilling the assistant's turn with
        the `{"rating": "` scaffold, letting the model respond freely. Since
        the rating decision distribution and the automatic rating
        counterfactuals both rest on that scaffold's token boundary, neither
        is computed in this mode (`rating_distribution` comes back `None`,
        `counterfactuals` holds only the custom one, if any) regardless of
        `enable_rating_counterfactuals`.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError(f"{self.name} model or processor failed to initialize")
        if taxonomy_override is not None:
            self.taxonomy_override = taxonomy_override
        self.enable_json_schema = enable_json_schema

        gen_kwargs = self._diagnostic_kwargs(max_new_tokens)
        inputs = self._build_inputs(image)
        prompt_token_count = int(inputs["input_ids"].shape[-1])

        started = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_logits=True,
                **self._progress_kwargs(progress, "generating", gen_kwargs),
            )
        generate_s = time.perf_counter() - started

        continuation = out.sequences[0, prompt_token_count:]
        generated_ids = [int(t) for t in continuation]
        steps, step0_probs = self._capture_steps(out.logits, generated_ids, top_k)
        distribution = (
            rating_distribution(self.tokenizer, step0_probs)
            if enable_json_schema
            else None
        )
        out["logits"] = None

        prefill_text = self._active_prefill()
        generated_text = prefill_text + self._decode(continuation)
        parsed = parse_output(generated_text, model_type=self.PARSE_MODEL_TYPE)

        started = time.perf_counter()
        counterfactuals: list[Counterfactual] = []
        if distribution is not None and enable_rating_counterfactuals:
            counterfactuals = self._rating_counterfactuals(
                image,
                generated_ids,
                distribution.candidate_first_tokens,
                max_new_tokens,
                top_k=top_k,
                progress=progress,
            )
        if custom_counterfactual is not None:
            position, forced_text = custom_counterfactual
            extra = self._custom_counterfactual(
                image,
                generated_ids,
                position,
                forced_text,
                max_new_tokens,
                top_k,
                progress,
            )
            if extra is not None:
                counterfactuals.append(extra)
        counterfactuals_s = time.perf_counter() - started

        return DiagnosticsTrace(
            image={
                "path": os.path.abspath(image_path),
                "name": os.path.basename(image_path),
            },
            model={
                "name": self.name,
                "model_id": self.MODEL_ID,
                "parse_model_type": self.PARSE_MODEL_TYPE,
                "quantization": self.QUANTIZATION,
            },
            decoding={
                "diagnostic": dict(gen_kwargs),
                "production": dict(self.GENERATE_KWARGS),
            },
            top_k=top_k,
            prefill=prefill_text,
            prompt_token_count=prompt_token_count,
            generated_text=generated_text,
            parsed=parsed,
            rating_distribution=distribution,
            steps=steps,
            counterfactuals=counterfactuals,
            timings_s={
                "generate": generate_s,
                "counterfactuals": counterfactuals_s,
            },
            generated_at=utc_now(),
        )

    def _rating_counterfactuals(
        self,
        image: Image.Image,
        generated_ids: list[int],
        first_tokens: dict[str, int],
        max_new_tokens: int | None = None,
        top_k: int = 5,
        progress: ProgressCallback | None = None,
    ) -> list[Counterfactual]:
        """Re-generate with each rating the model did *not* pick forced at step 0."""
        chosen = generated_ids[0] if generated_ids else None
        results: list[Counterfactual] = []
        pending = [(c, t) for c, t in first_tokens.items() if t != chosen]

        for index, (candidate, token_id) in enumerate(pending, start=1):
            try:
                text, steps = self.counterfactual(
                    image,
                    generated_ids,
                    0,
                    [token_id],
                    max_new_tokens,
                    top_k=top_k,
                    progress=progress,
                    phase=f"counterfactual {index}/{len(pending)}",
                )
            except Exception:
                logger.warning(
                    "%s: counterfactual for %r failed",
                    self.name,
                    candidate,
                    exc_info=True,
                )
                continue
            results.append(
                Counterfactual(
                    kind="rating",
                    position=0,
                    forced_text=self.tokenizer.decode([token_id]),
                    forced_token_ids=[token_id],
                    generated_text=text,
                    parsed=parse_output(text, model_type=self.PARSE_MODEL_TYPE),
                    steps=steps,
                )
            )

        return results

    def _custom_counterfactual(
        self,
        image: Image.Image,
        generated_ids: list[int],
        position: int,
        forced_text: str,
        max_new_tokens: int | None,
        top_k: int,
        progress: ProgressCallback | None,
    ) -> Counterfactual | None:
        """Force arbitrary `forced_text` into the generation at `position` (a
        token index into this model's own real continuation) and regenerate
        from there. `position` is clamped to the real continuation's length,
        since a diagnostics pass can cover several models whose real
        continuations differ in length for the same requested position.
        """
        forced_prefix_ids = self.tokenizer.encode(forced_text, add_special_tokens=False)
        if not forced_prefix_ids:
            logger.warning(
                "%s: custom counterfactual text encoded to no tokens", self.name
            )
            return None
        clamped_position = max(0, min(position, len(generated_ids)))

        try:
            text, steps = self.counterfactual(
                image,
                generated_ids,
                clamped_position,
                forced_prefix_ids,
                max_new_tokens,
                top_k=top_k,
                progress=progress,
                phase="custom counterfactual",
            )
        except Exception:
            logger.warning("%s: custom counterfactual failed", self.name, exc_info=True)
            return None

        return Counterfactual(
            kind="custom",
            position=clamped_position,
            forced_text=forced_text,
            forced_token_ids=forced_prefix_ids,
            generated_text=text,
            parsed=parse_output(text, model_type=self.PARSE_MODEL_TYPE),
            steps=steps,
        )

    def counterfactual(
        self,
        image: Image.Image,
        generated_ids: list[int],
        position: int,
        forced_prefix_ids: list[int],
        max_new_tokens: int | None = None,
        top_k: int = 5,
        progress: ProgressCallback | None = None,
        phase: str = "counterfactual",
    ) -> tuple[str, list[TokenStep]]:
        """Regenerate with `forced_prefix_ids` spliced in at `position` of the
        continuation. The forcing is done by appending to `input_ids` —
        `decoder_input_ids` is encoder-decoder only and
        `PrefixConstrainedLogitsProcessor` only forces one step at a time.

        Returns the decoded text (forced prefix included) and the per-token
        logprob trace of the tokens generated *after* the forced prefix — the
        forced tokens themselves aren't generated, so they carry no logprob here.
        """
        image_token_ids = self._image_token_ids()
        clashing = image_token_ids.intersection(forced_prefix_ids)
        if clashing:
            raise ValueError(f"forced prefix contains image token ids {clashing}")

        inputs = self._build_inputs(image)
        prompt_len = int(inputs["input_ids"].shape[-1])

        forced = list(generated_ids[:position]) + list(forced_prefix_ids)
        forced_t = torch.tensor(
            [forced],
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
        inputs["input_ids"] = torch.cat([inputs["input_ids"], forced_t], dim=-1)
        inputs["attention_mask"] = torch.cat(
            [
                inputs["attention_mask"],
                torch.ones_like(forced_t, dtype=inputs["attention_mask"].dtype),
            ],
            dim=-1,
        )
        forced_len = int(inputs["input_ids"].shape[-1])

        gen_kwargs = self._diagnostic_kwargs(max_new_tokens)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_logits=True,
                **self._progress_kwargs(progress, phase, gen_kwargs),
            )

        continuation_ids = [int(t) for t in output.sequences[0][forced_len:]]
        steps, _ = self._capture_steps(output.logits, continuation_ids, top_k)

        text = self._active_prefill() + self._decode(output.sequences[0][prompt_len:])
        return text, steps

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def _build_inputs(self, image: Image.Image) -> dict:
        """Model-specific prompt/processor call. Returns processor output
        already moved onto the model device."""
