import logging
import os
import threading
import traceback
from dataclasses import dataclass
from enum import Enum

from PIL import Image as PILImage

from risk_assessment.models.safety_model import Rating, SafetyModel

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    CHILLING = "Chilling 😎"
    LOADING = "Loading ⚙️"
    RUNNING = "Running 👀"
    DONE = "Done ✅"
    ERROR = "Error ❌"


class ModelLoadError(Exception):
    pass


@dataclass
class ModelState:
    name: str
    status: ModelStatus
    current_task: str = ""
    latest_rating: str = ""
    latest_image: str = ""


class Orchestrator:
    def __init__(self, model_classes: list[type[SafetyModel]] | None = None):
        if model_classes is None:
            from risk_assessment.models.model_LLavaguard import LavaGuard
            from risk_assessment.models.model_Pixtral import Pixtral
            from risk_assessment.models.model_Qwen3 import Qwen3

            model_classes = [LavaGuard, Qwen3, Pixtral]
        self.model_classes = model_classes
        self.models: dict[str, ModelState] = {
            cls.name: ModelState(cls.name, ModelStatus.CHILLING)
            for cls in model_classes
        }
        self.assessments: dict[str, list[Rating]] = {
            cls.name: [] for cls in model_classes
        }
        self.majority_votes: dict[str, str] = {}
        self._lock = threading.Lock()

    def get_state(self) -> dict[str, ModelState]:
        return self.models.copy()

    def update_model_state(self, model_name: str, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.models[model_name], key, value)

    def update_majority_votes_live(self) -> None:
        from risk_assessment.models.utils import calculate_majority_vote

        with self._lock:
            model_dicts = {
                name: {a.image: a.rating for a in lst}
                for name, lst in self.assessments.items()
            }
            all_images: set[str] = set()
            for d in model_dicts.values():
                all_images |= set(d.keys())

            for image_name in all_images:
                ratings = [
                    model_dicts[name][image_name]
                    for name in model_dicts
                    if image_name in model_dicts[name]
                ]
                if len(ratings) == len(self.model_classes):
                    self.majority_votes[image_name] = calculate_majority_vote(ratings)

    def _run_model(
        self, model_class: type[SafetyModel], image_path: str
    ) -> list[Rating]:
        from risk_assessment.models.utils import get_images

        return self._run_model_on_files(model_class, get_images(image_path))

    def _run_model_on_files(
        self, model_class: type[SafetyModel], image_files: list[str]
    ) -> list[Rating]:
        """Load MODEL_CLASS, run it over exactly IMAGE_FILES, unload it.

        Appends every rating to self.assessments[name] (the cumulative,
        process-lifetime record used by the TUI / write_combined_csv), but
        returns only the ratings produced by THIS call — callers processing
        a chunk get a chunk-scoped result, not the running total.
        """
        from risk_assessment.models.utils import model_device

        name = model_class.name
        self.update_model_state(
            name, status=ModelStatus.LOADING, current_task=f"Loading {name} model..."
        )

        model = model_class()
        try:
            model.load_model()
        except Exception as exc:
            self.update_model_state(
                name, status=ModelStatus.ERROR, current_task=f"Model load failed: {exc}"
            )
            raise ModelLoadError(str(exc)) from exc

        inner_model = getattr(model, "model", None)
        device = model_device(inner_model)
        logger.info("%s loaded on device: %s", name, device)
        if not device.startswith("cuda"):
            logger.warning(
                "%s is NOT running on GPU (device=%s) — this will be extremely slow "
                "and 4/8-bit quantized weights typically require CUDA to load at all.",
                name,
                device,
            )

        # device_map="auto"/a dict can silently split a model across devices
        # (e.g. offload some layers to CPU when Accelerate is conservative
        # about VRAM headroom) — checking only the first parameter's device,
        # as above, would miss that. hf_device_map records where every
        # module actually landed, so check all of it.
        device_map = getattr(inner_model, "hf_device_map", None)
        if device_map:
            offloaded = {
                module: dev
                for module, dev in device_map.items()
                if not str(dev).startswith("cuda")
            }
            if offloaded:
                logger.warning(
                    "%s has %d module(s) NOT on GPU: %s — this will be extremely "
                    "slow. Check available VRAM (nvidia-smi) before a full run.",
                    name,
                    len(offloaded),
                    offloaded,
                )

        self.update_model_state(name, current_task=f"Loaded on {device}")

        run_success = False
        ratings: list[Rating] = []
        try:
            self.update_model_state(
                name, status=ModelStatus.RUNNING, current_task="Processing images..."
            )
            logger.info("%s: processing %d images", name, len(image_files))

            for i, image_file in enumerate(image_files):
                image_name = os.path.basename(image_file)
                self.update_model_state(
                    name,
                    current_task=f"Processing image {i + 1}/{len(image_files)}",
                    latest_image=image_name,
                )
                try:
                    image = PILImage.open(image_file).convert("RGB")
                    assessment = model.process_image(image, image_name)
                except Exception as exc:
                    logger.warning(
                        "%s skipped %s: %s", name, image_name, exc, exc_info=True
                    )
                    self.update_model_state(
                        name, current_task=f"Skipped {image_name}: {exc}"
                    )
                    continue

                ratings.append(assessment)
                self.assessments[name].append(assessment)
                logger.info(
                    "%s [%d/%d] %s → %s",
                    name,
                    i + 1,
                    len(image_files),
                    image_name,
                    assessment.rating,
                )
                self.update_model_state(name, latest_rating=assessment.rating)
                self.update_majority_votes_live()

            run_success = True
            return ratings
        finally:
            self.update_model_state(name, current_task=f"Unloading {name} model...")
            model.unload_model()
            if run_success:
                self.update_model_state(
                    name, status=ModelStatus.DONE, current_task="Done"
                )

    def run_chunk(self, image_files: list[str]) -> dict[str, list[Rating]]:
        """Run every configured model over exactly IMAGE_FILES (a chunk),
        loading/unloading each model once for this chunk. Returns
        {model_name: [Rating, ...]} scoped to this chunk only — the shape
        expected by labelstudio_metadata_import.pipeline.combine_ratings.

        Raises ModelLoadError if a model fails to load; nothing is caught
        here, unlike run_all(), since the caller (which owns checkpointing)
        needs to know a chunk did not complete.
        """
        chunk_assessments: dict[str, list[Rating]] = {}
        for model_class in self.model_classes:
            chunk_assessments[model_class.name] = self._run_model_on_files(
                model_class, image_files
            )
        return chunk_assessments

    def run_all(self, path: str) -> None:
        from risk_assessment.models.utils import write_combined_csv

        def handle_fatal_load_error(exc: Exception) -> None:
            for model_name in self.models:
                self.update_model_state(
                    model_name,
                    status=ModelStatus.ERROR,
                    current_task=f"Aborted due to model load failure: {exc}",
                )

        for model_class in self.model_classes:
            try:
                self._run_model(model_class, path)
            except ModelLoadError as exc:
                logger.error(
                    "%s load error:\n%s", model_class.name, traceback.format_exc()
                )
                handle_fatal_load_error(exc)
                return
            except Exception as exc:
                logger.error(
                    "%s runtime error:\n%s", model_class.name, traceback.format_exc()
                )
                self.update_model_state(
                    model_class.name,
                    status=ModelStatus.ERROR,
                    current_task=f"Failed: {exc}",
                )

        write_combined_csv(self.assessments)

    def get_majority_votes(self) -> dict[str, str]:
        with self._lock:
            return self.majority_votes.copy()

    def run_test(self, image_path: str | None = None) -> None:
        import tempfile
        from pathlib import Path

        if image_path is None:
            tmp = tempfile.mkdtemp()
            test_image_path = str(Path(tmp) / "test.jpg")
            img = PILImage.new("RGB", (64, 64), color=(100, 149, 237))
            img.save(test_image_path)
            path = test_image_path
        else:
            path = image_path

        logger.info("Starting test run with: %s", path)
        self.run_all(path)
        logger.info("Test run complete. Majority votes: %s", self.majority_votes)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_test()
