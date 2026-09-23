import logging
import os
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from PIL import Image as PILImage

from glaswacht.models.safety_model import Rating, SafetyModel

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
            from glaswacht.models.model_LLavaguard import LavaGuard
            from glaswacht.models.model_Pixtral import Pixtral
            from glaswacht.models.model_Qwen3 import Qwen3

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
        self._ratings_by_image: dict[str, dict[str, str]] = defaultdict(dict)
        self._lock = threading.Lock()

    def get_state(self) -> dict[str, ModelState]:
        return self.models.copy()

    def update_model_state(self, model_name: str, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.models[model_name], key, value)

    def _record_rating(self, model_name: str, image_name: str, rating: str) -> None:
        from glaswacht.models.utils import calculate_majority_vote

        with self._lock:
            per_image = self._ratings_by_image[image_name]
            per_image[model_name] = rating
            if len(per_image) == len(self.model_classes):
                self.majority_votes[image_name] = calculate_majority_vote(
                    list(per_image.values())
                )

    def _run_model(
        self, model_class: type[SafetyModel], image_path: str
    ) -> list[Rating]:
        from glaswacht.models.utils import get_images

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

        run_success = False
        try:
            self.update_model_state(
                name, status=ModelStatus.RUNNING, current_task="Processing images..."
            )
            image_files = get_images(image_path)
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
                self._record_rating(name, image_name, assessment.rating)

            run_success = True
            return self.assessments[name]
        finally:
            self.update_model_state(name, current_task=f"Unloading {name} model...")
            model.unload_model()
            if run_success:
                self.update_model_state(
                    name, status=ModelStatus.DONE, current_task="Done"
                )

    def run_all(self, path: str) -> None:
        from glaswacht.models.utils import write_combined_csv

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
