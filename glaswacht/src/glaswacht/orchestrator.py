import logging
import os
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from PIL import Image as PILImage

from glaswacht.models.safety_model import Rating, SafetyModel

logger = logging.getLogger(__name__)

# the logger every glaswacht module logs under, written to each run's log file
_PACKAGE_LOGGER = logging.getLogger("glaswacht")


class ModelStatus(Enum):
    WAITING = "Waiting ⏳"
    LOADING = "Loading ⚙️"
    PROCESSING = "Processing 🔎"
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
    skipped: int = 0


@dataclass
class Skip:
    """An image a model could not assess."""

    model: str
    image: str
    error: str


class Orchestrator:
    def __init__(self, model_classes: list[type[SafetyModel]] | None = None):
        if model_classes is None:
            from glaswacht.models.model_Gemma3 import Gemma3
            from glaswacht.models.model_LLavaguard import LavaGuard
            from glaswacht.models.model_Pixtral import Pixtral
            from glaswacht.models.model_Qwen3 import Qwen3

            model_classes = [
                cls
                for cls in [Qwen3, Pixtral, Gemma3, LavaGuard]
                if cls.enabled_by_default
            ]
        self.model_classes = model_classes
        self._lock = threading.Lock()
        # the files of the latest run; results_csv is None until it is written
        self.log_path: Path | None = None
        self.results_csv: Path | None = None
        self.reset()

    def reset(self) -> None:
        """Forget everything a previous run processed."""
        with self._lock:
            self.models: dict[str, ModelState] = {
                cls.name: ModelState(cls.name, ModelStatus.WAITING)
                for cls in self.model_classes
            }
            self.assessments: dict[str, list[Rating]] = {
                cls.name: [] for cls in self.model_classes
            }
            self.majority_votes: dict[str, str] = {}
            self._ratings_by_image: dict[str, dict[str, str]] = defaultdict(dict)
            self.skips: list[Skip] = []

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
                name, status=ModelStatus.PROCESSING, current_task="Processing images..."
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
                    # CUDA errors span many lines, the first one says what went wrong
                    error = (
                        f"{type(exc).__name__}: {next(iter(str(exc).splitlines()), '')}"
                    )
                    with self._lock:
                        self.skips.append(Skip(name, image_name, error))
                    self.update_model_state(
                        name,
                        current_task=f"Skipped {image_name}: {error}",
                        skipped=self.models[name].skipped + 1,
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

    def run_all(self, path: str, output_dir: str | Path) -> Path | None:
        """Run every model on the images at `path`, starting from a clean state.

        Writes the run's log and results CSV to `output_dir`, named after the
        date and time the run started. Returns the CSV, or None when a model
        failed to load and the run was aborted.
        """
        from glaswacht.models.utils import output_folder, run_files

        self.reset()
        output_dir = output_folder(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_csv, self.log_path = run_files(output_dir, datetime.now())
        self.results_csv = None

        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s")
        )
        handler.setLevel(logging.INFO)
        if _PACKAGE_LOGGER.getEffectiveLevel() > logging.INFO:
            _PACKAGE_LOGGER.setLevel(logging.INFO)
        _PACKAGE_LOGGER.addHandler(handler)
        try:
            logger.info(
                "Run started: images=%s, models=%s",
                path,
                ", ".join(cls.name for cls in self.model_classes),
            )
            if self._run_models(path):
                self._write_results(results_csv)
                self.results_csv = results_csv
            return self.results_csv
        finally:
            _PACKAGE_LOGGER.removeHandler(handler)
            handler.close()

    def _write_results(self, results_csv: Path) -> None:
        from glaswacht.models.utils import write_combined_csv

        write_combined_csv(self.assessments, results_csv)
        logger.info("Results written to %s", results_csv)

    def _run_models(self, path: str) -> bool:
        """Run the models one after another. False when one failed to load,
        which aborts the run."""

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
                return False
            except Exception as exc:
                logger.error(
                    "%s runtime error:\n%s", model_class.name, traceback.format_exc()
                )
                self.update_model_state(
                    model_class.name,
                    status=ModelStatus.ERROR,
                    current_task=f"Failed: {exc}",
                )
        return True

    def get_majority_votes(self) -> dict[str, str]:
        with self._lock:
            return self.majority_votes.copy()

    def get_skips(self) -> list[Skip]:
        with self._lock:
            return self.skips.copy()

    def run_test(
        self, image_path: str | None = None, output_dir: str | Path = "glaswacht_output"
    ) -> None:
        import tempfile

        if image_path is None:
            tmp = tempfile.mkdtemp()
            test_image_path = str(Path(tmp) / "test.jpg")
            img = PILImage.new("RGB", (64, 64), color=(100, 149, 237))
            img.save(test_image_path)
            path = test_image_path
        else:
            path = image_path

        logger.info("Starting test run with: %s", path)
        self.run_all(path, output_dir)
        logger.info("Test run complete. Majority votes: %s", self.majority_votes)
