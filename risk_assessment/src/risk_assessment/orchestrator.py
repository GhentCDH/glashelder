import logging
import traceback
from dataclasses import dataclass
from enum import Enum

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
    def __init__(self):
        self.models = {
            "model1": ModelState("LavaGuard", ModelStatus.CHILLING),
            "model2": ModelState("Qwen3", ModelStatus.CHILLING),
            "model3": ModelState("Pixtral", ModelStatus.CHILLING),
        }
        self.assessments_lavaguard = []
        self.assessments_qwen = []
        self.assessments_pixtral = []
        self.majority_votes = {}
        self._lock = __import__("threading").Lock()

    def update_status(self, model_name: str, new_status: ModelStatus):
        if model_name in self.models:
            current_name = self.models[model_name].name
            self.models[model_name] = ModelState(current_name, new_status)
            print(f"{model_name} status updated to: {new_status.value}")
        else:
            print(f"Model {model_name} not found.")

    def get_state(self):
        return self.models.copy()

    def update_model_state(self, model_id: str, **kwargs):
        for key, value in kwargs.items():
            setattr(self.models[model_id], key, value)

    def update_majority_votes_live(self):
        """Recalculate majority votes based on current assessments."""
        from risk_assessment.models.utils import calculate_majority_vote

        with self._lock:
            # Create dictionaries keyed by image name
            lava_dict = {
                a.get("image"): a.get("rating", "Unknown")
                for a in self.assessments_lavaguard
            }
            qwen_dict = {
                a.get("image"): a.get("rating", "Unknown")
                for a in self.assessments_qwen
            }
            pixtral_dict = {
                a.get("image"): a.get("rating", "Unknown")
                for a in self.assessments_pixtral
            }

            # Get all image names
            all_images = (
                set(lava_dict.keys()) | set(qwen_dict.keys()) | set(pixtral_dict.keys())
            )

            # Calculate majority for images that have all three assessments
            for image_name in all_images:
                if (
                    image_name in lava_dict
                    and image_name in qwen_dict
                    and image_name in pixtral_dict
                ):
                    majority = calculate_majority_vote(
                        lava_dict[image_name],
                        qwen_dict[image_name],
                        pixtral_dict[image_name],
                    )
                    self.majority_votes[image_name] = majority

    def run_all(self, path):
        from risk_assessment.models.model_LLavaguard import run_lavaguard
        from risk_assessment.models.model_Pixtral import run_pixtral
        from risk_assessment.models.model_Qwen3 import run_qwen
        from risk_assessment.models.utils import write_combined_csv

        def handle_fatal_load_error(exc: Exception) -> None:
            for model_id in self.models:
                self.update_model_state(
                    model_id,
                    status=ModelStatus.ERROR,
                    current_task=f"Aborted due to model load failure: {exc}",
                )

        try:
            run_lavaguard(self, path)
        except ModelLoadError as exc:
            logger.error("LLavaGuard load error:\n%s", traceback.format_exc())
            handle_fatal_load_error(exc)
            return
        except Exception as exc:
            logger.error("LLavaGuard runtime error:\n%s", traceback.format_exc())
            self.update_model_state(
                "model1",
                status=ModelStatus.ERROR,
                current_task=f"Failed: {exc}",
            )

        try:
            run_qwen(self, path)
        except ModelLoadError as exc:
            logger.error("Qwen load error:\n%s", traceback.format_exc())
            handle_fatal_load_error(exc)
            return
        except Exception as exc:
            logger.error("Qwen runtime error:\n%s", traceback.format_exc())
            self.update_model_state(
                "model2",
                status=ModelStatus.ERROR,
                current_task=f"Failed: {exc}",
            )

        try:
            run_pixtral(self, path)
        except ModelLoadError as exc:
            logger.error("Pixtral load error:\n%s", traceback.format_exc())
            handle_fatal_load_error(exc)
            return
        except Exception as exc:
            logger.error("Pixtral runtime error:\n%s", traceback.format_exc())
            self.update_model_state(
                "model3",
                status=ModelStatus.ERROR,
                current_task=f"Failed: {exc}",
            )

        # Write combined results to CSV
        write_combined_csv(
            self.assessments_lavaguard, self.assessments_qwen, self.assessments_pixtral
        )

    def get_majority_votes(self):
        """Get the dictionary of majority votes by image name."""
        with self._lock:
            return self.majority_votes.copy()

    def run_test(self, image_path: str | None = None):
        """Run the full pipeline on a single image for debugging.

        If *image_path* is None a small synthetic JPEG is created in a
        temporary directory and used as the test image.  Pass an explicit
        path to test with a real image instead.
        """

        from PIL import Image as PILImage

        if image_path is None:
            img = PILImage.new("RGB", (64, 64), color=(100, 149, 237))
            test_image_path = "/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop/sn01004006_tiff.jpg"
            img.save(test_image_path)
            logger.info("Test image created at: %s", test_image_path)
            path = test_image_path
        else:
            path = image_path

        logger.info("Starting test run with: %s", path)
        self.run_all(path)
        logger.info("Test run complete. Majority votes: %s", self.majority_votes)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_test()
