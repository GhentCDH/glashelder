import functools
import logging
import threading
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import override

from PIL import Image as PILImage
from rich.markup import escape, render

from glaswacht.ledger import Ledger
from glaswacht.models.safety_model import Rating, SafetyModel

logger = logging.getLogger(__name__)

# the logger every glaswacht module logs under, written to each run's log file
_PACKAGE_LOGGER = logging.getLogger("glaswacht")

# how much of a rationale the per-image log line shows; the CSV has all of it
_EXCERPT_LENGTH = 300

# the ledger's name in the output folder when no other one is given
LEDGER_NAME = "ledger.jsonl"


class _PlainFormatter(logging.Formatter):
    """Writes records logged with Rich markup (`extra={"markup": True}`) as
    plain text, so the log file holds no markup tags."""

    @override
    def formatMessage(self, record: logging.LogRecord) -> str:
        text = super().formatMessage(record)
        return render(text).plain if getattr(record, "markup", False) else text


def _excerpt(text: str, length: int = _EXCERPT_LENGTH) -> str:
    """The start of `text` on one line, cut at a word and marked with … when
    there is more."""
    text = " ".join(text.split())
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + " …"


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


@dataclass
class _Work:
    """An image of this run and the models that still have to assess it."""

    file: str
    key: str
    models: list[str]


@dataclass
class RunSummary:
    """How far the image folder is, counted in images."""

    # images in the folder
    total: int = 0
    # images every model has assessed, or given up on, before this run
    finished_before: int = 0
    # images this run takes, at most --max-images
    selected: int = 0
    # images left for a later run once this one ended
    remaining: int = 0
    # images this run finished: every model has now assessed or given up on them
    finished: int = 0
    # images at least one model gave up on after repeated failures
    given_up: list[str] = field(default_factory=list)
    # True when the run was stopped early (a signal from the scheduler)
    stopped: bool = False


class Orchestrator:
    def __init__(
        self,
        model_classes: list[type[SafetyModel]] | None = None,
        full_precision: bool = False,
    ):
        if model_classes is None:
            from glaswacht.models.model_Gemma3 import Gemma3
            from glaswacht.models.model_LLavaguard import LavaGuard
            from glaswacht.models.model_Ministral3 import Ministral3
            from glaswacht.models.model_Pixtral import Pixtral
            from glaswacht.models.model_Qwen3 import Qwen3

            model_classes = [
                cls
                for cls in [Qwen3, Ministral3, Gemma3, Pixtral, LavaGuard]
                if cls.enabled_by_default
            ]
        self.model_classes = model_classes
        # load the models without quantization where they have such a version
        self.full_precision = full_precision
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # the files of the latest run; results_csv is None until it is written
        self.log_path: Path | None = None
        self.results_csv: Path | None = None
        self.ledger: Ledger | None = None
        self._run_name = ""
        # True when the latest run was aborted because a model failed to load
        self.aborted = False
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
            # per image, the models whose rating its majority vote waits for
            self._voters: dict[str, set[str]] = {}
            self.skips: list[Skip] = []
            self.summary = RunSummary()

    def request_stop(self) -> None:
        """Stop the run after the image being assessed now; what is done so
        far is still written out. Safe to call from a signal handler."""
        self._stop.set()

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
            voters = self._voters.get(image_name, set(self.models))
            if voters and voters <= per_image.keys():
                self.majority_votes[image_name] = calculate_majority_vote(
                    [per_image[m] for m in voters]
                )

    # --- what to do ---------------------------------------------------------

    def _plan(self, path: str, max_images: int | None, recursive: bool) -> list[_Work]:
        """The images this run assesses, with the models each still needs,
        in folder order. Ratings from earlier runs count towards the
        majority votes of these images."""
        from glaswacht.models.utils import get_images, image_key

        assert self.ledger is not None
        names = [cls.name for cls in self.model_classes]
        files = get_images(path, recursive=recursive)
        todo: list[_Work] = []
        given_up: list[str] = []
        for file in files:
            key = image_key(path, file)
            if any(self.ledger.gave_up(key, m) for m in names):
                given_up.append(key)
            pending = self.ledger.pending(key, names)
            if pending:
                todo.append(_Work(file, key, pending))

        selected = todo if max_images is None else todo[:max_images]
        for work in selected:
            self._voters[work.key] = {
                m for m in names if not self.ledger.gave_up(work.key, m)
            }
            for m in names:
                earlier = self.ledger.rating(work.key, m)
                if earlier is not None:
                    self._record_rating(m, work.key, earlier.rating)

        self.summary = RunSummary(
            total=len(files),
            finished_before=len(files) - len(todo),
            selected=len(selected),
            remaining=len(todo),
            given_up=given_up,
        )
        logger.info(
            "%d images: %d finished in earlier runs, %d to do, %d of them this run",
            len(files),
            self.summary.finished_before,
            len(todo),
            len(selected),
        )
        if given_up:
            logger.warning(
                "%d images given up on by a model after repeated failures, "
                "see the ledger %s",
                len(given_up),
                self.ledger.path,
            )
        return selected

    # --- one model, one image ----------------------------------------------

    def _load(self, model_class: type[SafetyModel]) -> SafetyModel:
        name = model_class.name
        self.update_model_state(
            name, status=ModelStatus.LOADING, current_task=f"Loading {name} model..."
        )
        model = model_class(full_precision=self.full_precision)
        try:
            model.load_model()
        except Exception as exc:
            self.update_model_state(
                name, status=ModelStatus.ERROR, current_task=f"Model load failed: {exc}"
            )
            # free whatever part of it did load before the run aborts
            try:
                model.unload_model()
            except Exception:
                logger.warning("%s: unloading after the failed load failed", name)
            raise ModelLoadError(str(exc)) from exc
        self.update_model_state(
            name, status=ModelStatus.PROCESSING, current_task="Processing images..."
        )
        return model

    def _unload(self, model: SafetyModel, done: bool) -> None:
        name = model.name
        self.update_model_state(name, current_task=f"Unloading {name} model...")
        model.unload_model()
        if done:
            self.update_model_state(name, status=ModelStatus.DONE, current_task="Done")

    def _assess(
        self,
        model: SafetyModel,
        open_image: Callable[[], PILImage.Image],
        key: str,
        number: int,
        count: int,
    ) -> None:
        """Have `model` assess one image, the `number`th of its `count` this
        run, and write the outcome to the ledger straight away."""
        assert self.ledger is not None
        name = model.name
        context = {
            "model_id": model.model_id,
            "precision": model.precision,
            "run": self._run_name,
        }
        self.update_model_state(
            name,
            current_task=f"Processing image {number}/{count}",
            latest_image=key,
        )
        try:
            assessment = model.process_image(open_image(), key)
        except Exception as exc:
            logger.warning("%s skipped %s: %s", name, key, exc, exc_info=True)
            # CUDA errors span many lines, the first one says what went wrong
            error = f"{type(exc).__name__}: {next(iter(str(exc).splitlines()), '')}"
            self.ledger.record_failure(key, name, error, **context)
            with self._lock:
                self.skips.append(Skip(name, key, error))
            self.update_model_state(
                name,
                current_task=f"Skipped {key}: {error}",
                skipped=self.models[name].skipped + 1,
            )
            return

        self.ledger.record_done(name, assessment, **context)
        self.assessments[name].append(assessment)
        # Rich markup for a bold rating; the log file gets it plain
        logger.info(
            "%s \\[%d/%d] %s → [bold]%s[/bold] | %s\n    %s",
            name,
            number,
            count,
            escape(key),
            escape(assessment.rating),
            escape(assessment.category),
            escape(_excerpt(assessment.rationale)),
            extra={"markup": True},
        )
        self.update_model_state(name, latest_rating=assessment.rating)
        self._record_rating(name, key, assessment.rating)

    @staticmethod
    def _opener(file: str) -> Callable[[], PILImage.Image]:
        """Opens `file` once however many models ask for it; a failure is
        not kept, so every model records its own."""
        return functools.cache(lambda: PILImage.open(file).convert("RGB"))

    # --- the two ways to run ------------------------------------------------

    def _run_one_at_a_time(self, work: list[_Work]) -> None:
        """Load the models one after another, each assessing all its images
        before the next loads: for a GPU that holds one model at a time."""
        for model_class in self.model_classes:
            name = model_class.name
            mine = [w for w in work if name in w.models]
            if not mine:
                self.update_model_state(
                    name, status=ModelStatus.DONE, current_task="Nothing to do"
                )
                continue
            if self._stop.is_set():
                return
            model = self._load(model_class)
            done = False
            try:
                logger.info("%s: processing %d images", name, len(mine))
                for number, w in enumerate(mine, start=1):
                    if self._stop.is_set():
                        break
                    self._assess(model, self._opener(w.file), w.key, number, len(mine))
                else:
                    done = True
            except Exception as exc:
                logger.error("%s runtime error:\n%s", name, traceback.format_exc())
                self.update_model_state(
                    name, status=ModelStatus.ERROR, current_task=f"Failed: {exc}"
                )
            finally:
                self._unload(model, done)

    def _run_all_loaded(self, work: list[_Work]) -> None:
        """Load every model at once, then take the images one by one, each
        assessed by all models that still need it: for a GPU that holds
        them all. A stopped run leaves no image half done."""
        needed = {m for w in work for m in w.models}
        loaded: dict[str, SafetyModel] = {}
        counts = {m: sum(m in w.models for w in work) for m in needed}
        numbers = dict.fromkeys(needed, 0)
        finished = False
        try:
            for model_class in self.model_classes:
                if model_class.name in needed:
                    loaded[model_class.name] = self._load(model_class)
                else:
                    self.update_model_state(
                        model_class.name,
                        status=ModelStatus.DONE,
                        current_task="Nothing to do",
                    )
            for w in work:
                if self._stop.is_set():
                    break
                open_image = self._opener(w.file)
                for name in w.models:
                    numbers[name] += 1
                    self._assess(
                        loaded[name], open_image, w.key, numbers[name], counts[name]
                    )
            else:
                finished = True
        finally:
            for model in loaded.values():
                self._unload(model, finished)

    # --- a run ---------------------------------------------------------------

    def run_all(
        self,
        path: str,
        output_dir: str | Path,
        *,
        max_images: int | None = None,
        load_all: bool = False,
        ledger_path: str | Path | None = None,
        recursive: bool = False,
    ) -> Path | None:
        """Run every model on the images at `path` that the ledger says it
        has not assessed yet, at most `max_images` of them.

        Writes the run's log and results CSV to `output_dir`, named after the
        date and time the run started, and every assessment to the ledger,
        `ledger.jsonl` in `output_dir` unless `ledger_path` names another.
        `load_all` keeps all models in GPU memory at once and goes image by
        image. Returns the CSV, or None when there was nothing to do or a
        model failed to load (then `aborted` is True).
        """
        from glaswacht.models.utils import image_path, output_folder, run_files

        # before any model loads: a wrong path would otherwise load every
        # model only to find no images, and write an empty results CSV
        resolved = image_path(path)
        if resolved is None:
            raise FileNotFoundError(f"image folder or file not found: {path}")
        path = str(resolved)

        self.reset()
        self._stop.clear()
        self.aborted = False
        output_dir = output_folder(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_csv, self.log_path = run_files(output_dir, datetime.now())
        self._run_name = results_csv.stem
        self.results_csv = None
        self.ledger = Ledger(
            output_folder(ledger_path) if ledger_path else output_dir / LEDGER_NAME
        )

        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(
            _PlainFormatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s")
        )
        handler.setLevel(logging.INFO)
        if _PACKAGE_LOGGER.getEffectiveLevel() > logging.INFO:
            _PACKAGE_LOGGER.setLevel(logging.INFO)
        _PACKAGE_LOGGER.addHandler(handler)
        try:
            logger.info(
                "Run started: images=%s, models=%s, ledger=%s, %s, %s, max images=%s",
                path,
                ", ".join(cls.name for cls in self.model_classes),
                self.ledger.path,
                "full precision" if self.full_precision else "quantized where set up",
                "all models loaded at once" if load_all else "one model at a time",
                max_images if max_images is not None else "no limit",
            )
            work = self._plan(path, max_images, recursive)
            if not work:
                logger.info("Nothing to do: every image is finished")
                for name in self.models:
                    self.update_model_state(
                        name, status=ModelStatus.DONE, current_task="Nothing to do"
                    )
                return None

            try:
                if load_all:
                    self._run_all_loaded(work)
                else:
                    self._run_one_at_a_time(work)
            except ModelLoadError as exc:
                logger.error("Model load error:\n%s", traceback.format_exc())
                for name, state in self.models.items():
                    if state.status != ModelStatus.ERROR:
                        self.update_model_state(
                            name,
                            status=ModelStatus.ERROR,
                            current_task=f"Aborted due to model load failure: {exc}",
                        )
                self.aborted = True

            self._finish_summary(path, recursive)
            if self.aborted:
                return None
            self._write_results(results_csv, [w.key for w in work])
            self.results_csv = results_csv
            return self.results_csv
        finally:
            _PACKAGE_LOGGER.removeHandler(handler)
            handler.close()

    def _finish_summary(self, path: str, recursive: bool) -> None:
        from glaswacht.models.utils import get_images, image_key

        assert self.ledger is not None
        names = [cls.name for cls in self.model_classes]
        self.summary.remaining = sum(
            bool(self.ledger.pending(image_key(path, f), names))
            for f in get_images(path, recursive=recursive)
        )
        self.summary.finished = (
            self.summary.total - self.summary.finished_before - self.summary.remaining
        )
        self.summary.stopped = self._stop.is_set()
        logger.info(
            "Run %s: %d images finished, %d left for a later run",
            "stopped early" if self.summary.stopped else "ended",
            self.summary.finished,
            self.summary.remaining,
        )

    def _write_results(self, results_csv: Path, keys: list[str]) -> None:
        """The CSV of this run's images, with every model's rating, including
        those made in earlier runs."""
        from glaswacht.models.utils import write_combined_csv

        assert self.ledger is not None
        ledger = self.ledger
        assessments = {
            cls.name: [
                r for key in keys if (r := ledger.rating(key, cls.name)) is not None
            ]
            for cls in self.model_classes
        }
        write_combined_csv(assessments, results_csv)
        logger.info("Results written to %s", results_csv)

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

        tmp = tempfile.mkdtemp()
        if image_path is None:
            test_image_path = str(Path(tmp) / "test.jpg")
            img = PILImage.new("RGB", (64, 64), color=(100, 149, 237))
            img.save(test_image_path)
            path = test_image_path
        else:
            path = image_path

        logger.info("Starting test run with: %s", path)
        # a ledger of its own, so a test never marks real images as done
        self.run_all(path, output_dir, ledger_path=Path(tmp) / LEDGER_NAME)
        logger.info("Test run complete. Majority votes: %s", self.majority_votes)
