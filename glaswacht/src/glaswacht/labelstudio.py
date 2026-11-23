"""Compare glaswacht runs with human annotations from a Label Studio export.

`select_disagreements` picks images where the annotators and the old models
disagreed, and `write_selection` writes them out as an image folder that
`glaswacht run` can assess, plus a reference CSV with the human ratings.
`compare` then scores a run's results CSV against that reference.

The export (project 75) holds, per task, the image as a base64 data URI, the
ratings of the three models that ran then (`model_<n>_rating`), and per
annotator their own rating (`eigen_classificatie`) and note
(`eigen_toelichting`), plus whether they agreed with each model
(`model<n>_akkoord`).
"""

import base64
import csv
import json
import mimetypes
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from glaswacht.models.safety_model import RATINGS

# how severe each rating is, for majorities, ties and the direction of a miss
SEVERITY = {rating: level for level, rating in enumerate(RATINGS)}

OLD_MODELS = (1, 2, 3)

REFERENCE_FIELDS = [
    "image",
    "task_id",
    "human_rating",
    "human_ratings",
    "annotators",
    "old_majority",
    "disagreement",
    *(f"old_model_{n}_rating" for n in OLD_MODELS),
    "annotator_notes",
]


def majority(ratings: list[str]) -> str | None:
    """The most common rating; a tie goes to the more severe one, as the
    majority vote of a run does. None without any known rating."""
    known = [r for r in ratings if r in SEVERITY]
    if not known:
        return None
    counts = Counter(known)
    top = max(counts.values())
    return max((r for r, n in counts.items() if n == top), key=SEVERITY.get)


def direction(reference: str, rating: str) -> str:
    """How `rating` misses `reference`: agree, too strict or too lenient."""
    if rating not in SEVERITY:
        return "no answer"
    if rating == reference:
        return "agree"
    return "too strict" if SEVERITY[rating] > SEVERITY[reference] else "too lenient"


@dataclass
class Task:
    task_id: int
    image_uri: str = field(repr=False)
    human_ratings: list[str]
    old_ratings: list[str]
    notes: list[str]
    # annotators who explicitly disagreed ("nee") with an old model
    rejections: int

    @property
    def human_rating(self) -> str | None:
        return majority(self.human_ratings)

    @property
    def unanimous(self) -> bool:
        return len(set(self.human_ratings)) == 1

    @property
    def old_majority(self) -> str | None:
        return majority(self.old_ratings)

    @property
    def disagreement(self) -> str:
        """e.g. "human Safe, models Unsafe"."""
        return f"human {self.human_rating}, models {self.old_majority}"

    @property
    def image_name(self) -> str:
        mime = self.image_uri.split(";", 1)[0].removeprefix("data:")
        extension = mimetypes.guess_extension(mime) or ".jpg"
        return f"task_{self.task_id}{'.jpg' if extension == '.jpeg' else extension}"


def load_tasks(export_path: Path) -> list[Task]:
    """The annotated tasks of a Label Studio JSON export."""
    with open(export_path, encoding="utf-8") as f:
        export = json.load(f)
    tasks = []
    for item in export:
        human, notes, rejections = [], [], 0
        for annotation in item["annotations"]:
            if annotation["was_cancelled"]:
                continue
            for result in annotation["result"]:
                name, value = result["from_name"], result["value"]
                if name == "eigen_classificatie":
                    human.extend(value["choices"])
                elif name == "eigen_toelichting":
                    notes.extend(t for t in value["text"] if t.strip())
                elif name.endswith("_akkoord") and value["choices"] == ["nee"]:
                    rejections += 1
        if not human:
            continue
        data = item["data"]
        tasks.append(
            Task(
                task_id=item["id"],
                image_uri=data["image"],
                human_ratings=human,
                old_ratings=[data.get(f"model_{n}_rating", "") for n in OLD_MODELS],
                notes=notes,
                rejections=rejections,
            )
        )
    return tasks


def select_disagreements(tasks: list[Task], count: int) -> list[Task]:
    """`count` tasks where the annotators agree with each other but not with
    the old models' majority.

    The tasks are grouped by kind of disagreement (e.g. "human Safe, models
    Unsafe") and taken from each group in turn, so a selection covers models
    being too strict and too lenient instead of only the most common case.
    Within a group, tasks with more annotators and more explicit rejections
    of the models come first. The selection is the same every time.
    """
    candidates = [
        t
        for t in tasks
        if t.unanimous
        and t.old_majority is not None
        and t.human_rating != t.old_majority
    ]
    groups: dict[str, list[Task]] = {}
    for task in sorted(
        candidates,
        key=lambda t: (-len(t.human_ratings), -t.rejections, t.task_id),
    ):
        groups.setdefault(task.disagreement, []).append(task)
    # the largest groups first, so leftovers go to the most common cases
    queues = sorted(groups.values(), key=len, reverse=True)
    selected: list[Task] = []
    while len(selected) < count and any(queues):
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.pop(0))
    return selected


def write_selection(tasks: list[Task], out_dir: Path) -> Path:
    """Write the tasks' images to `out_dir`/images and their human and old
    model ratings to `out_dir`/reference.csv, which it returns."""
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    reference = out_dir / "reference.csv"
    with open(reference, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REFERENCE_FIELDS)
        writer.writeheader()
        for task in tasks:
            encoded = task.image_uri.split(",", 1)[1]
            (images / task.image_name).write_bytes(base64.b64decode(encoded))
            writer.writerow(
                {
                    "image": task.image_name,
                    "task_id": task.task_id,
                    "human_rating": task.human_rating,
                    "human_ratings": "; ".join(task.human_ratings),
                    "annotators": len(task.human_ratings),
                    "old_majority": task.old_majority,
                    "disagreement": task.disagreement,
                    **{
                        f"old_model_{n}_rating": rating
                        for n, rating in zip(OLD_MODELS, task.old_ratings, strict=True)
                    },
                    "annotator_notes": " | ".join(task.notes),
                }
            )
    return reference


@dataclass
class Score:
    agree: int = 0
    too_strict: int = 0
    too_lenient: int = 0
    no_answer: int = 0

    def add(self, outcome: str) -> None:
        name = outcome.replace(" ", "_")
        setattr(self, name, getattr(self, name) + 1)

    @property
    def total(self) -> int:
        return self.agree + self.too_strict + self.too_lenient + self.no_answer


@dataclass
class Answer:
    """One rating of an image, scored against the annotators."""

    rating: str
    outcome: str
    category: str = ""
    rationale: str = ""


def _scored(human: str, rating: str, category: str = "", rationale: str = "") -> Answer:
    return Answer(rating, direction(human, rating), category, rationale)


@dataclass
class ImageComparison:
    image: str
    task_id: str
    human_rating: str
    human_ratings: list[str]
    notes: list[str]
    disagreement: str
    old_models: list[Answer]
    old_majority: Answer
    new_models: dict[str, Answer]
    new_majority: Answer

    @property
    def any_new_model_differs(self) -> bool:
        return any(a.outcome != "agree" for a in self.new_models.values())


@dataclass
class Comparison:
    reference_csv: Path
    results_csv: Path
    models: list[str]
    images: list[ImageComparison]
    # reference images the run has no results for
    missing: list[str]

    @property
    def scores(self) -> dict[str, Score]:
        """Per scored column, in display order: old majority, every new
        model, new majority."""
        columns = {"old majority": [i.old_majority for i in self.images]}
        for model in self.models:
            columns[model] = [i.new_models[model] for i in self.images]
        columns["new majority"] = [i.new_majority for i in self.images]
        scores = {}
        for label, answers in columns.items():
            score = scores[label] = Score()
            for answer in answers:
                score.add(answer.outcome)
        return scores

    @property
    def by_disagreement(self) -> list[tuple[str, Score]]:
        """How the new majority does per kind of old disagreement."""
        kinds: dict[str, Score] = {}
        for image in self.images:
            kinds.setdefault(image.disagreement, Score()).add(
                image.new_majority.outcome
            )
        return sorted(kinds.items(), key=lambda item: -item[1].total)

    @property
    def confusion(self) -> dict[str, dict[str, int]]:
        """Images per annotator rating (rows) and new majority (columns)."""
        answers = [*RATINGS, "no answer"]
        table = {human: dict.fromkeys(answers, 0) for human in RATINGS}
        for image in self.images:
            column = image.new_majority.rating
            table[image.human_rating][
                column if column in SEVERITY else "no answer"
            ] += 1
        return table


def build_comparison(reference_csv: Path, results_csv: Path) -> Comparison:
    """Score a run's results CSV against a reference.csv from
    `write_selection`. Raises ValueError when they share no images."""
    with open(reference_csv, newline="", encoding="utf-8") as f:
        reference = {row["image"]: row for row in csv.DictReader(f)}
    with open(results_csv, newline="", encoding="utf-8") as f:
        results = {row["image"]: row for row in csv.DictReader(f)}

    shared = [image for image in reference if image in results]
    if not shared:
        raise ValueError(f"no images of {reference_csv} are in {results_csv}")
    models = [
        column.removesuffix("_rating")
        for column in next(iter(results.values()))
        if column.endswith("_rating")
    ]

    images = []
    for name in shared:
        ref, run = reference[name], results[name]
        human = ref["human_rating"]

        images.append(
            ImageComparison(
                image=name,
                task_id=ref["task_id"],
                human_rating=human,
                human_ratings=ref["human_ratings"].split("; "),
                notes=[n for n in ref["annotator_notes"].split(" | ") if n],
                disagreement=ref["disagreement"],
                old_models=[
                    _scored(human, ref[f"old_model_{n}_rating"]) for n in OLD_MODELS
                ],
                old_majority=_scored(human, ref["old_majority"]),
                new_models={
                    m: _scored(
                        human,
                        run[f"{m}_rating"],
                        run.get(f"{m}_category", ""),
                        run.get(f"{m}_rationale", ""),
                    )
                    for m in models
                },
                new_majority=_scored(human, run["majority_vote"]),
            )
        )
    missing = [image for image in reference if image not in results]
    return Comparison(reference_csv, results_csv, models, images, missing)


def text_report(comparison: Comparison) -> str:
    """The comparison as a plain-text summary and table per image."""
    c = comparison
    lines = [
        f"Compared {len(c.images)} images of {c.reference_csv.name} "
        f"with {c.results_csv.name}"
    ]
    if c.missing:
        lines.append(f"Not in the results ({len(c.missing)}): {', '.join(c.missing)}")
    lines += [
        "",
        f"{'':14} {'agree':>7} {'too strict':>11} {'too lenient':>12} {'no answer':>10}",
    ]
    scores = c.scores
    for label, score in scores.items():
        lines.append(
            f"{label:14} {score.agree:>3}/{score.total:<3} {score.too_strict:>11} "
            f"{score.too_lenient:>12} {score.no_answer:>10}"
        )

    short = {"Safe": "Safe", "Potentially Problematic": "PotProb", "Unsafe": "Unsafe"}
    labels = list(scores)
    lines += [
        "",
        f"{'image':18} {'human':8} " + " ".join(f"{label[:12]:12}" for label in labels),
    ]
    for image in c.images:
        answers = [image.old_majority, *image.new_models.values(), image.new_majority]
        cells = [
            f"{short.get(a.rating, a.rating or '-') + ('' if a.outcome == 'agree' else '*'):12}"
            for a in answers
        ]
        lines.append(
            f"{Path(image.image).stem:18} {short[image.human_rating]:8} "
            + " ".join(cells)
        )
    lines += ["", "* differs from the human rating"]
    return "\n".join(lines)


# longest side of the image previews embedded in the HTML report
_PREVIEW_SIZE = 700


def _preview(path: Path) -> str | None:
    """A small JPEG of the image as a data URI, so the report is one file."""
    from io import BytesIO

    from PIL import Image

    if not path.is_file():
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((_PREVIEW_SIZE, _PREVIEW_SIZE))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def write_html_report(comparison: Comparison, path: Path) -> Path:
    """Write the comparison as one self-contained HTML page: summary, the
    new majority per kind of disagreement and against the annotators, and
    every image with all ratings, categories and rationales."""
    from datetime import datetime

    from jinja2 import Environment, FileSystemLoader, select_autoescape

    images_dir = comparison.reference_csv.parent / "images"
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    html = environment.get_template("compare.html").render(
        c=comparison,
        ratings=RATINGS,
        previews={i.image: _preview(images_dir / i.image) for i in comparison.images},
        originals={
            i.image: (images_dir / i.image).resolve().as_uri()
            for i in comparison.images
        },
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    path.write_text(html, encoding="utf-8")
    return path


def compare(reference_csv: Path, results_csv: Path) -> str:
    """A plain-text report of how a run's results agree with the human
    ratings, next to how the old models did on the same images."""
    try:
        return text_report(build_comparison(reference_csv, results_csv))
    except ValueError as error:
        return str(error)
