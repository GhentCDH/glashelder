"""The ledger and how a run uses it, with fake models: no GPU needed.

Run with `uv run pytest glaswacht/tests`.
"""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from glaswacht.ledger import MAX_ATTEMPTS, Ledger
from glaswacht.models.safety_model import Rating, SafetyModel
from glaswacht.orchestrator import LEDGER_NAME, Orchestrator

# every image each fake model was asked to assess, over all runs of a test
CALLS: list[tuple[str, str]] = []


class _Fake(SafetyModel):
    name = "fake"
    rating = "Safe"
    loads = 0

    def load_model(self) -> None:
        type(self).loads += 1

    def unload_model(self) -> None:
        pass

    def process_image(self, image, image_name: str = "") -> Rating:
        CALLS.append((self.name, image_name))
        return Rating(
            self.rating, "NA: None applying", f"{self.name} saw it", image_name
        )


class Alpha(_Fake):
    name = "alpha"


class Beta(_Fake):
    name = "beta"
    rating = "Unsafe"


class Gamma(_Fake):
    name = "gamma"
    rating = "Unsafe"


class Broken(_Fake):
    """Fails on every image whose name starts with "bad"."""

    name = "broken"

    def process_image(self, image, image_name: str = "") -> Rating:
        if image_name.startswith("bad"):
            CALLS.append((self.name, image_name))
            raise RuntimeError("CUDA out of memory\nmore detail")
        return super().process_image(image, image_name)


MODELS = [Alpha, Beta, Gamma]


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()
    yield


@pytest.fixture
def images(tmp_path: Path) -> Path:
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ["a.jpg", "b.jpg", "c.png", "d.jpg", "e.jpg"]:
        Image.new("RGB", (8, 8)).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    return folder


def _run(images: Path, out: Path, models=MODELS, **kwargs) -> Orchestrator:
    orchestrator = Orchestrator(models)
    orchestrator.run_all(str(images), out, **kwargs)
    return orchestrator


def _ledger_lines(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / LEDGER_NAME).read_text().splitlines()]


def _csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("load_all", [False, True])
def test_max_images_continues_where_the_last_run_stopped(images, tmp_path, load_all):
    out = tmp_path / "out"

    first = _run(images, out, max_images=2, load_all=load_all)
    assert first.summary.selected == first.summary.finished == 2
    assert first.summary.remaining == 3
    assert sorted({image for _, image in CALLS}) == ["a.jpg", "b.jpg"]
    assert [row["image"] for row in _csv(first.results_csv)] == ["a.jpg", "b.jpg"]

    CALLS.clear()
    second = _run(images, out, max_images=2, load_all=load_all)
    assert sorted({image for _, image in CALLS}) == ["c.png", "d.jpg"]
    assert second.summary.remaining == 1

    CALLS.clear()
    third = _run(images, out, max_images=2, load_all=load_all)
    assert {image for _, image in CALLS} == {"e.jpg"}
    assert third.summary.remaining == 0

    CALLS.clear()
    fourth = _run(images, out, max_images=2, load_all=load_all)
    assert CALLS == []
    assert fourth.results_csv is None
    assert not fourth.aborted

    # every model assessed every image exactly once, over all four runs
    lines = _ledger_lines(out)
    assert len(lines) == 5 * len(MODELS)
    assert len({(line["image"], line["model"]) for line in lines}) == len(lines)
    assert all(line["status"] == "done" for line in lines)


def test_load_all_goes_image_by_image(images, tmp_path):
    out = tmp_path / "out"
    _run(images, out, max_images=2, load_all=True)
    assert CALLS == [
        ("alpha", "a.jpg"),
        ("beta", "a.jpg"),
        ("gamma", "a.jpg"),
        ("alpha", "b.jpg"),
        ("beta", "b.jpg"),
        ("gamma", "b.jpg"),
    ]


def test_one_at_a_time_and_load_all_write_the_same(images, tmp_path):
    a = _run(images, tmp_path / "a", load_all=False)
    b = _run(images, tmp_path / "b", load_all=True)
    assert _csv(a.results_csv) == _csv(b.results_csv)
    assert a.get_majority_votes() == b.get_majority_votes()


def test_a_partly_assessed_image_only_goes_to_the_missing_models(images, tmp_path):
    out = tmp_path / "out"
    _run(images, out, models=[Alpha], max_images=1)
    assert CALLS == [("alpha", "a.jpg")]

    CALLS.clear()
    orchestrator = _run(images, out, max_images=1)
    # a.jpg still needed beta and gamma, so it is this run's one image
    assert CALLS == [("beta", "a.jpg"), ("gamma", "a.jpg")]
    (row,) = _csv(orchestrator.results_csv)
    assert row["alpha_rating"] == "Safe"
    assert row["beta_rating"] == row["gamma_rating"] == "Unsafe"
    assert row["majority_vote"] == "Unsafe"
    assert orchestrator.get_majority_votes() == {"a.jpg": "Unsafe"}


def test_failures_are_retried_then_given_up(images, tmp_path):
    (images / "bad.jpg").write_bytes((images / "a.jpg").read_bytes())
    out = tmp_path / "out"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        CALLS.clear()
        orchestrator = _run(images, out, models=[Alpha, Broken])
        assert ("broken", "bad.jpg") in CALLS
        assert [s.image for s in orchestrator.get_skips()] == ["bad.jpg"]
        assert orchestrator.get_skips()[0].error == "RuntimeError: CUDA out of memory"
        if attempt == 1:
            # the other images went through on the first run
            assert orchestrator.summary.remaining == 1
        # alpha assessed bad.jpg on the first run only
        assert (("alpha", "bad.jpg") in CALLS) == (attempt == 1)

    CALLS.clear()
    orchestrator = _run(images, out, models=[Alpha, Broken])
    assert CALLS == []
    assert orchestrator.summary.given_up == ["bad.jpg"]
    assert orchestrator.summary.remaining == 0

    failed = [line for line in _ledger_lines(out) if line["status"] == "failed"]
    assert len(failed) == MAX_ATTEMPTS
    assert {line["image"] for line in failed} == {"bad.jpg"}


def test_an_unreadable_image_is_a_failure_not_a_crash(images, tmp_path):
    (images / "zz_corrupt.jpg").write_bytes(b"not a jpeg")
    out = tmp_path / "out"
    orchestrator = _run(images, out, load_all=True)
    assert {s.model for s in orchestrator.get_skips()} == {"alpha", "beta", "gamma"}
    assert {s.image for s in orchestrator.get_skips()} == {"zz_corrupt.jpg"}
    assert len(_csv(orchestrator.results_csv)) == 5


def test_a_line_cut_off_mid_write_is_skipped(tmp_path):
    path = tmp_path / LEDGER_NAME
    ledger = Ledger(path)
    ledger.record_done("alpha", Rating("Safe", "NA: None applying", "fine", "a.jpg"))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"image": "b.jpg", "model": "alp')  # killed here

    ledger = Ledger(path)
    assert ledger.rating("a.jpg", "alpha").rating == "Safe"
    assert ledger.rating("b.jpg", "alpha") is None
    # the next line starts on its own line and is read back
    ledger.record_done("alpha", Rating("Unsafe", "O2: Violence", "x", "b.jpg"))
    assert Ledger(path).rating("b.jpg", "alpha").rating == "Unsafe"


def test_a_stop_request_finishes_the_image_and_writes_the_csv(images, tmp_path):
    out = tmp_path / "out"

    class Stopper(Alpha):
        def process_image(self, image, image_name=""):
            rating = super().process_image(image, image_name)
            orchestrator.request_stop()
            return rating

    orchestrator = Orchestrator([Stopper, Beta])
    orchestrator.run_all(str(images), out, load_all=True)
    # the first image is finished by both models, then the run stops
    assert CALLS == [("alpha", "a.jpg"), ("beta", "a.jpg")]
    assert orchestrator.summary.stopped
    assert orchestrator.summary.selected == 5
    assert orchestrator.summary.finished == 1
    assert orchestrator.summary.remaining == 4
    assert [row["image"] for row in _csv(orchestrator.results_csv)] == ["a.jpg"]


def test_recursive_keys_images_by_their_relative_path(images, tmp_path):
    sub = images / "box1"
    sub.mkdir()
    Image.new("RGB", (8, 8)).save(sub / "a.jpg")
    out = tmp_path / "out"

    flat = _run(images, out, models=[Alpha])
    assert "box1/a.jpg" not in {image for _, image in CALLS}
    assert flat.summary.total == 5

    CALLS.clear()
    _run(images, out, models=[Alpha], recursive=True)
    # a.jpg at the top was done already; the one in box1 is another image
    assert CALLS == [("alpha", "box1/a.jpg")]


@pytest.mark.parametrize("full_precision", [False, True])
def test_the_precision_reaches_the_models_and_the_ledger(
    images, tmp_path, full_precision
):
    class Quantizable(Alpha):
        MODEL_ID = "org/quantizable"

        @property
        def precision(self) -> str:
            return "bf16" if self.full_precision else "nf4"

    out = tmp_path / "out"
    Orchestrator([Quantizable], full_precision=full_precision).run_all(
        str(images), out, max_images=1
    )
    (line,) = _ledger_lines(out)
    assert line["model_id"] == "org/quantizable"
    assert line["precision"] == ("bf16" if full_precision else "nf4")
