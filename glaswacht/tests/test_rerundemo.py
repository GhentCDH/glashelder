import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glaswacht.models.safety_model import Rating, SeenImage, TokenStep, Trace
from glaswacht.rerundemo import deepdive, generation
from glaswacht.rerundemo.data import CLASSES, compute_stats, load_results, normalise
from glaswacht.rerundemo.demo import build_demo
from glaswacht.rerundemo.replay import replay

MODELS = ("lavaguard", "qwen3", "pixtral")


def write_results(
    path: Path, rows: list[tuple[str, str, str, str]], schema: str = "numbered"
):
    if schema == "numbered":
        cols = [f"model_{i}_{f}" for i in (1, 2, 3) for f in ("rating", "rationale")]
    else:
        cols = [f"{m}_{f}" for m in MODELS for f in ("rating", "rationale")]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", *cols])
        for name, a, b, c in rows:
            w.writerow([name, a, f"why {a}", b, f"why {b}", c, f"why {c}"])


ROWS = [
    ("a.jpg", "Safe", "Safe", "Safe"),
    ("b.jpg", "Unsafe", "Safe", "Safe"),  # outvoted
    ("c.jpg", "Unsafe", "Potentially Problematic", "Safe"),  # three-way tie
    ("d.jpg", "Unknown", "Safe", "Safe"),
    ("e.jpg", "Error", "Safe", "Safe"),  # not a known class
]


@pytest.mark.parametrize("schema", ["numbered", "named"])
def test_load_results_both_schemas(tmp_path, schema):
    write_results(tmp_path / "r.csv", ROWS, schema)
    rows = load_results(tmp_path / "r.csv")
    assert [r.image for r in rows] == [r[0] for r in ROWS]
    assert rows[1].majority == "Safe"  # one Unsafe vote is outvoted
    assert rows[2].majority == "Unsafe"  # ties go to the most cautious
    assert rows[4].ratings["lavaguard"] == "Unknown"  # "Error" is displayed as Unknown


def test_normalise():
    assert normalise("Safe") == "Safe"
    assert normalise("garbage") == "Unknown"
    assert normalise("") == "Unknown"


def test_stats(tmp_path):
    write_results(tmp_path / "r.csv", ROWS)
    stats = compute_stats(load_results(tmp_path / "r.csv"))
    assert stats.n == 5
    assert stats.distribution["lavaguard"].tolist() == [1, 0, 2, 2]
    for mat in stats.confusion.values():
        assert mat.sum() == 5
        assert mat.shape == (len(CLASSES), len(CLASSES))
    assert stats.all_agree[-1] == pytest.approx(1 / 5)
    # the three-way split is the most contested, the unanimous plate is not queued
    assert stats.review_order[0] == 2
    assert 0 not in stats.review_order
    assert stats.unsafe_rate["lavaguard"][-1] == pytest.approx(2 / 5)


def make_plate(path: Path, size=(200, 120)):
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)
    ).save(path)


def test_replay_writes_valid_recording(tmp_path):
    write_results(tmp_path / "r.csv", ROWS)
    for name, *_ in ROWS[:4]:  # e.jpg is missing on purpose
        make_plate(tmp_path / name)
    out = tmp_path / "out.rrd"
    replay(tmp_path / "r.csv", tmp_path, out=out, workers=2)
    assert out.stat().st_size > 0
    assert out.with_suffix(".rbl").exists()


def fake_trace(model: str, rating: str, n_steps: int) -> Trace:
    steps = [
        TokenStep(i, f"t{i} ", i, 0.9 - i * 0.01, 0.3, [(f"t{i} ", 0.9), ("x", 0.05)])
        for i in range(n_steps)
    ]
    return Trace(
        model=model,
        image="a.jpg",
        prefill='{\n    "rating": "',
        steps=steps,
        text="".join(s.token for s in steps),
        rating=Rating(rating, "NA: None applying", "because", "a.jpg"),
        label_probs={"Safe": 0.8, "Unsafe": 0.1, "Potentially Problematic": 0.05},
        seen=[SeenImage("resized", np.zeros((32, 48, 3), np.uint8), 16)],
        prompt_tokens=500,
        image_tokens=300,
        latency_s=1.5,
        vram_mb=9000.0,
    )


def test_traces_roundtrip_and_log(tmp_path):
    for stem in ("p1", "p2"):
        d = tmp_path / stem
        d.mkdir()
        make_plate(d / "plate.jpg")
        for k, m in enumerate(MODELS):
            deepdive.save_trace(d, fake_trace(m, "Safe", 3 + k))
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "models": list(MODELS),
                "plates": ["p1", "p2"],
                "load_s": dict.fromkeys(MODELS, 4.0),
            }
        )
    )

    run = deepdive.load_traces(tmp_path)
    assert [p.name for p in run.plates] == ["p1", "p2"]
    t = run.plates[0].traces["pixtral"]
    assert len(t.steps) == 5 and t.steps[0].top[0] == ("t0 ", 0.9)
    assert t.seen[0].pixels.shape == (32, 48, 3)

    starts = generation.log_generation(run)  # no recording active: logs are dropped
    assert starts[0] == 0 and starts[1] > len(run.plates[0].traces["pixtral"].steps)

    build_demo(tmp_path, out=tmp_path / "demo.rrd")
    assert (tmp_path / "demo.rrd").stat().st_size > 0
    assert (tmp_path / "demo.rbl").exists()


def test_mosaic_and_grid():
    tiles = [SeenImage(f"t{i}", np.full((10, 20, 3), i, np.uint8)) for i in range(5)]
    pic = deepdive.mosaic(tiles)
    assert pic.shape == (20, 60, 3)  # 5 tiles -> 3 columns x 2 rows
    assert len(deepdive.patch_grid((32, 48), 16)) == 4 + 3


def test_mosaic_anyres_grid():
    """Overview on the left, 6 crops arranged to match a wide (3:2) original."""
    base = SeenImage("overview", np.zeros((10, 10, 3), np.uint8))
    crops = [SeenImage(f"c{i}", np.full((10, 10, 3), 9, np.uint8)) for i in range(6)]
    pic = deepdive.mosaic([base, *crops], aspect=1.5)
    # 2 rows x 3 cols of 10px crops with 6px gaps: 26 high, 42 wide, plus overview + 40px gap
    assert pic.shape[0] == 26
    assert pic.shape[1] == 26 + 40 + 42


def test_layout_keeps_words_together():
    steps = [
        TokenStep(i, tok, 0, 1.0, 0.0, [])
        for i, tok in enumerate(["ab", "cd", " ef", "gh", "\n", "ij"])
    ]
    t = fake_trace("qwen3", "Safe", 0)
    t.prefill, t.steps = "", steps
    cw = generation.char_w()
    segs = generation.layout(t, width=int(cw * 6))
    pos = {s.text: (round(s.x / cw), round(s.y / generation.LINE_H)) for s in segs}
    assert pos["ab"] == (0, 0) and pos["cd"] == (2, 0)  # sub-word tokens abut
    assert pos[" ef"][1] == 1 and pos["gh"][1] == 1  # the word that did not fit moved
    assert pos[" ef"][0] == -1  # its leading space hangs off the left edge
    assert pos["ij"] == (0, 2)  # newline token breaks the line


def test_prob_colour():
    sure, torn = generation.prob_colour(1.0), generation.prob_colour(0.0)
    assert sure[1] > sure[0] and torn[0] > torn[1]  # green when sure, red when torn
    assert generation.prob_colour(0.5) != generation.prob_colour(0.1)
    assert generation.text_on((250, 250, 250)) != generation.text_on((10, 10, 10))
