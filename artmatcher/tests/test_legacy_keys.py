"""Files written before `wikiart_*` became `artwork_*` still read."""

import json

from artmatcher.batch import read_json_output
from artmatcher.web.store import artwork_keys, normalize_dataset


def test_renames_old_keys() -> None:
    old = {"wikiart_path": "a.jpg", "wikiart_index": 3, "rank": 0}
    assert artwork_keys(old) == {"artwork_path": "a.jpg", "artwork_index": 3, "rank": 0}


def test_current_keys_win() -> None:
    both = {"artwork_path": "new.jpg", "wikiart_path": "old.jpg"}
    assert artwork_keys(both) == {"artwork_path": "new.jpg"}


def test_old_labels() -> None:
    dataset = normalize_dataset(
        {"positives": [{"wikiart_path": "a.jpg", "wikiart_idx": 3}]}
    )
    assert dataset.positives == [
        {"artwork_path": "a.jpg", "artwork_idx": 3, "is_match": True}
    ]


def test_old_batch_output(tmp_path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {"metadata": {}, "results": [{"wikiart_path": "a.jpg", "wikiart_index": 3}]}
        )
    )
    assert read_json_output(path)["results"] == [
        {"artwork_path": "a.jpg", "artwork_index": 3}
    ]
