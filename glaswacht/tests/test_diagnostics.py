"""Pure-function tests for glaswacht.models.diagnostics.

No pytest in this workspace, so: `uv run python glaswacht/tests/test_diagnostics.py`.
"""

import json

from glaswacht.models.diagnostics import (
    RATING_CANDIDATES,
    SCHEMA_VERSION,
    DiagnosticsTrace,
    RatingDistribution,
    TokenAlternative,
    TokenStep,
    rating_distribution,
    token_pieces,
)


class FakeTokenizer:
    """Byte-level-ish stand-in: ids are indices into a token table."""

    def __init__(self, table: dict[int, str]):
        self.table = table

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [i for i, t in self.table.items() if text.startswith(t)]
        if not ids:
            raise KeyError(text)
        return [max(ids, key=lambda i: len(self.table[i]))]

    def decode(self, ids) -> str:
        out = b"".join(self.table[i].encode("utf-8") for i in ids)
        return out.decode("utf-8", errors="replace")


class ByteTokenizer:
    """Every id is one raw byte, so multi-byte characters span tokens."""

    def decode(self, ids) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def test_token_pieces_joins_to_decode():
    tok = FakeTokenizer({1: "Safe", 2: '",', 3: "\n    ", 4: '"category"'})
    ids = [1, 2, 3, 4]
    pieces = token_pieces(tok, ids)
    assert "".join(pieces) == tok.decode(ids)
    assert pieces == ["Safe", '",', "\n    ", '"category"']


def test_token_pieces_holds_back_multibyte():
    tok = ByteTokenizer()
    ids = list("héllo 🎨".encode())
    pieces = token_pieces(tok, ids)
    assert "".join(pieces) == tok.decode(ids)
    assert "�" not in "".join(pieces)
    # the continuation bytes render as empty, the final byte carries the char
    assert len(pieces) == len(ids)


def test_rating_distribution_raw_normalized_residual():
    tok = FakeTokenizer({10: "Safe", 20: "Unsafe", 30: "Potentially Problematic"})
    probs = {10: 0.8, 20: 0.02, 30: 0.12}
    dist = rating_distribution(tok, probs)
    assert dist.exact is True
    assert dist.candidate_first_tokens == {
        "Safe": 10,
        "Unsafe": 20,
        "Potentially Problematic": 30,
    }
    assert dist.raw == {"Safe": 0.8, "Unsafe": 0.02, "Potentially Problematic": 0.12}
    assert abs(dist.residual - 0.06) < 1e-9
    assert abs(sum(dist.normalized.values()) - 1.0) < 1e-9
    assert abs(dist.normalized["Safe"] - 0.8 / 0.94) < 1e-9


def test_rating_distribution_collision_falls_back_to_inexact():
    # "Safe" and "Unsafe" collapsing onto one id would make the proxy ambiguous
    tok = FakeTokenizer({10: "Safe", 30: "Potentially Problematic"})
    tok.encode = lambda text, add_special_tokens=False: [
        30 if text.startswith("Potentially") else 10
    ]
    dist = rating_distribution(tok, {10: 0.5, 30: 0.1})
    assert dist.exact is False
    assert dist.candidate_first_tokens["Safe"] == dist.candidate_first_tokens["Unsafe"]


def test_rating_distribution_zero_mass_does_not_divide_by_zero():
    tok = FakeTokenizer({10: "Safe", 20: "Unsafe", 30: "Potentially Problematic"})
    dist = rating_distribution(tok, {10: 0.0, 20: 0.0, 30: 0.0})
    assert dist.normalized == dict.fromkeys(RATING_CANDIDATES, 0.0)
    assert dist.residual == 1.0


def test_trace_to_dict_is_json_serializable_and_complete():
    trace = DiagnosticsTrace(
        image={"path": "/abs/x.jpg", "name": "x.jpg"},
        model={
            "name": "qwen3",
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "parse_model_type": "qwen",
            "quantization": "nf4-4bit",
        },
        decoding={
            "diagnostic": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
            "production": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
        },
        top_k=1,
        prefill='{\n    "rating": "',
        prompt_token_count=1234,
        generated_text='{\n    "rating": "Safe"}',
        parsed={"rating": "Safe"},
        rating_distribution=RatingDistribution(
            raw={"Safe": 0.8123456789},
            normalized={"Safe": 1.0},
            residual=0.1876543211,
            exact=True,
            candidate_first_tokens={"Safe": 25663},
        ),
        steps=[
            TokenStep(
                i=0,
                token_id=25663,
                token="Safe",
                raw_token="Safe",
                prob=0.8123456789,
                logprob=-0.2078,
                alternatives=[
                    TokenAlternative(25663, "Safe", "Safe", 0.8123456789),
                ],
            )
        ],
        timings_s={"load": 41.23456, "generate": 6.8},
        generated_at="2026-09-23T14:22:10Z",
    )
    doc = trace.to_dict()
    assert doc["schema_version"] == SCHEMA_VERSION == 1
    assert sorted(doc) == sorted(
        [
            "schema_version",
            "generated_at",
            "image",
            "model",
            "decoding",
            "top_k",
            "prefill",
            "prompt_token_count",
            "generated_text",
            "parsed",
            "rating_distribution",
            "steps",
            "counterfactuals",
            "timings_s",
        ]
    )
    assert doc["steps"][0]["prob"] == 0.812346
    assert doc["timings_s"]["load"] == 41.235
    assert doc["counterfactuals"] == []
    json.dumps(doc, ensure_ascii=False)


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all diagnostics tests passed")


if __name__ == "__main__":
    main()
