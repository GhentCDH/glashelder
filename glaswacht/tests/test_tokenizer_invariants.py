"""Tokenizer-level invariants the diagnostics rating distribution rests on.

These are what silently break if someone edits `JSON_PREFILL`, swaps a
checkpoint, or the pinned transformers changes the Mistral pre-tokenizer.
Needs the checkpoints in `~/.cache/huggingface`, but no GPU.

No pytest in this workspace, so:
`HF_HUB_OFFLINE=1 uv run python glaswacht/tests/test_tokenizer_invariants.py`.
"""

import os
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoProcessor, GenerationConfig

from glaswacht.models.diagnostics import RATING_CANDIDATES, candidate_first_tokens
from glaswacht.models.model_LLavaguard import LavaGuard
from glaswacht.models.model_Pixtral import Pixtral, _fix_mistral_pretokenizer
from glaswacht.models.model_Qwen3 import Qwen3
from glaswacht.models.utils import get_prompt

MODELS = (LavaGuard, Pixtral, Qwen3)

# Pinned so a checkpoint or prefill change has to be a deliberate edit here.
EXPECTED_FIRST_TOKENS = {
    "lavaguard": {"Safe": 25663, "Unsafe": 78770, "Potentially Problematic": 44616},
    "qwen3": {"Safe": 25663, "Unsafe": 78770, "Potentially Problematic": 44616},
    "pixtral": {"Safe": 53907, "Unsafe": 3877, "Potentially Problematic": 63026},
}

PREFILL = LavaGuard.JSON_PREFILL


def _tokenizer(model_class):
    tokenizer = AutoProcessor.from_pretrained(model_class.MODEL_ID).tokenizer
    if model_class is Pixtral:
        _fix_mistral_pretokenizer(tokenizer)
    return tokenizer


def test_prefill_ends_on_a_quote_token():
    for model_class in MODELS:
        tokenizer = _tokenizer(model_class)
        ids = tokenizer.encode(PREFILL, add_special_tokens=False)
        last = tokenizer.convert_ids_to_tokens(ids)[-1]
        assert last == 'Ġ"', (model_class.name, last)


def test_rating_candidate_first_tokens_are_distinct_and_pinned():
    for model_class in MODELS:
        tokenizer = _tokenizer(model_class)
        first = candidate_first_tokens(tokenizer)
        assert first == EXPECTED_FIRST_TOKENS[model_class.name], (
            model_class.name,
            first,
        )
        assert len(set(first.values())) == len(RATING_CANDIDATES), model_class.name


def test_prefill_tokenization_is_boundary_stable():
    """Appending a rating must not retokenize the prefill, otherwise the
    bare first-token id is not the one the model actually emits at step 0."""
    for model_class in MODELS:
        tokenizer = _tokenizer(model_class)
        prefill_ids = tokenizer.encode(PREFILL, add_special_tokens=False)
        first = candidate_first_tokens(tokenizer)
        for candidate in RATING_CANDIDATES:
            joint = tokenizer.encode(PREFILL + candidate, add_special_tokens=False)
            assert joint[: len(prefill_ids)] == prefill_ids, (
                model_class.name,
                candidate,
            )
            assert joint[len(prefill_ids)] == first[candidate], (
                model_class.name,
                candidate,
            )


def test_pixtral_pretokenizer_fix_is_installed_and_behavior_neutral():
    from tokenizers import Regex, pre_tokenizers

    from glaswacht.models.model_Pixtral import _MISTRAL_SPLIT_PATTERN

    tokenizer = AutoProcessor.from_pretrained(Pixtral.MODEL_ID).tokenizer
    taxonomy = get_prompt()
    prompt = f"<s>[INST]{taxonomy}\n[IMG][/INST]{PREFILL}"
    probes = {
        "taxonomy": taxonomy,
        "prompt": prompt,
        "prefill": PREFILL,
        **{c: c for c in RATING_CANDIDATES},
    }
    before = {
        k: tokenizer.encode(v, add_special_tokens=False) for k, v in probes.items()
    }
    shipped = tokenizer.backend_tokenizer.pre_tokenizer

    # the shipped regex really is the broken one: it caps digit runs at three
    assert [p for p, _ in shipped.pre_tokenize_str("1234")] == ["123", "4"]

    _fix_mistral_pretokenizer(tokenizer)

    corrected = tokenizer.backend_tokenizer.pre_tokenizer
    assert isinstance(corrected, pre_tokenizers.Sequence)
    expected = pre_tokenizers.Split(
        pattern=Regex(_MISTRAL_SPLIT_PATTERN), behavior="isolated"
    )
    assert str(corrected[0]) == str(expected)
    assert [p for p, _ in corrected.pre_tokenize_str("1234")] == ["1", "2", "3", "4"]

    after = {
        k: tokenizer.encode(v, add_special_tokens=False) for k, v in probes.items()
    }
    assert after == before, [k for k in before if before[k] != after[k]]


def test_production_decode_is_greedy_and_warning_free():
    for model_class in MODELS:
        kwargs = model_class.GENERATE_KWARGS
        assert kwargs["do_sample"] is False, model_class.name
        assert kwargs["num_beams"] == 1, model_class.name
        for key in ("temperature", "top_p", "top_k"):
            assert kwargs[key] is None, (model_class.name, key)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = GenerationConfig.from_pretrained(model_class.MODEL_ID)
            config.update(**kwargs)
            config.validate()
        assert not caught, (model_class.name, [str(w.message) for w in caught])
        assert config.do_sample is False, model_class.name


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tokenizer invariant tests passed")


if __name__ == "__main__":
    main()
