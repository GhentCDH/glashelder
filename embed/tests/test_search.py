"""Checks for chunked similarity search.

Run with `uv run python embed/tests/test_search.py`.
"""

import torch

from embed.search import find_similar, perform_search, place, topk


def data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    refs = torch.nn.functional.normalize(torch.randn(5_000, 32), dim=1)
    return refs, torch.randn(300, 32)


def full_topk(queries: torch.Tensor, refs: torch.Tensor, k: int):
    return torch.topk(torch.nn.functional.normalize(queries, dim=1) @ refs.T, k, dim=1)


def test_chunks_give_the_full_result() -> None:
    refs, queries = data()
    expected_values, expected_indices = full_topk(queries, refs, 10)
    # a chunk size that does not divide the refs, and one smaller than k
    for chunk in (777, 7):
        values, indices = topk(queries, refs, 10, chunk=chunk)
        assert torch.equal(indices, expected_indices)
        assert torch.allclose(values, expected_values, atol=1e-5)


def test_float16_refs() -> None:
    refs, queries = data()
    expected_values, expected_indices = full_topk(queries, refs, 5)
    values, indices = topk(queries, refs.half(), 5)
    assert values.dtype == torch.float32
    assert (indices[:, 0] == expected_indices[:, 0]).float().mean() > 0.99
    assert (values - expected_values).abs().max() < 1e-3


def test_single_query_helpers() -> None:
    refs, queries = data()
    indices, _ = perform_search(3, queries[0], refs, None)
    assert torch.equal(indices, full_topk(queries[:1], refs, 3)[1][0])
    indices, _ = find_similar(42, 4, refs, None)
    assert 42 not in indices.tolist() and len(indices) == 4


def test_place_keeps_a_cpu_matrix() -> None:
    refs, _ = data()
    assert place(refs, "cpu") is refs


if __name__ == "__main__":
    test_chunks_give_the_full_result()
    test_float16_refs()
    test_single_query_helpers()
    test_place_keeps_a_cpu_matrix()
    print("ok")
