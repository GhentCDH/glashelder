"""Cluster package - Image clustering and matching using generated embeddings."""

__all__ = ["perform_search"]


def __getattr__(name: str):
    # on first use only: the search pulls in torch, the CLI does not need it
    if name == "perform_search":
        from embed.search import perform_search

        return perform_search
    raise AttributeError(f"module 'cluster' has no attribute {name!r}")
