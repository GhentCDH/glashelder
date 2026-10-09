def get_device() -> str:
    """The best available device for tensor operations."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
