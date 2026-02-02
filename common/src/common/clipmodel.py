MODEL = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"


def get_model(model: str | None = None):
    if model is None:
        model = MODEL

    from transformers import CLIPModel

    return CLIPModel.from_pretrained(MODEL)


def get_processor(model: str | None = None):
    if model is None:
        model = MODEL

    from transformers import CLIPProcessor

    return CLIPProcessor.from_pretrained(MODEL)


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
