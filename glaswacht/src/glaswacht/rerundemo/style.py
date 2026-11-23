"""Colours and markdown helpers shared by every layer of the demo."""

import rerun as rr

RATING_COLOURS = {
    "Safe": (64, 168, 92),
    "Potentially Problematic": (240, 170, 40),
    "Unsafe": (224, 62, 54),
    "Unknown": (140, 140, 148),
}
MODEL_COLOURS = {
    "lavaguard": (66, 133, 244),
    "qwen3": (255, 138, 61),
    "pixtral": (160, 108, 213),
}
MODEL_TITLES = {
    "lavaguard": "LlavaGuard 7B",
    "qwen3": "Qwen3-VL 8B",
    "pixtral": "Pixtral 12B",
}
MODEL_FACTS = {
    "lavaguard": (
        "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf",
        "4-bit nf4",
        "sampled (T=0.2, 2 beams)",
    ),
    "qwen3": ("Qwen/Qwen3-VL-8B-Instruct", "4-bit nf4", "greedy"),
    "pixtral": ("mistral-community/pixtral-12b", "8-bit int8", "greedy"),
}
MARKDOWN = rr.MediaType.MARKDOWN


def title(model: str) -> str:
    return MODEL_TITLES.get(model, model)


def colour(model: str) -> tuple[int, int, int]:
    return MODEL_COLOURS.get(model, (120, 120, 120))


def md(text: str) -> rr.TextDocument:
    return rr.TextDocument(text, media_type=MARKDOWN)
