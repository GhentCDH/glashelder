import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from PIL import Image


@dataclass
class Rating:
    """Output of a risk assessment model"""

    rating: str
    category: str
    rationale: str
    image: str = field(default="")


@dataclass
class TokenStep:
    """One generated token and the distribution it was drawn from."""

    index: int
    token: str
    token_id: int
    prob: float
    entropy: float
    top: list[tuple[str, float]]


@dataclass
class SeenImage:
    """What a model's image processor actually feeds the network."""

    name: str
    pixels: np.ndarray  # H x W x 3 uint8
    patch_size: int | None = None


@dataclass
class Trace:
    """Everything `inspect()` can see of one forward/generate pass."""

    model: str
    image: str
    prefill: str
    steps: list[TokenStep]
    text: str
    rating: Rating
    label_probs: dict[str, float]
    seen: list[SeenImage]
    prompt_tokens: int
    image_tokens: int
    latency_s: float
    vram_mb: float


class SafetyModel(ABC):
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and not isinstance(
            getattr(cls, "name", None), str
        ):
            raise TypeError(f"{cls.__name__} must define class attribute 'name: str'")

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @abstractmethod
    def process_image(self, image: Image.Image, image_name: str = "") -> Rating:
        pass
