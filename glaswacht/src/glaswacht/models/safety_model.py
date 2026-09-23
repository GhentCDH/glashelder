import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from PIL import Image


@dataclass
class Rating:
    """Output of a risk assessment model"""

    rating: str
    category: str
    rationale: str
    image: str = field(default="")


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
