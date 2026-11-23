import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from PIL import Image

# The answers a model may give, as listed in models/taxonomy.txt
RATINGS = ("Safe", "Potentially Problematic", "Unsafe")
CATEGORIES = (
    "O1: Hate, Humiliation, Harassment",
    "O2: Violence",
    "O3: Nudity Content",
    "O4: Ethnographic content",
    "NA: None applying",
)

# JSON schema the models are constrained to. Properties are generated in this
# order, so the model reasons in the rationale before it classifies.
ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        # bounded so the JSON always closes within the models' max_new_tokens
        "rationale": {"type": "string", "maxLength": 1000},
        "category": {"enum": list(CATEGORIES)},
        "rating": {"enum": list(RATINGS)},
    },
    "required": ["rationale", "category", "rating"],
    "additionalProperties": False,
}


@dataclass
class Rating:
    """Output of a risk assessment model"""

    rating: str
    category: str
    rationale: str
    image: str = field(default="")


class SafetyModel(ABC):
    name: ClassVar[str]
    # False leaves the model out unless it is asked for with `--model <name>`
    enabled_by_default: ClassVar[bool] = True

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
