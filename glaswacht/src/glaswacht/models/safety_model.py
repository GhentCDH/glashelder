import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from PIL import Image

# The answers a model may give, as listed in models/taxonomy.txt
RATINGS = ("Safe", "Potentially Problematic", "Unsafe")
POLICY_CATEGORIES = (
    "O1: Hate, Humiliation, Harassment",
    "O2: Violence",
    "O3: Nudity Content",
    "O4: Ethnographic content",
)
NO_CATEGORY = "NA: None applying"
CATEGORIES = (*POLICY_CATEGORIES, NO_CATEGORY)

# how a list of categories is written in one CSV cell
CATEGORY_SEPARATOR = "; "

# JSON schema the models are constrained to. Properties are generated in this
# order, so the model reasons in the rationale before it classifies.
ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        # bounded so the JSON always closes within the models' max_new_tokens
        "rationale": {"type": "string", "maxLength": 2000},
        # every category that applies, or only NO_CATEGORY when none does
        "category": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"enum": list(POLICY_CATEGORIES)},
                    "minItems": 1,
                    "maxItems": len(POLICY_CATEGORIES),
                },
                {
                    "type": "array",
                    "items": {"const": NO_CATEGORY},
                    "minItems": 1,
                    "maxItems": 1,
                },
            ]
        },
        "rating": {"enum": list(RATINGS)},
    },
    "required": ["rationale", "category", "rating"],
    "additionalProperties": False,
}


def join_categories(category: str | list[str]) -> str:
    """The categories of an answer as one string, in taxonomy order and each
    once: the schema cannot rule out a repeated category."""
    if isinstance(category, str):
        return category
    known = [c for c in CATEGORIES if c in category]
    unknown = [c for c in dict.fromkeys(category) if c not in CATEGORIES]
    return CATEGORY_SEPARATOR.join(known + unknown) or NO_CATEGORY


@dataclass
class Rating:
    """Output of a risk assessment model"""

    rating: str
    # every category that applies, joined with CATEGORY_SEPARATOR
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
