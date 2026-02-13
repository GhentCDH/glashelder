import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# for defining paths relative to project root
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent

assets_env_var = os.getenv("ASSETS_DIR")
ASSETS_DIR: Path | None = Path(assets_env_var) if assets_env_var is not None else None

__all__ = ["ASSETS_DIR", "PROJECT_ROOT"]
