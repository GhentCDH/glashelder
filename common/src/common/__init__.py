from pathlib import Path

from dotenv import load_dotenv

from .auth import require_auth
from .serve import run_app

load_dotenv()

# for defining paths relative to project root
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent

__all__ = ["PROJECT_ROOT", "require_auth", "run_app"]
