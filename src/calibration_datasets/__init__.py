# Import all the modules so that registry is populated. Must import registry last
from . import auto
# ruff: noqa : I001 # Ignore isort for registry since we need to import it last
from .registry import DATASETS_KEY, DATASETS_HELP_KEY

__all__ = [
    "DATASETS_KEY",
    "DATASETS_HELP_KEY",
    "auto",
]