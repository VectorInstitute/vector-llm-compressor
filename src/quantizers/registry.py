from typing import Type
from collections.abc import Callable

from .base import QuantizerBase

QUANTIZER_KEY: dict[str, Type[QuantizerBase]] = {}
QUANTIZER_HELP_KEY: dict[str, str] = {}

def register_recipe(name: str | None = None) -> Callable:
    """Decorator factory for registering quantizaters with the CLI.

    Args:
        name (str | None, optional): The name to register the quantizater under. If left as None then 
            `cls.__name__` is used.
    Returns:
        Callable: A decorator that registers a subclass of QuantizerBase
    """
    def decorator(cls: Type[QuantizerBase]) -> Type[QuantizerBase]:
        key = cls.__name__ if name is None else name

        QUANTIZER_KEY[key] = cls
        QUANTIZER_HELP_KEY[key] = cls.__help__()
        return cls
    return decorator