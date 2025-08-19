from typing import Type
from collections.abc import Callable

from .base import ModelBase

MODELS_KEY: dict[str, Type[ModelBase]] = {}
MODELS_HELP_KEY: dict[str, str] = {}

def register_model(name: str | None = None) -> Callable:
    """Decorator factory for registering a pretrained model with the CLI.

    Args:
        name (str | None, optional): The name to register the pretrained model under. If left as None then 
            `cls.__name__` is used.
    Returns:
        Callable: A decorator that registers a subclass of ModelBase
    """
    def decorator(cls: Type[ModelBase]) -> Type[ModelBase]:
        key = cls.__name__ if name is None else name

        MODELS_KEY[key] = cls
        MODELS_HELP_KEY[key] = cls.__help__()
        return cls
    return decorator