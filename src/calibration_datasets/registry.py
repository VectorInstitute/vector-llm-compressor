from typing import Type
from collections.abc import Callable

from .base import CalibrationDatasetBase

# A string key for all the calibration datasets
DATASETS_KEY: dict[str, Type[CalibrationDatasetBase]] = {}

# A string key providing descriptions of each calibration dataset
DATASETS_HELP_KEY: dict[str, str] = {}

def register_dataset(name: str | None = None) -> Callable:
    """Decorator factory for registering calibration datasets with the CLI.

    Args:
        name (str | None, optional): The name to register the calibration dataset under. If left as None then 
            `cls.__name__` is used.
    Returns:
        Callable: A decorator that registers a subclass of CalibrationDatasetBase
    """
    def decorator(cls: Type[CalibrationDatasetBase]) -> Type[CalibrationDatasetBase]:
        key = cls.__name__ if name is None else name

        DATASETS_KEY[key] = cls
        DATASETS_HELP_KEY[key] = cls.__help__()
        return cls
    return decorator