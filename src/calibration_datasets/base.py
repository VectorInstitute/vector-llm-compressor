from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class CalibrationDatasetBase(ABC):
    """Base class for loading calibration datasets."""

    @abstractmethod
    def get_dataset(self, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> Dataset:
        """Class method that returns a huggingface datasets.Dataset

        Args:
            tokenizer (PreTrainedTokenizerBase): The model tokenizer to use on the dataset
            **kwargs (Any): To allow subclasses to access additional keyword arguments if needed.

        Returns:
            Dataset | IterableDataset : A huggingface compatible dataset.
        """
        pass
    
    @classmethod
    @abstractmethod
    def __help__(cls) -> str:
        """
        Returns:
            str: A brief help string that describes dataset, used by CLI
        """