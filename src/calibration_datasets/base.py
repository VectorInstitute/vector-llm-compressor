from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset, IterableDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class CalibrationDatasetBase(ABC):
    @classmethod
    @abstractmethod
    def get_dataset(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        num_samples: int = 512,
        max_sequence_length: int = 2048,
        **kwargs: Any
    ) -> Dataset | IterableDataset:
        """Class method that returns a huggingface datasets.Dataset

        Args:
            tokenizer (PreTrainedTokenizerBase): The model tokenizer to use on the dataset
            num_samples (int, optional): Number of samples to use from the dataset. Defaults to 512.
            max_sequence_length (int, optional): The max sequence length. Defaults to 2048.
            **kwargs (Any): To allow the user to add new required keyword arguments if needed.

        Returns:
            Dataset | IterableDataset : A huggingface compatible dataset.
        """
        pass
