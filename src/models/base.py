from abc import ABC, abstractmethod
from typing import Any

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class ModelBase(ABC):
    """Base class for loading pretrained models."""

    @abstractmethod
    def get_model(self, **kwargs: Any) -> PreTrainedModel:
        """Get the pretrained model

        Returns:
            PreTrainedModel: A huggingface compatible pretrained model
            **kwargs (Any)
        """
        pass

    @abstractmethod
    def get_tokenizer(self, **kwargs: Any) -> PreTrainedTokenizerBase:
        """Get the tokenizer for the pretrained model

        Returns:
            PreTrainedTokenizerBase: A huggingface compatible tokenizer
        """
        pass
    
    @classmethod
    @abstractmethod
    def __help__(cls) -> str:
        """
        Returns:
            str: A brief description of the pretrained model, used by CLI.
        """
