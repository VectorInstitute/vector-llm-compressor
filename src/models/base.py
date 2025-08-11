from abc import abstractmethod
from typing import Any

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class ModelBase:
    @classmethod
    @abstractmethod
    def get_model(cls, **kwargs: Any) -> PreTrainedModel:
        """Get the pretrained model

        Returns:
            PreTrainedModel: A huggingface compatible pretrained model
            **kwargs (Any)
        """
        pass

    @classmethod
    @abstractmethod
    def get_tokenizer(cls, **kwargs: Any) -> PreTrainedTokenizerBase:
        pass
