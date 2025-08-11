from typing import Any

from base import ModelBase
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class AutoHFModel(ModelBase):
    """Uses the huggingface Auto Classes to retreive the model and tokenizer from huggingface.co based on the model_id"""
    @classmethod
    def get_model(cls, *, model_id: str, **kwargs: Any) -> PreTrainedModel:
        """Uses the huggingface AutoModelForCausalLM to retreive model.

        Args:
            model_id (str): The 

        Returns:
            PreTrainedModel: _description_
        """
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    
    @classmethod
    def get_tokenizer(cls, *, model_id: str, **kwargs: Any) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(model_id)