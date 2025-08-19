from typing import Any

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import ModelBase
from .registry import register_model


@register_model("auto")
class AutoHFModel(ModelBase):
    """Uses the huggingface Auto Classes to retreive the model and tokenizer from huggingface.co based on the model_id"""
    def __init__(self, model_id: str) -> None:
        """Initialize AutoHFModel with a specific model id

        Args:
            model_id (str): Can be either the model id of a pretrained model hosted inside a model repo on 
                huggingface. co or the path to a directory containing model weights saved using the huggingface 
                `save_pretrained` method.
        """
        self.model_id = model_id

    def get_model(self, **kwargs: Any) -> PreTrainedModel:
        """Uses the huggingface AutoModelForCausalLM to retreive model.

        Args:
            **kwargs (Any): Additional keyword arguments for the `AutoModelForCausalLM.from_pretrained` method.

        Returns:
            PreTrainedModel: _description_
        """
        # Load the model using the AutoModelForCausalLM class
        # Setting device_map to auto here is super important, greatly affects runtime.
        return AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="auto", device_map="auto", **kwargs)
    
    def get_tokenizer(self, **kwargs: Any) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(self.model_id)
    
    @classmethod
    def __help__(cls) -> str:
        return "Any HuggingFace model compatible with AutoModelForCausalLM."