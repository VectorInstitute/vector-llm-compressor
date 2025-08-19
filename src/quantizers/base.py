
from abc import ABC, abstractmethod

from datasets import Dataset
from transformers.modeling_utils import PreTrainedModel


class QuantizerBase(ABC):
    """Base class for quantizers."""    
    @abstractmethod
    def quantize(self, model: PreTrainedModel, dataset: Dataset | None = None) -> PreTrainedModel:
        """Quantizes a hugging face transformers compatible model.

        Args:
            model (PreTrainedModel): The model to quantize.
            dataset (Dataset | IterableDataset | None): A hugging face datasets compatible calibration dataset to use 
                for quantization. Defaults to None.

        Returns:
            PreTrainedModel : The quantized model.
        """
        pass
    
    @classmethod
    @abstractmethod
    def __help__(cls) -> str:
        pass

