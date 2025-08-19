from abc import abstractmethod
from pathlib import Path

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers import Modifier
from transformers.modeling_utils import PreTrainedModel

from .base import QuantizerBase


class LLMCQuantizerBase(QuantizerBase):
    def __init__(self, log_dir: Path | str = "llmc_logs") -> None:
        """Base class for quantizaters that use the `llmcompressor` package.

        Args:
            log_dir (Path | str, optional): Where to save the output logs of the quantization process. Defaults to 
                "llmc_logs".
        """
        self.log_dir = log_dir

    @abstractmethod
    def get_recipe(self) -> str | Modifier | list[Modifier]:
        """Define the llmcompressor compatible recipe to use for quantization.

        Note that `llmcompressor` uses hugging face's `compressed-tensors` package under the hood. Therefore you can either get complicated with defining the quantization recipe very manually, or use one of the presets. `llmcompressor` uses the same presets schemes that are available in `compressed-tensors`. As of writing these are:

        ::
        
            # Unquantized (no-op)
            - "UNQUANTIZED"
            # Integer weight only schemes
            - "W8A16"
            - "W4A16"
            - "W4A16_ASYM"
            # Integer weight and activation schemes
            - "W8A8"
            - "INT8"  # alias for W8A8
            - "W4A8"
            # Float weight and activation schemes
            - "FP8"
            - "FP8_DYNAMIC"
            - "FP8_BLOCK"
            - "NVFP4A16"
            - "NVFP4"

        `llmcompressor` has poor documentation. You can use the `config_groups` argument to specify multiple different schemes for different "targets" (specific layers or tensors) within the model. You can also pass a single scheme to the `scheme` argument, which is then used for the whole model on the targets specified by the `targets` argument (hence targets are not specified within the scheme as with `config_groups`). The preferred approach is to pass a string key referencing one of the preset schemes described above to the `scheme` argument.
        All activation quantization in the presets is done dynamically, this setting is saved in the config file of the model and vLLM knows to do dynamic activation quantization during runtime from this setting. For static activation quantization you need to define your own scheme.

        Returns:
            str | Modifier | list[Modifier]: _description_
        """
        pass

    def quantize(self, model: PreTrainedModel, dataset: Dataset | None = None) -> PreTrainedModel:
        recipe = self.get_recipe()
        # llmcompressor has some datasets registered with their library. 
        # If you are manually passing a Dataset like we are, then most dataset args to oneshot do nothing.
        return oneshot(
            # model_args:
            model=model,
            # recipe_args:
            recipe=recipe, # type: ignore  # Type hint in llmcompressor is just wrong
            # dataset_args:
            dataset=dataset, # type: ignore  # Type hint in llmcompressor is just wrong
            num_calibration_samples=512 if dataset is None else len(dataset),  # Arg doesn't matter if dataset is None
            shuffle_calibration_samples=True,
            # other args
            output_dir=None,
            log_dir=str(self.log_dir),
        )
    
    @classmethod
    @abstractmethod
    def __help__(cls) -> str:
        pass

