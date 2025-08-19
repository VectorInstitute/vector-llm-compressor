from typing import Literal
from pathlib import Path

from llmcompressor.modifiers import Modifier
from compressed_tensors.quantization import ActivationOrdering
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

from .registry import register_recipe
from .llmcompressor_recipe import LLMCQuantizerBase


@register_recipe("llmc_sqgptq_w8a8int8")
class LLMC_SQGPTQInt8(LLMCQuantizerBase):
    def __init__(
        self,
        smoothing_strength: float = 0.8,
        block_size: int = 128,
        dampening_frac: float = 0.001,
        offload_hessians: bool = False,
        actorder: Literal["basic", "weight", "group"] = "weight",
        ignore: list[str] = ["lm_head"],
        log_dir: Path | str = "llmc_logs",
    ) -> None:
        """A `llmcompressor` based implementation for weight and activation quantization to `int8`.

        - weights: `int8`, static
        - activations: `int8`, dynamic
        - kv_cache: full precision

        First the SmoothQuant algorithm adjusts the model weights to reduce the dynamic range of (smooth) the
        activations during runtime. Then the GPTQ algorithm is used for static weight quantization and quantizes the
        weights to `int8`. Activations are quantized dynamically at runtime (dynamic quantization settings saved in
        model config). Only linear layers are quantized.

        Args:
            smoothing_strength (float, optional): A float between 0 and 1 that controls the strength of the smoothing
                appliad by SmoothQuant. Defaults to 0.8.
            block_size (int, optional): The number of columns that GPTQ should compress in one pass during weight
                quantization. Defaults to 128.
            dampening_frac (float, optional): Amount of dampening to apply to the Hessian (as a fraction of the
                diagonal norm) used by GPTQ during weight quantization. Defaults to 0.001.
            offload_hessians (bool, optional): Setting to True decreases memory usage during PTQ but increases runtime
                of PTQ. Defaults to False.
            actorder (Literal[&quot;basic&quot;, &quot;weight&quot;, &quot;group&quot;], optional): Strategy used by
                GPTQ for 'activation ordering' when quantizating the weights. Determines the quantization order and
                grouping of columns in the weight tensor. The "basic" strategy applies no activation ordering. The
                "weight" strategy improves accuracy and no cost. The "group" strategy further improves accuracy at the
                cost of slightly increased latency. Defaults to "weight". See the following for more details:
                https://github.com/vllm-project/vllm/pull/8135
            ignore (list[str], optional): List of module class names or submodule names to leave unquantized. Pass an
                empty list to quantize all Linear layers in the model. Defaults  to ["lm_head"].
            log_dir (Path | str, optional): Where to save the `llmcompressor` output logs. A directory will be created
                if it does not exist. Defaults to "llmc_logs".
        """
        # Run parent init to set log dir
        super().__init__(log_dir=log_dir)
        # Set other reciper parameters
        self.smoothing_strength = smoothing_strength
        self.block_size = block_size
        self.dampening_frac = dampening_frac
        self.offload_hessians = offload_hessians
        self.ignore = ignore
        assert actorder in ["basic", "weight", "group"], "actorder must be one of: 'basic', 'weight' or 'group'"
        self.actorder = None if actorder == "basic" else ActivationOrdering(actorder)

    def get_recipe(self) -> list[Modifier]:
        """Recipe for W8A8-Int8 quantization. Targets are set to linear and we use the preset `W8A8` settings.

            Smoothquant is run before quantization to make activations easier to quantize during runtime.

        Returns:
            list[Modifier]: The `llmcompressor` recipe for `W8A8-Int8` quantization
        """
        recipe = [
            SmoothQuantModifier(smoothing_strength=self.smoothing_strength),
            GPTQModifier(
                block_size=self.block_size,
                dampening_frac=self.dampening_frac,
                offload_hessians=self.offload_hessians,
                actorder=self.actorder,
                ignore=self.ignore,
                targets=["Linear"],  # Common practice to target linear layers as this is where most of the weights are.
                scheme="W8A8",  # Preset scheme for int8 static weight PTQ and dynamic activation quantization
            ),
        ]

        return recipe

    @classmethod
    def __help__(cls) -> str:
        return (
            "`llmcompressor` implementation of W8A8-Int8. "
            "SmoothQuant, GPTQ for static weight quantization to `int8`, "
            "dynamic per token activation quantization to `int8` at runtime."
        )
