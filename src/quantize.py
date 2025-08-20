from argparse import ArgumentParser

from utils import QuantizerHelpAction
from models import MODELS_KEY
from quantizers import QUANTIZER_KEY
from models.auto import AutoHFModel
from calibration_datasets import DATASETS_KEY
from calibration_datasets.auto import AutoHFDataset


def parse_args() -> dict:
    """CLI interface for parsing CLI arguments and returning a dict.

    Returns:
        dict: A dictionary containing the CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Path to a pretrained model, or a model identifier from huggingface.co/models. (string)"
    )
    # So we are going to implement our own thing here rather than using oneshot, but we will include an auto class
    parser.add_argument(
        "dataset",
        type=str,
        help="The name of a dataset from the huggingface datasets library to use for calibration. (string)",
    )
    parser.add_argument(
        "quantizer", type=str, help="One of the predefined quantizaters defined in this library. (string)"
    )
    parser.add_argument("output_dir", type=str, help="Path to a folder in which to save the quantized model. (string)")
    parser.add_argument(
        "--quantizers",
        required=False,
        action=QuantizerHelpAction,
        help="List available predefined quantization recipes and exit.",
    )
    parser.add_argument(
        "--split",
        required=False,
        default=None,
        type=str,
        help="The name of the split to use from the dataset specified by the `dataset` argument. (string)",
    )
    parser.add_argument(
        "--num-samples",
        required=False,
        type=int,
        default=512,
        help="The number of samples from the dataset to use for calibration. Defaults to 512. (integer)",
    )
    parser.add_argument(
        "--max-seq-length",
        required=False,
        type=int,
        default=2048,
        help="The maximum sequence length (in tokens) to use during calibration. Defaults to 2048. (integer)",
    )
    parser.add_argument(
        "--model-args",
        nargs="*",
        default=[],
        help="Keyword arguments for the pretrained model in `key=value` format delimited by spaces. WARNING: This flag/feature is not yet implemented.",
    )
    parser.add_argument(
        "--quantizer-args",
        nargs="*",
        default=[],
        help="Keyword arguments for the quantizater in `key=value` format delimited by spaces. WARNING: This flag/feature is not yet implemented.",
    )
    # Would be cool to offer a flag where if you pass the recipe/dataset/model it prints out the special keyword arguments for the recipe/dataset/model
    # Probably should restructure to first split args into dataset, model and recipe. Then for each add flags for additional args that are specific to a specific recipe/dataset/model. Then base class should by default accept kwargs in constructor, and just retrieve the specific kwargs they expect.

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    pretrained = MODELS_KEY[args["model"]]() if args["model"] in MODELS_KEY else AutoHFModel(model_id=args["model"])
    model = pretrained.get_model()
    tokenizer = pretrained.get_tokenizer()

    if args["dataset"] in DATASETS_KEY:
        dataset = DATASETS_KEY[args["dataset"]]().get_dataset(tokenizer)
    else:
        assert args["split"] is not None, (
            "If using AutoHFDataset you must define which split to use using the `--split` flag"
        )
        dataset = AutoHFDataset(
            dataset_id=args["dataset"],
            split=args["split"],
            num_samples=args["num_samples"],
            max_sequence_length=args["max_seq_length"],
        ).get_dataset(tokenizer)

    if args["quantizer"] not in QUANTIZER_KEY:
        raise ValueError(
            f"{args['quantizer']} is not one of the registered/predefined quantizers. Use --quantizers flag to see available quantizers."
        )
    else:
        quantizer = QUANTIZER_KEY[args["quantizer"]]()

    model = quantizer.quantize(model, dataset)
    # llmcompressor adds the save_compressed argument to save_pretrained. We might want to change pipeline to have quantizers accept outputdir and save model.
    model.save_pretrained(save_directory=args["output_dir"], save_compressed=True)
