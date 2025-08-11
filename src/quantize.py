from llmcompressor.args import parse_args
from transformers.hf_argparser import HfArgumentParser
from llmcompressor.args import ModelArguments, DatasetArguments
import sys
from argparse import ArgumentParser


def parse_llm_args():
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        required=True,
        type=str,
        help="A string specifying a model from the set of available models specified in the MODELS_KEY"
    )
    parser.add_argument(
        "dataset",
        required=True,
        type=str,
        help="One of the specified datasets"
    )
    parser.add_argument(
        "recipe",
        required=True,
        type=str,
        help="One of the predefined quantization recipes defined in this library."
    )



if __name__ == "__main__":
    parse_llm_args()
