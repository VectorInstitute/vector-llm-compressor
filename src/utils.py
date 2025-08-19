from argparse import SUPPRESS, Action
import textwrap

from quantizers import QUANTIZER_HELP_KEY


def parse_kwargs(pairs):
    pass

def show_help(help_dict: dict[str, str]):
    for key, val in help_dict.items():
        print(f"\n{key}:")
        print(f"{textwrap.indent(textwrap.fill(val, width=89), prefix="\t")}")

class QuantizerHelpAction(Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs, default=SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        print("Available Quantizers:\n")
        show_help(QUANTIZER_HELP_KEY)
        parser.exit()
