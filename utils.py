import logging
import sys
from textwrap import TextWrapper

import datasets
import huggingface_hub
import matplotlib.pyplot as plt
import torch
import transformers
from IPython.display import set_matplotlib_formats


def set_plot_style():
    set_matplotlib_formats("pdf", "svg")
    # Apply custom plot style (previously defined in plotting.mplstyle)
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.figsize": (6, 4),
        "axes.prop_cycle": plt.cycler("color", [
            "0071bc", "f7931e", "c1272d", "009245", "ffde00", "9900cc",
        ]),
        "font.size": 12.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    logging.getLogger("matplotlib").setLevel(level=logging.ERROR)


def display_library_version(library):
    print(f"Using {library.__name__} v{library.__version__}")


def setup_chapter():
    # Check if we have a GPU
    if not torch.cuda.is_available():
        print("No GPU was detected! This notebook can be *very* slow without a GPU 🐢")
    else:
        print("GPU was detected! This notebook can be *very* fast with a GPU 🐰")
    # Give visibility on versions of the core libraries
    display_library_version(transformers)
    display_library_version(datasets)
    # Disable all info / warning messages
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    # Logging is only available for the chapters that don't depend on Haystack
    if huggingface_hub.__version__ == "0.0.19":
        huggingface_hub.logging.set_verbosity_error()
    # Use O'Reilly style for plots
    set_plot_style()


def wrap_print_text(print):
    """Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927"""

    def wrapped_func(text):
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper(
            width=80,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        return print("\n".join(wrapper.fill(line) for line in text.split("\n")))

    return wrapped_func


print = wrap_print_text(print)
