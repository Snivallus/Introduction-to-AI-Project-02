import logging
import sys
from textwrap import TextWrapper
from typing import Callable, List

import datasets
import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from IPython.display import set_matplotlib_formats


def set_plot_style() -> None:
    """Configure matplotlib with custom O'Reilly-style plot defaults."""
    set_matplotlib_formats("pdf", "svg")
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.figsize": (6, 4),
        "axes.prop_cycle": plt.cycler("color", [
            "#0071bc", "#f7931e", "#c1272d", "#009245", "#ffde00", "#9900cc",
        ]),
        "font.size": 12.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    logging.getLogger("matplotlib").setLevel(level=logging.ERROR)


def display_library_version(library) -> None:
    """Print the name and version of a given Python module.

    Args:
        library: An imported module with a __version__ attribute.
    """
    print(f"Using {library.__name__} v{library.__version__}")


def setup_chapter() -> None:
    """Initialise the notebook environment for the chapter.

    Checks for GPU availability, displays core library versions,
    suppresses verbose logging, and applies the custom plot style.
    """
    if not torch.cuda.is_available():
        print("No GPU was detected! This notebook can be very slow "
              "without a GPU \U0001f422")
    else:
        print("GPU was detected! This notebook can be very fast "
              "with a GPU \U0001f430")
    display_library_version(transformers)
    display_library_version(datasets)
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    if huggingface_hub.__version__ == "0.0.19":
        huggingface_hub.logging.set_verbosity_error()
    set_plot_style()


def wrap_print_text(print_fn: Callable) -> Callable:
    """Wrap a print function to wrap long text at 80 characters.

    This is used to make long console output more readable in the
    notebook environment.

    Adapted from:
    https://stackoverflow.com/questions/27621655/how-to-overload-print-function

    Args:
        print_fn: The original print function to wrap.

    Returns:
        A wrapped print function that word-wraps its input text.
    """

    def wrapped_func(text: object) -> None:
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper(
            width=80,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        return print_fn("\n".join(wrapper.fill(line)
                                  for line in text.split("\n")))

    return wrapped_func


print = wrap_print_text(print)


def evaluate_classifiers(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_valid_raw: np.ndarray,
    y_valid: np.ndarray,
    class_names: List[str],
) -> None:
    """Train four classifiers on extracted features and compare performance.

    The classifiers are: Softmax (Logistic Regression), Random Forest,
    SVM (RBF kernel), and KNN.  A 2x2 confusion matrix grid is plotted
    alongside accuracy scores for each model.

    Args:
        X_train_raw: Training feature matrix of shape (n_train, n_features).
        y_train: Training labels of shape (n_train,).
        X_valid_raw: Validation feature matrix of shape (n_valid, n_features).
        y_valid: Validation labels of shape (n_valid,).
        class_names: List of class names for confusion matrix axes.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, \
        confusion_matrix

    # Scale features for distance-based classifiers (SVM, KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_valid_scaled = scaler.transform(X_valid_raw)

    classifiers = {
        "Softmax Regression":
            LogisticRegression(max_iter=10000),
        "Random Forest":
            RandomForestClassifier(n_estimators=100,
                                   min_samples_leaf=5,
                                   random_state=51),
        "SVM (RBF Kernel)":
            SVC(kernel="rbf"),
        "KNN (k=50, cosine)":
            KNeighborsClassifier(n_neighbors=50, 
                                 metric="cosine",
                                 weights="distance"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    results = {}
    for ax, (name, clf) in zip(axes, classifiers.items()):
        # Use scaled features for SVM and KNN; raw features for others
        if isinstance(clf, (SVC, KNeighborsClassifier)):
            X_train = X_train_scaled
            X_valid = X_valid_scaled
        else:
            X_train = X_train_raw
            X_valid = X_valid_raw

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
        results[name] = acc
        print(f"  {name:35s}  Accuracy: {acc:.4f}")

        # Plot normalised confusion matrix
        cm = confusion_matrix(y_valid, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        ax.set_title(f"{name}\nAccuracy: {acc:.4f}")

    plt.tight_layout()
    plt.show()