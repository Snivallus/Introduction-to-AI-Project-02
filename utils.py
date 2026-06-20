import logging
from textwrap import TextWrapper
from typing import Any, Callable, Dict, List
from pathlib import Path

import datasets
import huggingface_hub
import matplotlib.pyplot as plt
import pandas as pd
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
    save_dir: str = "./figures",
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
        save_dir: Directory to save the figure as PDF (created if needed).
            Set to None to skip saving.
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
    if save_dir is not None:
        import os
        from pathlib import Path
        os.makedirs(save_dir, exist_ok=True)
        save_path = Path(save_dir) / "classifier_comparison.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()

    # Print sorted comparison
    print("\n--- Performance Comparison ---")
    for rank, (name, acc) in enumerate(
            sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {rank}. {name:35s}  {acc:.4f}")


def train_finetune_experiment(
    exp_idx: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    model_ckpt: str,
    train_dataset: "datasets.Dataset",
    eval_dataset: "datasets.Dataset",
    num_labels: int,
    compute_metrics_fn: Callable,
    ckpt_dir: str = "./checkpoints",
    num_epochs: int = 10,
    early_stopping_patience: int = 3,
) -> Dict[str, Any]:
    """Train one fine-tuning experiment with given hyperparameters.

    A fresh ``AutoModelForSequenceClassification`` is loaded from
    ``model_ckpt`` so that each experiment starts from the same
    pretrained weights.  Training metrics are extracted from the
    Trainer's built-in ``log_history``; checkpoints are saved to
    ``ckpt_dir``.

    Args:
        exp_idx: Experiment index (1-based) for naming.
        learning_rate: Peak learning rate for the optimizer.
        batch_size: Batch size per device for train and eval.
        weight_decay: L2 regularisation weight decay.
        model_ckpt: Hugging Face model checkpoint identifier.
        train_dataset: Training dataset with "label" column.
        eval_dataset: Validation dataset with "label" column.
        num_labels: Number of output classes.
        compute_metrics_fn: Metric function for the Trainer
            (see ``transformers.Trainer``).
        ckpt_dir: Directory for experiment checkpoints.
        num_epochs: Maximum number of training epochs.
        early_stopping_patience: Stop if no eval improvement
            after this many epochs.

    Returns:
        Dict with keys: ``experiment_id``, ``learning_rate``,
        ``batch_size``, ``weight_decay``, ``best_val_accuracy``,
        ``epochs_trained``.
    """
    import os
    from pathlib import Path
    from transformers import (
        AutoModelForSequenceClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    os.makedirs(ckpt_dir, exist_ok=True)

    output_dir = Path(ckpt_dir) / f"exp_{exp_idx:02d}"

    print(f"[{exp_idx}] lr={learning_rate}, bs={batch_size}, "
          f"wd={weight_decay}  ...  ", end="", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = (AutoModelForSequenceClassification
             .from_pretrained(model_ckpt, num_labels=num_labels)
             .to(device))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=max(
            1, len(train_dataset) // batch_size // 10),
        remove_unused_columns=False,
        disable_tqdm=True,
        report_to="none",
        log_level="error",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience),
        ],
    )

    trainer.train()

    # Extract per-epoch evaluation accuracy from training logs
    log_history = trainer.state.log_history
    eval_accuracies = [
        entry["eval_accuracy"]
        for entry in log_history
        if "eval_accuracy" in entry
    ]
    best_val_accuracy = max(eval_accuracies) if eval_accuracies else 0.0
    epochs_trained = len(eval_accuracies)

    print(f"best acc={best_val_accuracy:.4f}, "
          f"epochs={epochs_trained}/{num_epochs}")

    return {
        "experiment_id": exp_idx,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "best_val_accuracy": best_val_accuracy,
        "epochs_trained": epochs_trained,
    }


def plot_hyperparameter_effects(
    results_df: "pd.DataFrame",
    save_dir: str = "./figures",
    metric: str = "best_val_accuracy",
) -> None:
    """Plot each hyperparameter's effect on validation accuracy.

    For each hyperparameter (learning_rate, batch_size, weight_decay),
    a line chart shows the mean accuracy with :math:`\\pm 1` standard
    deviation.  Individual experiment results are overlaid as scatter
    points with jitter.  The figure is saved as a 300 dpi PDF.

    Args:
        results_df: DataFrame with columns ``learning_rate``,
            ``batch_size``, ``weight_decay``, and the metric column.
        save_dir: Directory to save the PDF figure.
        metric: Column name for the accuracy metric to plot.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    param_configs = [
        ("learning_rate", "Learning Rate"),
        ("batch_size", "Batch Size"),
        ("weight_decay", "Weight Decay"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    for ax, (param_col, param_label) in zip(axes, param_configs):
        grouped = results_df.groupby(param_col)[metric]
        means = grouped.mean()
        stds = grouped.std()

        x = range(len(means))
        ax.errorbar(x, means.values, yerr=stds.values, fmt="-o",
                     capsize=5, capthick=2, linewidth=2, markersize=8,
                     color="#2196F3", ecolor="#FF5722")

        for xi, (val, group) in enumerate(grouped):
            jitter = np.random.default_rng(42).uniform(-0.05, 0.05,
                                                       len(group))
            ax.scatter(xi + jitter, group.values, alpha=0.5,
                       color="#FF5722", s=30)

        tick_labels = [str(v) for v in means.index]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(param_label)
        ax.set_ylabel("Best Validation Accuracy")
        ax.set_title(f"Effect of {param_label}")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hyperparameter Tuning: Effect Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "hyperparameter_effects.pdf",
                dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path / 'hyperparameter_effects.pdf'}")


def tune_hyperparameters(
    model_ckpt: str,
    train_dataset: "datasets.Dataset",
    eval_dataset: "datasets.Dataset",
    num_labels: int,
    compute_metrics_fn: Callable,
    ckpt_dir: str = "./checkpoints",
    results_dir: str = "./figures",
) -> None:
    """Run a 3x3x3 grid search over learning rate, batch size, and
    weight decay using the Hugging Face ``Trainer`` API.

    Each combination is trained independently from the same pretrained
    checkpoint.  Results and analysis plots are saved to ``results_dir``.

    Args:
        model_ckpt: Hugging Face model checkpoint identifier.
        train_dataset: Training dataset with "label" column.
        eval_dataset: Validation dataset with "label" column.
        num_labels: Number of output classes.
        compute_metrics_fn: Metric function for the Trainer
            (see ``transformers.Trainer``).
        ckpt_dir: Directory for per-experiment checkpoints.
        results_dir: Directory for CSV results and analysis PDF.
    """
    import os
    from itertools import product
    import pandas as pd

    os.makedirs(results_dir, exist_ok=True)

    param_grid = {
        "learning_rate": [5e-4, 1e-3, 5e-3],
        "batch_size": [4, 8, 16],
        "weight_decay": [0, 0.01, 0.1],
    }

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    print(f"Total experiments: {len(combinations)}")

    results = []
    for exp_idx, (lr, bs, wd) in enumerate(combinations, 1):
        result = train_finetune_experiment(
            exp_idx=exp_idx,
            learning_rate=lr,
            batch_size=bs,
            weight_decay=wd,
            model_ckpt=model_ckpt,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_labels=num_labels,
            compute_metrics_fn=compute_metrics_fn,
            ckpt_dir=ckpt_dir,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        Path(results_dir) / "hyperparameter_tuning_results.csv", index=False)

    # Print sorted results
    print("\n=== Hyperparameter Tuning Results ===")
    print(results_df.sort_values("best_val_accuracy", ascending=False)
          .to_string(index=False))

    # Plot hyperparameter effect analysis
    plot_hyperparameter_effects(results_df, save_dir=results_dir, metric="best_val_accuracy")