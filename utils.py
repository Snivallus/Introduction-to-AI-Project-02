import logging
import warnings
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
    # Suppress harmless PyTorch gather warning during predict()
    warnings.filterwarnings(
        "ignore", message="Was asked to gather along dimension 0")
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

    def wrapped_func(text: object, **kwargs) -> None:
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper(
            width=80,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        return print_fn("\n".join(wrapper.fill(line)
                                  for line in text.split("\n")),
                        **kwargs)

    return wrapped_func


print = wrap_print_text(print)

SEED = 51


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
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

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
    ckpt_dir: str = "./checkpoints",
    num_epochs: int = 10,
    early_stopping_patience: int = 2,
    device_id: int = 0,
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
        ckpt_dir: Directory for experiment checkpoints.
        num_epochs: Maximum number of training epochs.
        early_stopping_patience: Stop if no eval improvement
            after this many epochs.
        device_id: GPU device index for this worker.

    Returns:
        Dict with keys: ``experiment_id``, ``learning_rate``,
        ``batch_size``, ``weight_decay``, ``best_val_accuracy``,
        ``epochs_trained``.
    """
    import os
    import warnings
    from pathlib import Path
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    # Suppress harmless warnings in subprocess workers
    warnings.filterwarnings(
        "ignore", message="Was asked to gather along dimension 0")
    transformers.logging.set_verbosity_error()

    os.makedirs(ckpt_dir, exist_ok=True)

    output_dir = Path(ckpt_dir) / f"exp_{exp_idx:02d}"

    print(f"[{exp_idx}] lr={learning_rate}, bs={batch_size}, "
          f"wd={weight_decay}  ...  ", end="\n", flush=True)

    device = torch.device(f"cuda:{device_id}"
                          if torch.cuda.is_available() else "cpu")

    model = (AutoModelForSequenceClassification
             .from_pretrained(model_ckpt, num_labels=num_labels)
             .to(device))

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=max(1, len(train_dataset) // batch_size // num_epochs // 5),
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
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
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience),
        ],
    )

    trainer.train()
    # Save best model weights to disk (load_best_model_at_end loads them
    # into memory, but we need an explicit save for future loading)
    trainer.save_model(str(output_dir))

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
            jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(group))
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
    num_epochs: int = 10,
    early_stopping_patience: int = 2,
    ckpt_dir: str = "./checkpoints",
    results_dir: str = "./figures",
) -> None:
    """Run a 3x3x3 grid search in parallel across all available GPUs.

    Each combination is trained independently from the same pretrained
    checkpoint.  Experiments are distributed across GPUs via round-robin
    using ``ProcessPoolExecutor`` with a spawn context.  Results and
    analysis plots are saved to ``results_dir``.

    Args:
        model_ckpt: Hugging Face model checkpoint identifier.
        train_dataset: Training dataset with "label" column.
        eval_dataset: Validation dataset with "label" column.
        num_labels: Number of output classes.
        ckpt_dir: Directory for per-experiment checkpoints.
        results_dir: Directory for CSV results and analysis PDF.
    """
    import os
    import concurrent.futures as cf
    import multiprocessing as mp
    from itertools import product
    import pandas as pd

    os.makedirs(results_dir, exist_ok=True)

    param_grid = {
        "learning_rate": [1e-5, 3e-5, 1e-4],
        "batch_size": [8, 16, 32],
        "weight_decay": [0.01, 0.1, 1.0],
    }

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    num_gpus = torch.cuda.device_count()
    max_workers = max(1, num_gpus)

    print(f"Total experiments: {len(combinations)}")
    print(f"GPUs available: {num_gpus}, Workers: {max_workers}")

    experiment_args = []
    for exp_idx, (lr, bs, wd) in enumerate(combinations, 1):
        device_id = (exp_idx - 1) % max_workers if num_gpus > 0 else 0
        experiment_args.append((
            exp_idx, lr, bs, wd, model_ckpt,
            train_dataset, eval_dataset, num_labels,
            ckpt_dir, num_epochs, early_stopping_patience, device_id,
        ))

    results = []

    if max_workers > 1:
        ctx = mp.get_context("spawn")
        print(f"\nStarting parallel execution with {max_workers} workers...")
        print(f"Multiprocessing context: {ctx.get_start_method()}\n")

        with cf.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx,
        ) as executor:
            future_map = {
                executor.submit(train_finetune_experiment, *args): args
                for args in experiment_args
            }
            for future in cf.as_completed(future_map):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    args = future_map[future]
                    print(f"[{args[0]:02d}] FAILED: {e}")
    else:
        print(f"\nRunning sequentially (no multi-GPU available)...\n")
        for args in experiment_args:
            result = train_finetune_experiment(*args)
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        Path(results_dir) / "hyperparameter_tuning_results.csv", index=False)

    print("\n=== Hyperparameter Tuning Results ===")
    print(results_df.sort_values("best_val_accuracy", ascending=False)
          .to_string(index=False))

    plot_hyperparameter_effects(
        results_df, save_dir=results_dir, metric="best_val_accuracy")


def gradient_attack(
    text: str,
    target_label: int,
    model: "transformers.AutoModelForSequenceClassification",
    tokenizer: "transformers.AutoTokenizer",
    num_steps: int = 10,
    max_replacements: int = 3,
) -> Dict[str, Any]:
    """Generate an adversarial example via HotFlip-style token attack.

    For each attack step, the gradient of the target-label loss w.r.t. the
    token embeddings is used to select a token replacement that maximally
    pushes the prediction toward the target class.  Only complete word
    tokens (not ``##`` subword continuations or special tokens) are
    considered for replacement.

    Args:
        text: Original input text to attack.
        target_label: Desired misclassification label index.
        model: The fine-tuned model in evaluation mode.
        tokenizer: The pretrained tokenizer.
        num_steps: Maximum number of attack iterations.
        max_replacements: Maximum number of tokens to replace.

    Returns:
        Dict with keys: ``original_text``, ``adversarial_text``,
        ``original_prediction``, ``adversarial_prediction``,
        ``target_label``, ``success``, ``num_replacements``,
        ``original_probs``, ``adversarial_probs``.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    model.eval()
    device = next(model.parameters()).device
    class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # ---- encode original text and get baseline prediction ----
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]          # [seq_len]
    attention_mask = inputs["attention_mask"][0]

    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0),
                       attention_mask=attention_mask.unsqueeze(0)).logits[0]
        probs = F.softmax(logits, dim=-1)
        orig_pred = int(logits.argmax())

    # ---- identify attackable token positions (skip special + subword) ----
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                   tokenizer.pad_token_id, tokenizer.unk_token_id,
                   tokenizer.mask_token_id}
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attack_positions = [
        i for i, t in enumerate(tokens)
        if input_ids[i].item() not in special_ids
        and not t.startswith("##")
    ]

    if not attack_positions:
        return {"original_text": text, "adversarial_text": text,
                "success": False, "num_replacements": 0}

    current_ids = input_ids.clone()
    embedding = model.distilbert.embeddings.word_embeddings
    replacements = 0
    current_text = text
    current_attention_mask = attention_mask

    print(f"  Attack: '{text}' -> target={class_names[target_label]}")

    for step in range(1, num_steps + 1):
        # HotFlip: compute gradient of target loss w.r.t. input embeddings
        model.zero_grad()
        emb_out = embedding(current_ids.unsqueeze(0))
        outputs = model(
            inputs_embeds=emb_out,
            attention_mask=current_attention_mask.unsqueeze(0),
            labels=torch.tensor([target_label]).to(device))
        loss = outputs.loss
        emb_grad = torch.autograd.grad(loss, emb_out)[0][0]  # [seq_len, 768]

        # Score = emb_grad · embedding[candidate] — lower = better for target
        scores = torch.matmul(emb_grad, embedding.weight.T)  # [seq_len, vocab]
        for pos in attack_positions:
            scores[pos, current_ids[pos]] = float("inf")

        best_pos, best_candidate = None, None
        best_score = float("inf")
        for pos in attack_positions:
            min_val, min_idx = scores[pos].min(dim=-1)
            if min_val.item() < best_score:
                best_score = min_val.item()
                best_pos = pos
                best_candidate = min_idx.item()

        if best_candidate is None:
            print(f"    Step {step}: no suitable replacement, stopping.")
            break

        replacements += 1
        current_ids[best_pos] = best_candidate
        current_text = tokenizer.decode(current_ids, skip_special_tokens=True)

        # Re-tokenize and update sequence-dependent variables
        new_inputs = tokenizer(current_text, return_tensors="pt").to(device)
        current_ids = new_inputs["input_ids"][0].clone()
        current_attention_mask = new_inputs["attention_mask"][0]
        new_tokens = tokenizer.convert_ids_to_tokens(current_ids)
        attack_positions = [
            j for j, t in enumerate(new_tokens)
            if current_ids[j].item() not in special_ids
            and not t.startswith("##")
        ]
        tokens = new_tokens

        with torch.no_grad():
            new_pred = int(model(**new_inputs).logits[0].argmax())
        print(f"    Step {step}: replaced -> "
              f"'{tokenizer.decode(best_candidate)}' "
              f"(pred={class_names[new_pred]})")

        if new_pred == target_label:
            print(f"  -> Attack succeeded!")
            break

        if replacements >= max_replacements:
            print(f"  -> Max replacements ({max_replacements}) reached.")
            break

    # Final evaluation
    final_inputs = tokenizer(current_text, return_tensors="pt").to(device)
    with torch.no_grad():
        final_logits = model(**final_inputs).logits[0]
        final_probs = F.softmax(final_logits, dim=-1)
        adv_pred = int(final_logits.argmax())
    with torch.no_grad():
        orig_logits = model(input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0)).logits[0]
        orig_probs = F.softmax(orig_logits, dim=-1)
        orig_pred = int(orig_logits.argmax())

    print(f"  Original: '{text}' -> {class_names[orig_pred]} "
          f"({orig_probs[orig_pred]:.3f})")
    print(f"  Adversarial: '{current_text}' -> {class_names[adv_pred]} "
          f"({final_probs[adv_pred]:.3f})")
    print()

    return {
        "original_text": text,
        "adversarial_text": current_text,
        "original_prediction": class_names[orig_pred],
        "adversarial_prediction": class_names[adv_pred],
        "target_label": class_names[target_label],
        "success": adv_pred == target_label,
        "num_replacements": replacements,
        "original_probs": orig_probs.cpu().numpy(),
        "adversarial_probs": final_probs.cpu().numpy(),
    }