# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational NLP project that fine-tunes **DistilBERT** (a distilled BERT variant) for emotion recognition on Twitter text. The dataset has 6 emotion classes: sadness, joy, love, anger, fear, surprise.

The main workflow is contained in `project.ipynb`. Students complete 6 TODO tasks:
1. Build a character-level `token2idx` dictionary
2. Implement the `[CLS]` hidden state extraction function
3. Train and compare Scikit-Learn classifiers on extracted features
4. Tune hyperparameters for the fine-tuned model
5. Analyze classification errors and successes
6. Generate 5 adversarial examples and explain why they fool the model

## Key Files

| File | Purpose |
|------|---------|
| `project.ipynb` | Main notebook with all code and TODO tasks |
| `utils.py` | Utility functions: plot style setup, library version display, print wrapper |
| `install.py` | Dependency installation script (pip + git-lfs) |
| `requirements.txt` | Python dependencies (transformers, datasets, umap-learn, matplotlib) |
| `environment.yml` | Conda environment (Python 3.9, CUDA 11.3, pytorch-scatter, notebook) |
| `CODING_STYLE.md` | Coding conventions (PEP 8, Google-style docstrings, PyTorch conventions) |
| `plotting.mplstyle` | Matplotlib style sheet |

## Data & Model Assets

| Path | Description |
|------|-------------|
| `emotion/` | CARER emotion dataset (train=16k, validation=2k, test=2k) |
| `emotion/split/` | Pre-split parquet files (train/validation/test) |
| `emotion/unsplit/` | Full unsplit dataset (416k examples) |
| `distilbert-base-uncased/` | Local DistilBERT checkpoint (model, tokenizer, config) |
| `images/` | Architecture diagrams and visualizations for notebook markdown |
| `DistilBERT/` | LaTeX source and PDF for DistilBERT paper |

## Environment Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate book

# Or pip only
pip install -r requirements.txt
```

## Key Libraries

- **transformers** v4.16.2: AutoModel, AutoTokenizer, Trainer, pipeline APIs
- **datasets** v1.16.1: Dataset loading and processing (map, set_format)
- **torch** with CUDA 11.3: Model training and inference
- **scikit-learn**: Classifiers (LogisticRegression, DummyClassifier), metrics (accuracy, f1, confusion matrix)
- **umap-learn** v0.5.1: Dimensionality reduction for hidden state visualization
- **matplotlib**: Plotting (class distribution, confusion matrix, probability bars)

## Architecture (Notebook Pipeline)

The notebook follows this sequence, implemented in `project.ipynb`:

1. **Dataset loading** — Load emotion dataset via `datasets.load_dataset()`
2. **EDA** — Convert to pandas DataFrame, inspect class distribution, tweet length distribution
3. **Tokenization** — Character → WordPiece subword using DistilBERT tokenizer, applied via `Dataset.map()`
4. **Feature extraction** — Freeze DistilBERT, extract `[CLS]` hidden states (768-dim vectors) for all examples
5. **UMAP visualization** — Project hidden states to 2D for qualitative inspection
6. **Scikit-Learn classifier** — Train classifiers on extracted features (logistic regression baseline, plus other models as TODO)
7. **Fine-tuning** — End-to-end training with `Trainer` API using `AutoModelForSequenceClassification`
8. **Error analysis** — Sort validation samples by cross-entropy loss; inspect high-loss and low-loss cases
9. **Inference** — Save model, load into `pipeline("text-classification")`, predict on custom tweets

## TODO Tasks (Student Assignments)

- **Task 1** (`cell-34`): Build `token2idx` dict mapping characters to integers for character-level tokenization
- **Task 2** (`cell-112`): Complete `extract_hidden_states()` — move inputs to device, run model forward, return `[CLS]` vector
- **Task 3** (`cell-132`): Try different Scikit-Learn classifiers, compare results against LogisticRegression baseline
- **Task 4** (`cell-154`): Tune `TrainingArguments` parameters (learning rate, epochs, batch size, weight decay)
- **Task 5** (`cell-173`): Analyze high-loss (misclassified) and low-loss (correctly classified) samples
- **Task 6** (`cell-185`): Generate 5 adversarial examples that fool the model, explain why

## Common Commands

```bash
# Launch Jupyter notebook
jupyter notebook project.ipynb

# Install dependencies
python install.py
```
