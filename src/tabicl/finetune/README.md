# `tabicl.finetune`

Fine-tuning scripts and utilities for TabICL.

## Installation

The fine-tuning scripts require extra dependencies declared under the `finetune` optional group:

```bash
uv sync --extra finetune
```

## Folder structure

```
tabicl/finetune/
├── make_data.py                 # Generate and persist a synthetic dataset to disk
├── sft.py                       # Supervised fine-tuning (cross-entropy loss)
├── dpo.py                       # DPO fine-tuning (preference optimisation)
├── evaluate.py                  # Evaluate a checkpoint on the Iris dataset (raw PyTorch)
├── generate_preference_data.py  # Preference-pair generators used by dpo.py
├── utils.py                     # Shared helpers (data loading, model utils, …)
└── demo.ipynb                   # End-to-end demo notebook
```

## Scripts

All scripts use [tyro](https://github.com/brentyi/tyro) for CLI argument parsing. Run any script with `--help` to see the full list of options.

---

### `make_data.py` — generate a dataset

Creates a synthetic imbalanced multi-class dataset with `sklearn.make_classification`, applies stratified train/val/test splits, fits a `StandardScaler` on the training fold, and saves all six arrays (`X`/`y` for train, val, test) to a single `.npz` file.

The saved file can be passed to `sft.py` or `dpo.py` via `--dataset-path` to ensure multiple runs operate on identical data.

```bash
# All defaults (saves to data/dataset.npz)
uv run python -m tabicl.finetune.make_data

# Custom output path
uv run python -m tabicl.finetune.make_data --output-path data/my_dataset.npz

# Custom dataset shape
uv run python -m tabicl.finetune.make_data --n-samples 2000 --n-classes 4
```

---

### `sft.py` — supervised fine-tuning

Fine-tunes a pretrained TabICL checkpoint with cross-entropy loss. Uses an episodic training strategy: each batch consists of random in-context/query splits of the dataset, augmenting the effective training data across steps.

```bash
# All defaults (generates data on the fly)
uv run python -m tabicl.finetune.sft

# Use a pre-generated dataset
uv run python -m tabicl.finetune.sft --dataset-path data/dataset.npz

# Custom hyperparameters
uv run python -m tabicl.finetune.sft --epochs 30 --lr 5e-6

# Save the fine-tuned checkpoint
uv run python -m tabicl.finetune.sft --output-dir runs/my_sft --save-ckpt
```

TensorBoard logs are written to `--output-dir` (default: `runs/sft`). The checkpoint is saved as `<output-dir>/tabicl_sft.ckpt` when `--save-ckpt` is set.

---

### `dpo.py` — Direct Preference Optimisation fine-tuning

Fine-tunes a pretrained TabICL checkpoint with DPO loss. A frozen copy of the initial checkpoint serves as the reference model. Preference pairs (chosen/rejected) are generated on-the-fly from one of four strategies controlled by `--preference-generator`:

| Strategy | Description |
|---|---|
| `random` | Sample a uniformly random incorrect class (default) |
| `majority` | Use the globally most frequent class as the rejected label |
| `hard` | Use the highest-probability wrong class from the frozen reference model |
| `confusion` | Use the most historically confused class per true label (one-time confusion-matrix pass) |

```bash
# All defaults
uv run python -m tabicl.finetune.dpo

# Use a pre-generated dataset
uv run python -m tabicl.finetune.dpo --dataset-path data/dataset.npz

# Choose a preference strategy
uv run python -m tabicl.finetune.dpo --preference-generator hard
uv run python -m tabicl.finetune.dpo --preference-generator confusion

# Custom DPO hyperparameters
uv run python -m tabicl.finetune.dpo --beta 0.5 --epochs 10 --lr 1e-5

# Save the fine-tuned checkpoint
uv run python -m tabicl.finetune.dpo --output-dir runs/my_dpo --save-ckpt
```

TensorBoard logs are written to `--output-dir` (default: `runs/dpo`). The checkpoint is saved as `<output-dir>/tabicl_dpo.ckpt` when `--save-ckpt` is set.

---

By: Abdelhakim Benechehab.
