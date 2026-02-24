"""Shared utilities for TabICL fine-tuning scripts (sft.py, dpo.py)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from tabicl import TabICL


REPO_ID = "jingang/TabICL"


# ---------------------------------------------------------------------------
# Seeding / device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg) -> torch.device:
    """Return the compute device specified by ``cfg.device``, or auto-detect."""
    if cfg.device is not None:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint loading / freezing
# ---------------------------------------------------------------------------

def load_pretrained(cfg) -> tuple[TabICL, dict]:
    """Load a pretrained TabICL checkpoint and return ``(model, model_config)``.

    Looks up ``cfg.checkpoint_path`` (local file) or downloads
    ``cfg.checkpoint_version`` from HuggingFace Hub.
    """
    if cfg.checkpoint_path is not None:
        ckpt_path = Path(cfg.checkpoint_path)
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    else:
        try:
            ckpt_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=cfg.checkpoint_version,
                    local_files_only=True,
                )
            )
        except LocalEntryNotFoundError:
            print(f"Downloading '{cfg.checkpoint_version}' from HF Hub ({REPO_ID}) …")
            ckpt_path = Path(
                hf_hub_download(repo_id=REPO_ID, filename=cfg.checkpoint_version)
            )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "config" in ckpt and "state_dict" in ckpt, "Malformed checkpoint."

    model = TabICL(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["config"]


def apply_freeze(model: TabICL, cfg) -> None:
    """Freeze sub-networks in-place according to ``cfg.freeze_{col,row,icl}``."""
    if cfg.freeze_col:
        for p in model.col_embedder.parameters():
            p.requires_grad = False
    if cfg.freeze_row:
        for p in model.row_interactor.parameters():
            p.requires_grad = False
    if cfg.freeze_icl:
        for p in model.icl_predictor.parameters():
            p.requires_grad = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_and_split(cfg) -> tuple[np.ndarray, ...]:
    """Generate an imbalanced synthetic dataset and return train/val/test splits.

    Uses ``cfg.val_size``, ``cfg.test_size``, and ``cfg.seed``.
    """
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_classes=cfg.n_classes,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        weights=cfg.weights,
        random_state=cfg.dataset_random_state,
    )

    # First split: carve out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    # Second split: validation from the remaining
    relative_val = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, random_state=cfg.seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train).astype(np.int64)
    y_val   = le.transform(y_val).astype(np.int64)
    y_test  = le.transform(y_test).astype(np.int64)

    print(
        f"Dataset splits — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}\n"
        f"Classes: {le.classes_}  weights={cfg.weights}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


class EpisodicDataset(TensorDataset):
    """Generate random in-context episodes from a fixed tabular dataset.

    Each call to ``__getitem__`` samples a fresh random train/test split of
    the full dataset so that the DataLoader produces diverse episodes per epoch.

    Parameters
    ----------
    X : np.ndarray of shape (N, H)
    y : np.ndarray of shape (N,)
    context_fraction : float
        Fraction of N used as in-context training examples.
    seed : int
        Base seed; per-item seeds are derived from it.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_fraction: float = 0.7,
        seed: int = 42,
    ):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.context_fraction = context_fraction
        self.n = len(y)
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        perm = rng.permutation(self.n)

        context_size = max(1, int(self.n * self.context_fraction))
        ctx_idx = perm[:context_size]
        qry_idx = perm[context_size:]

        # If all samples landed in context (tiny dataset edge-case), keep one query
        if len(qry_idx) == 0:
            ctx_idx, qry_idx = perm[:-1], perm[-1:]

        X_ctx = self.X[ctx_idx]
        y_ctx = self.y[ctx_idx]
        X_qry = self.X[qry_idx]
        y_qry = self.y[qry_idx]

        X_ep     = torch.cat([X_ctx, X_qry], dim=0)   # (T, H)
        y_ctx_ep = y_ctx.float()                        # (ctx_size,)
        y_qry_ep = y_qry                                # (qry_size,) int64

        return X_ep, y_ctx_ep, y_qry_ep, context_size


def collate_episodes(batch):
    """Pad variable-length episodes to the same (T, H) shape for batching.

    Returns
    -------
    X_batch     : (B, T_max, H)
    y_ctx_batch : (B, ctx_max)  – float for model input
    y_qry_batch : (B, qry_max)  – int64 for loss computation (-100 = padding)
    ctx_sizes   : list[int]
    """
    X_list, y_ctx_list, y_qry_list, ctx_sizes = zip(*batch)

    T_max   = max(x.shape[0] for x in X_list)
    ctx_max = max(y.shape[0] for y in y_ctx_list)
    qry_max = max(y.shape[0] for y in y_qry_list)
    H = X_list[0].shape[1]
    B = len(X_list)

    X_batch     = torch.zeros(B, T_max, H)
    y_ctx_batch = torch.zeros(B, ctx_max)
    y_qry_batch = torch.full((B, qry_max), fill_value=-100, dtype=torch.int64)

    for i, (X, yc, yq, _) in enumerate(zip(X_list, y_ctx_list, y_qry_list, ctx_sizes)):
        X_batch[i, :X.shape[0]] = X
        y_ctx_batch[i, :yc.shape[0]] = yc
        y_qry_batch[i, :yq.shape[0]] = yq

    return X_batch, y_ctx_batch, y_qry_batch, list(ctx_sizes)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: TabICL,
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_qry: np.ndarray,
    y_qry: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    """Run one evaluation episode and return ``(cross-entropy, accuracy)``.

    Uses all of ``X_ctx``/``y_ctx`` as the in-context set and evaluates
    predictions on ``X_qry``/``y_qry``.
    """
    model.eval()

    X_ep    = np.concatenate([X_ctx, X_qry], axis=0)
    X_t     = torch.from_numpy(X_ep).float().unsqueeze(0).to(device)
    y_ctx_t = torch.from_numpy(y_ctx.astype(np.float32)).unsqueeze(0).to(device)

    logits = model(X=X_t, y_train=y_ctx_t, return_logits=True)
    logits = logits.squeeze(0)  # (qry_size, num_classes)

    y_true = torch.from_numpy(y_qry.astype(np.int64)).to(device)
    loss   = F.cross_entropy(logits, y_true).item()

    preds = logits.argmax(dim=-1).cpu().numpy()
    acc   = accuracy_score(y_qry, preds)
    print("Classification report:\n", classification_report(y_qry, preds))
    return loss, acc
