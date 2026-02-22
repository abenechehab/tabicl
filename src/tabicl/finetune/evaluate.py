"""Evaluate TabICL v2 on the Iris dataset using the raw PyTorch interface.

This script demonstrates how to:
  1. Load the TabICL v2 pretrained checkpoint directly as a torch.nn.Module.
  2. Prepare a tabular classification dataset (Iris) as tensors.
  3. Run inference through the model's forward() method (no sklearn wrapper).
  4. Measure test accuracy.

The model expects:
  - X       : (B, T, H)  — B tables, T = train_size + test_size rows, H features
  - y_train : (B, train_size) — integer class labels (0-indexed) as float
The first `train_size` rows of X are the in-context training examples; the
remaining rows are the test samples whose labels the model must predict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from tabicl import TabICL


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

REPO_ID = "jingang/TabICL"
CHECKPOINT_V2 = "tabicl-classifier-v2-20260212.ckpt"


def load_tabicl_v2(checkpoint_version: str = CHECKPOINT_V2) -> TabICL:
    """Download (if needed) and load the TabICL v2 checkpoint.

    Parameters
    ----------
    checkpoint_version:
        Filename of the checkpoint on the HuggingFace Hub.

    Returns
    -------
    TabICL
        Model loaded with pretrained weights, set to eval mode on CPU.
    """
    try:
        ckpt_path = Path(
            hf_hub_download(repo_id=REPO_ID, filename=checkpoint_version, local_files_only=True)
        )
    except LocalEntryNotFoundError:
        print(f"Checkpoint '{checkpoint_version}' not cached locally.")
        print(f"Downloading from Hugging Face Hub ({REPO_ID}) …")
        ckpt_path = Path(hf_hub_download(repo_id=REPO_ID, filename=checkpoint_version))

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    assert "config" in checkpoint, "Checkpoint is missing the 'config' key."
    assert "state_dict" in checkpoint, "Checkpoint is missing the 'state_dict' key."

    print(f"Model config: {checkpoint['config']}")

    model = TabICL(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    iris = load_iris()
    X_raw, y_raw = iris.data.astype(np.float32), iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(
        f"Iris — train: {X_train.shape[0]} samples, "
        f"test: {X_test.shape[0]} samples, "
        f"features: {X_train.shape[1]}, "
        f"classes: {np.unique(y_raw)}"
    )

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # LabelEncoder ensures labels are 0-indexed contiguous integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.float32)
    y_test_enc = le.transform(y_test)

    # ------------------------------------------------------------------
    # 3. Build tensors for the PyTorch interface
    #    X  : (1, train_size + test_size, n_features)
    #    y_train : (1, train_size)
    # ------------------------------------------------------------------
    X_combined = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
    X_tensor = torch.from_numpy(X_combined).unsqueeze(0).to(device)          # (1, T, H)
    y_train_tensor = torch.from_numpy(y_train_enc).unsqueeze(0).to(device)   # (1, train_size)

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    model = load_tabicl_v2()
    model = model.to(device)

    # ------------------------------------------------------------------
    # 5. Inference
    #    In eval mode, forward() calls _inference_forward() which returns:
    #      logits  : (1, test_size, num_classes)  when return_logits=True
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits = model(
            X=X_tensor,
            y_train=y_train_tensor,
            return_logits=True,
            softmax_temperature=0.9,
        )

    # logits: (1, test_size, num_classes) → (test_size, num_classes)
    logits_np = logits.squeeze(0).cpu().numpy()
    y_pred_enc = np.argmax(logits_np, axis=-1)
    y_pred = le.inverse_transform(y_pred_enc)

    # ------------------------------------------------------------------
    # 6. Results
    # ------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy on Iris: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Predicted labels : {y_pred}")
    print(f"Ground-truth     : {y_test}")


if __name__ == "__main__":
    main()
