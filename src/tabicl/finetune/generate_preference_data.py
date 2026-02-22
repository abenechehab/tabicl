"""
generate_preference_data.py

Utilities to generate (x, positive_label, negative_label)
triplets for DPO-style training in classification problems.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Base Interface
# ============================================================

class PreferenceGenerator(ABC):
    """
    Base interface for generating preference pairs.
    """

    @abstractmethod
    def generate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            X_pref        (N, ...)
            pos_labels    (N,)
            neg_labels    (N,)
        """
        pass


# ============================================================
# 1️⃣ Random Negative Class
# ============================================================

class RandomNegativeGenerator(PreferenceGenerator):
    """
    For each example, sample a random incorrect class.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def generate(self, X, y, model=None):
        N = y.shape[0]

        neg_labels = torch.randint(
            low=0,
            high=self.num_classes,
            size=(N,),
            device=y.device,
        )

        # Ensure negatives differ from positives
        mask = neg_labels == y
        while mask.any():
            neg_labels[mask] = torch.randint(
                0, self.num_classes, (mask.sum(),), device=y.device
            )
            mask = neg_labels == y

        return X, y, neg_labels


# ============================================================
# 2️⃣ Majority Class as Negative
# ============================================================

class MajorityNegativeGenerator(PreferenceGenerator):
    """
    Always use majority class as negative (if not already).
    Useful for imbalance experiments.
    """

    def __init__(self, y_train: torch.Tensor):
        values, counts = torch.unique(y_train, return_counts=True)
        self.majority_class = values[counts.argmax()].item()

    def generate(self, X, y, model=None):
        neg_labels = torch.full_like(y, self.majority_class)

        # If sample already majority class, pick random other class
        mask = neg_labels == y
        if mask.any():
            unique_classes = torch.unique(y)
            alt_class = unique_classes[unique_classes != self.majority_class][0]
            neg_labels[mask] = alt_class

        return X, y, neg_labels


# ============================================================
# 3️⃣ Hard Negative Mining
# ============================================================

class HardNegativeMiningGenerator(PreferenceGenerator):
    """
    For each example:
        negative = highest-probability incorrect class
    """

    def generate(self, X, y, model):
        model.eval()
        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=-1)

        # Zero out correct class probabilities
        probs.scatter_(1, y.unsqueeze(1), 0.0)

        neg_labels = probs.argmax(dim=1)

        return X, y, neg_labels


# ============================================================
# 4️⃣ Confusion Set Pairing
# ============================================================

class ConfusionSetGenerator(PreferenceGenerator):
    """
    Use confusion matrix to identify most confused class pairs.

    negative = most frequently confused class for this true label
    """

    def __init__(self, confusion_matrix: np.ndarray):
        """
        confusion_matrix: shape (K, K)
        rows = true labels
        cols = predicted labels
        """
        self.confusion_matrix = confusion_matrix

        # For each true class, find most confused incorrect class
        K = confusion_matrix.shape[0]
        self.confused_map = {}

        for true_class in range(K):
            row = confusion_matrix[true_class].copy()
            row[true_class] = 0  # ignore correct predictions
            self.confused_map[true_class] = row.argmax()

    def generate(self, X, y, model=None):
        neg_labels = torch.tensor(
            [self.confused_map[int(label)] for label in y],
            device=y.device,
        )

        return X, y, neg_labels


# ============================================================
# Utility: Build Confusion Matrix
# ============================================================

def compute_confusion_matrix(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
) -> np.ndarray:

    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)

    conf = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y.cpu().numpy(), preds.cpu().numpy()):
        conf[true, pred] += 1

    return conf