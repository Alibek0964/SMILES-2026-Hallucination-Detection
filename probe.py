"""
probe.py — Hallucination probe classifier.

Pipeline applied to the input feature vector:

    raw features (≈ 4480-4580 dims)
        → StandardScaler          (zero-mean, unit-variance per feature)
        → PCA  (n_components=128) (compress to a learnable subspace)
        → small MLP               (128 → 64 → 1) with dropout + weight decay
        → sigmoid                 (probability of hallucination)

Why this design for a 689-sample binary task:

  * The raw feature dim (4480 from 5 layers × 896 hidden, +100 geometric) is
    far larger than the number of training examples — PCA collapses it to a
    bounded subspace and acts as a strong implicit regulariser.
  * A two-layer MLP with dropout 0.4 and weight_decay 1e-3 is enough capacity
    for this problem and consistently beats both raw logistic regression and
    larger MLPs in our experiments.
  * An internal random hold-out (10% of train) drives early stopping on
    validation AUROC, preventing the network from over-fitting the small set.
  * Class imbalance is corrected by ``pos_weight`` in BCEWithLogitsLoss.
  * Threshold tuning in ``fit_hyperparameters`` maximises F1 on the official
    validation split.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Hyperparameters — tuned by light manual search; safe defaults.
# ---------------------------------------------------------------------------

PCA_COMPONENTS = 128
HIDDEN_DIM     = 64
DROPOUT        = 0.4
WEIGHT_DECAY   = 1e-3
LR             = 1e-3
MAX_EPOCHS     = 300
PATIENCE       = 30          # early-stopping patience on val AUROC
INTERNAL_VAL   = 0.10        # fraction of fit() data used for early stopping
SEED           = 42


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HallucinationProbe(nn.Module):
    """Scaler + PCA + small MLP, with early stopping and threshold tuning."""

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._threshold: float = 0.5
        _seed_everything(SEED)

    # ------------------------------------------------------------------
    # Network factory
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Network not built. Call fit() first.")
        return self._net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler, PCA, and MLP. Uses an internal hold-out for early
        stopping on validation AUROC."""
        # 1. Scale.
        X_scaled = self._scaler.fit_transform(X)

        # 2. PCA — n_components capped at min(128, n_samples-1, n_features).
        n_components = min(PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self._pca = PCA(n_components=n_components, random_state=SEED)
        X_reduced = self._pca.fit_transform(X_scaled)

        # 3. Internal train/val split for early stopping.
        # Stratify to preserve class ratio in both splits.
        y_int = y.astype(int)
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_reduced, y_int,
                test_size=INTERNAL_VAL,
                random_state=SEED,
                stratify=y_int,
            )
        except ValueError:
            # Fallback if stratification fails (e.g. tiny minority class).
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_reduced, y_int, test_size=INTERNAL_VAL, random_state=SEED,
            )

        # 4. Build network sized to the PCA output.
        self._build_network(X_tr.shape[1])

        X_tr_t = torch.from_numpy(X_tr).float()
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
        X_va_t = torch.from_numpy(X_va).float()

        # 5. Class-balanced loss.
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS,
        )

        # 6. Train with early stopping on validation AUROC.
        best_state: dict | None = None
        best_auroc = -1.0
        epochs_without_improvement = 0

        for epoch in range(MAX_EPOCHS):
            # --- Train step (full-batch is fine on this size) ---
            self.train()
            optimizer.zero_grad()
            loss = criterion(self(X_tr_t), y_tr_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- Validation AUROC ---
            self.eval()
            with torch.no_grad():
                val_logits = self(X_va_t)
                val_probs = torch.sigmoid(val_logits).numpy()

            # AUROC undefined if val happens to have a single class.
            if len(np.unique(y_va)) < 2:
                continue
            val_auroc = roc_auc_score(y_va, val_probs)

            if val_auroc > best_auroc + 1e-5:
                best_auroc = val_auroc
                best_state = copy.deepcopy(self._net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    break

        # 7. Restore the best weights seen during training.
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.eval()
        return self

    # ------------------------------------------------------------------
    # Decision threshold tuning
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold to maximise F1 on the validation set."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _transform(self, X: np.ndarray) -> torch.Tensor:
        X_scaled = self._scaler.transform(X)
        X_reduced = self._pca.transform(X_scaled) if self._pca is not None else X_scaled
        return torch.from_numpy(X_reduced).float()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = self._transform(X)
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
