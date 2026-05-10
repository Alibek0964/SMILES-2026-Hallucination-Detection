"""
probe.py — lightweight binary MLP probe over Qwen2.5-0.5B hidden-state features.

Pipeline (v9, the shipped configuration):

    raw features  →  StandardScaler  →  5x bagged MLP ensemble
                                     →  out-of-fold threshold tuning
                                     →  final binary prediction

The MLP architecture matches the SMILES-2026 spec verbatim:

    Linear(input_dim → 256) → LayerNorm → GELU → Dropout(p)
    Linear(256 → 64)        → LayerNorm → GELU → Dropout(p)
    Linear(64  → 1)          (raw logit)

Why bagging + out-of-fold threshold rather than a single fit:
    The training set is small (689 rows) and the feature dim is wide
    (5 K – 7 K).  A single MLP overfits aggressively — train AUROC
    saturates at 1.00 while validation AUROC sits ~25 pp lower.  Bagging
    five MLPs trained on different 80 % stratified slices and averaging
    their probabilities cancels out per-fold overfit noise.  The
    decision threshold is then tuned on the full out-of-fold (OOF)
    probability vector — i.e. on probabilities that no single model has
    seen during training — so the calibration generalises to the
    unseen `test.csv` rows that `predictions.csv` is built from.

Training:
    * `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` to counter the
      70/30 class skew in `dataset.csv`.
    * `AdamW`, `lr = 8e-4`, `weight_decay = 5e-2`, `batch_size = 64`,
      `epochs = 40`, cosine LR schedule, gradient clipping at 1.0.
    * Higher dropout (0.40) and weight-decay than v8 because v8 saw
      train AUROC pinned at 1.00.

Threshold calibration:
    * `fit()` tunes the threshold on OOF predictions for accuracy
      (the contest's primary metric).
    * `fit_hyperparameters(X_val, y_val)` re-tunes on a per-fold
      external validation slice when called by `evaluate.py`.

Reproducibility: SEED seeds Python, NumPy and PyTorch RNGs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEED = 42

# MLP topology — matches the SMILES-2026 brief: input → 256 → 64 → 1.
HIDDEN1 = 256
HIDDEN2 = 64
DROPOUT = 0.40

# Optimisation.
LR = 8e-4
WEIGHT_DECAY = 5e-2
EPOCHS = 40
BATCH_SIZE = 64

# Bagging — number of MLPs averaged inside .predict_proba.
N_BAGS = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tune_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    """Threshold in [0, 1] that maximises accuracy on (probs, y).

    Sweeps both the raw probability values and a uniform 401-step grid so
    the optimum sits exactly on a probability boundary.
    """
    candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 401)]))
    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        pred = (probs >= t).astype(int)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = HIDDEN1,
                 hidden2: int = HIDDEN2, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden1)
        self.norm1 = nn.LayerNorm(hidden1)
        self.fc1 = nn.Linear(hidden1, hidden2)
        self.norm2 = nn.LayerNorm(hidden2)
        self.head = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)

        for m in (self.proj, self.fc1, self.head):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.gelu(self.norm1(self.proj(x))))
        h = self.dropout(F.gelu(self.norm2(self.fc1(h))))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class HallucinationProbe(nn.Module):
    """5x bagged MLP ensemble + out-of-fold threshold tuning."""

    def __init__(
        self,
        hidden1: int = HIDDEN1,
        hidden2: int = HIDDEN2,
        dropout: float = DROPOUT,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        n_bags: int = N_BAGS,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_bags = n_bags
        self.seed = seed

        self._scaler: StandardScaler | None = None
        self._nets: list[_MLP] = []
        self._device = _pick_device()
        self._threshold: float = 0.5

    # ----- training -------------------------------------------------------

    def _train_one_mlp(self, X_scaled: np.ndarray, y: np.ndarray, seed: int) -> _MLP:
        _set_seed(seed)

        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self._device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self._device)

        net = _MLP(X_scaled.shape[1], self.hidden1, self.hidden2,
                   self.dropout).to(self._device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self._device)

        n = X_t.shape[0]
        idx_all = np.arange(n)

        net.train()
        for _ in range(self.epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.batch_size):
                batch = idx_all[start:start + self.batch_size]
                xb = X_t[batch]
                yb = y_t[batch]
                logits = net(xb)
                loss = F.binary_cross_entropy_with_logits(
                    logits, yb, pos_weight=pos_weight,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
        net.eval()
        return net

    @staticmethod
    def _proba_one(net: _MLP, X_scaled: np.ndarray, device: torch.device) -> np.ndarray:
        with torch.no_grad():
            xb = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
            logits = net(xb).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """5-fold bag the MLPs, calibrate the threshold on OOF predictions.

        The OOF probability for a sample comes from the single MLP whose
        training fold did *not* contain that sample, so the threshold
        chosen on those probabilities reflects out-of-sample behaviour
        — closer to what we see on `test.csv`.
        """
        y = y.astype(int)
        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=self.n_bags, shuffle=True,
                              random_state=self.seed)
        self._nets = []
        oof_probs = np.zeros(len(y), dtype=np.float64)

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_s, y)):
            net = self._train_one_mlp(X_s[tr_idx], y[tr_idx],
                                      seed=self.seed + fold_idx)
            oof_probs[va_idx] = self._proba_one(net, X_s[va_idx], self._device)
            self._nets.append(net)

        self._threshold = _tune_threshold(oof_probs, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray,
    ) -> "HallucinationProbe":
        """Re-tune the decision threshold on an explicit validation set.

        Maximises accuracy (the primary ranking metric).  Called by
        `evaluate.py` per fold to override the OOF threshold with one
        chosen on the proper out-of-train validation slice.
        """
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _tune_threshold(probs, y_val.astype(int))
        return self

    # ----- inference ------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None or not self._nets:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        X_s = self._scaler.transform(X)
        probs = np.mean(
            [self._proba_one(net, X_s, self._device) for net in self._nets],
            axis=0,
        )
        return np.stack([1.0 - probs, probs], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def forward(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError(
            "HallucinationProbe wraps an internal _MLP ensemble; "
            "call predict_proba()."
        )
