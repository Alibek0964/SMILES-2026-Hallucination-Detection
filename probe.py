"""
probe.py — v15 = 3-base meta-stack:
    Base 1: MLP on all 9020 features
    Base 2: XGBoost on the 60 tabular features (geom + NLI + consistency + lex)
    Base 3: LogReg(L2) on PCA(128) of the 8960-dim hidden-state pool
    Meta:   LogReg(L2, C=1.0) over the 3 OOF probability streams

Why we added Base 3 in v15:
    v14's meta-LogReg consistently weighted XGB ≈ 2-3× more than the MLP,
    revealing that MLP on hidden states was mostly memorising (train AUROC
    99.9 %).  A LINEAR probe on a low-rank projection of the same hidden
    states cannot memorise — its capacity is bounded.  If the hidden-state
    pool carries any generalisable truthfulness signal at all, it shows
    up here, in a form the meta-LogReg can mix with XGB without inheriting
    MLP's overfit.

XGBoost is also slightly looser in v15:
    n_estimators 300 → 500
    reg_alpha   2.0 → 1.0
    reg_lambda  2.0 → 1.0
    With only 60 input features, v14's deep-regularisation profile was
    over-shrinking — we now let trees find more interactions before
    L1/L2 kicks in.

Iteration budget:
    Feature cache from v13 makes extraction trivial.  v15 first run is
    ~3 minutes end-to-end (cache load + 3-base training × 5 folds).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEED = 42
N_BAGS = 5

# MLP
HIDDEN1 = 256
HIDDEN2 = 64
DROPOUT = 0.40
LR_MLP = 8e-4
WEIGHT_DECAY = 5e-2
EPOCHS = 40
BATCH_SIZE = 64

# Hidden / tabular partition
HIDDEN_DIM = 8960
EXPECTED_TOTAL = 9020
TABULAR_DIM = EXPECTED_TOTAL - HIDDEN_DIM  # 60

# PCA for the linear-probe base
PCA_N = 128
LR_C_PCA = 0.5   # mild L2 — linear probe at p=128, n_train≈386

# XGBoost (relaxed from v14)
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 2
XGB_LEARNING_RATE = 0.03
XGB_MIN_CHILD_WEIGHT = 5
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.7
XGB_REG_ALPHA = 1.0
XGB_REG_LAMBDA = 1.0


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
# Base 1: MLP on all features
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, HIDDEN1)
        self.norm1 = nn.LayerNorm(HIDDEN1)
        self.fc1 = nn.Linear(HIDDEN1, HIDDEN2)
        self.norm2 = nn.LayerNorm(HIDDEN2)
        self.head = nn.Linear(HIDDEN2, 1)
        self.dropout = nn.Dropout(DROPOUT)

        for m in (self.proj, self.fc1, self.head):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.gelu(self.norm1(self.proj(x))))
        h = self.dropout(F.gelu(self.norm2(self.fc1(h))))
        return self.head(h).squeeze(-1)


def _train_mlp(
    X_scaled: np.ndarray, y: np.ndarray, seed: int, device: torch.device,
) -> _MLP:
    _set_seed(seed)
    X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    net = _MLP(X_scaled.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR_MLP, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    n = X_t.shape[0]
    idx_all = np.arange(n)

    net.train()
    for _ in range(EPOCHS):
        np.random.shuffle(idx_all)
        for start in range(0, n, BATCH_SIZE):
            batch = idx_all[start:start + BATCH_SIZE]
            xb, yb = X_t[batch], y_t[batch]
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


def _mlp_proba(net: _MLP, X_scaled: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xb = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
        logits = net(xb).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------
# Base 2: XGBoost on tabular
# ---------------------------------------------------------------------------

def _make_xgb(seed: int, scale_pos_weight: float):
    if not _HAS_XGB:
        return None
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Probe — 3-base meta-stack
# ---------------------------------------------------------------------------

class HallucinationProbe(nn.Module):
    """3-base meta-stack: MLP(all) ⊕ XGB(tabular) ⊕ LR+PCA(hidden)."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.seed = SEED
        self.n_bags = N_BAGS
        self._device = _pick_device()

        self._scaler_all: StandardScaler | None = None
        self._scaler_tab: StandardScaler | None = None
        self._scaler_hid: StandardScaler | None = None
        self._pca: PCA | None = None

        self._mlps: list[_MLP] = []
        self._xgbs: list = []
        self._lrs: list[LogisticRegression] = []

        self._meta: LogisticRegression | None = None
        self._threshold: float = 0.5
        self._has_tabular: bool = False

    # ----- helpers --------------------------------------------------------

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Returns (X_full, X_hidden_or_None, X_tabular_or_None)."""
        if X.shape[1] == EXPECTED_TOTAL:
            return X, X[:, :HIDDEN_DIM], X[:, HIDDEN_DIM:]
        return X, None, None

    # ----- training -------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        _set_seed(self.seed)
        y = y.astype(int)
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        scale_pos_weight = max(n_neg, 1.0) / max(n_pos, 1.0)

        X_full, X_hid, X_tab = self._split(X)
        self._has_tabular = X_tab is not None and _HAS_XGB

        # --- Preprocessors ---
        self._scaler_all = StandardScaler()
        X_full_s = self._scaler_all.fit_transform(X_full)

        if X_hid is not None:
            self._scaler_hid = StandardScaler()
            X_hid_s = self._scaler_hid.fit_transform(X_hid)
            n_components = min(PCA_N, X_hid_s.shape[1], X_hid_s.shape[0] - 1)
            self._pca = PCA(n_components=n_components, random_state=self.seed)
            X_hid_p = self._pca.fit_transform(X_hid_s)
        else:
            X_hid_p = None

        if self._has_tabular:
            self._scaler_tab = StandardScaler()
            X_tab_s = self._scaler_tab.fit_transform(X_tab)
        else:
            X_tab_s = None

        # --- 5-fold OOF for all 3 bases ---
        skf = StratifiedKFold(
            n_splits=self.n_bags, shuffle=True, random_state=self.seed,
        )

        self._mlps, self._xgbs, self._lrs = [], [], []
        oof_mlp = np.zeros(len(y), dtype=np.float64)
        oof_xgb = np.zeros(len(y), dtype=np.float64)
        oof_lr = np.zeros(len(y), dtype=np.float64)

        for fold, (tr, va) in enumerate(skf.split(X_full_s, y)):
            seed_f = self.seed + fold

            # Base 1: MLP on all features
            mlp = _train_mlp(X_full_s[tr], y[tr], seed_f, self._device)
            oof_mlp[va] = _mlp_proba(mlp, X_full_s[va], self._device)
            self._mlps.append(mlp)

            # Base 2: XGBoost on tabular
            if self._has_tabular:
                xgb = _make_xgb(seed_f, scale_pos_weight)
                xgb.fit(X_tab_s[tr], y[tr])
                oof_xgb[va] = xgb.predict_proba(X_tab_s[va])[:, 1]
                self._xgbs.append(xgb)

            # Base 3: LogReg on PCA-hidden
            if X_hid_p is not None:
                lr = LogisticRegression(
                    C=LR_C_PCA, penalty="l2", max_iter=3000,
                    random_state=seed_f, class_weight="balanced",
                    solver="liblinear",
                )
                lr.fit(X_hid_p[tr], y[tr])
                oof_lr[va] = lr.predict_proba(X_hid_p[va])[:, 1]
                self._lrs.append(lr)

        # --- meta-learner ---
        meta_streams: list[np.ndarray] = [oof_mlp]
        names = ["MLP"]
        if self._has_tabular:
            meta_streams.append(oof_xgb)
            names.append("XGB")
        if X_hid_p is not None:
            meta_streams.append(oof_lr)
            names.append("LR")

        meta_X = np.column_stack(meta_streams)
        self._meta = LogisticRegression(
            C=1.0, penalty="l2", max_iter=2000,
            random_state=self.seed, solver="liblinear",
        )
        self._meta.fit(meta_X, y)
        oof_final = self._meta.predict_proba(meta_X)[:, 1]
        self._threshold = _tune_threshold(oof_final, y)

        coefs = self._meta.coef_[0]
        weights_str = "  ".join(
            f"{n}={c:+.3f}" for n, c in zip(names, coefs)
        )
        print(
            f"[v15] meta weights: {weights_str}  "
            f"intercept={self._meta.intercept_[0]:+.3f}  "
            f"threshold={self._threshold:.3f}"
        )

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray,
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _tune_threshold(probs, y_val.astype(int))
        return self

    # ----- inference ------------------------------------------------------

    def _predict_streams(self, X: np.ndarray) -> np.ndarray:
        """Run all 3 base ensembles and return shape (n, n_streams) of probs."""
        X_full, X_hid, X_tab = self._split(X)
        X_full_s = self._scaler_all.transform(X_full)

        mlp_p = np.mean(
            [_mlp_proba(m, X_full_s, self._device) for m in self._mlps],
            axis=0,
        )
        streams = [mlp_p]

        if self._has_tabular and self._xgbs and X_tab is not None:
            X_tab_s = self._scaler_tab.transform(X_tab)
            xgb_p = np.mean(
                [m.predict_proba(X_tab_s)[:, 1] for m in self._xgbs],
                axis=0,
            )
            streams.append(xgb_p)

        if self._lrs and X_hid is not None and self._pca is not None:
            X_hid_p = self._pca.transform(self._scaler_hid.transform(X_hid))
            lr_p = np.mean(
                [m.predict_proba(X_hid_p)[:, 1] for m in self._lrs],
                axis=0,
            )
            streams.append(lr_p)

        return np.column_stack(streams)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._scaler_all is None or not self._mlps or self._meta is None:
            raise RuntimeError("Probe not fitted; call fit() first.")
        meta_X = self._predict_streams(X)
        probs = self._meta.predict_proba(meta_X)[:, 1]
        return np.stack([1.0 - probs, probs], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def forward(self, *_a, **_k):
        raise NotImplementedError(
            "HallucinationProbe wraps an MLP+XGB+LR meta-stack; "
            "call predict_proba() / predict() instead."
        )
