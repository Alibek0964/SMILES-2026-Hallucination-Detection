"""
splitting.py — 5-fold stratified cross-validation.

A single random split on 689 samples gives noisy metrics — the test slice is
roughly 100 samples, and one or two unlucky labels can swing AUROC by
several percentage points. We therefore use 5-fold StratifiedKFold so that
every sample appears in exactly one test fold and the reported numbers are
averaged across all five folds.

Within each fold:
  * 80% of the data is the "train + validation" pool, of which 1/8 (= 10% of
    the full dataset) becomes the validation slice used by
    ``HallucinationProbe.fit_hyperparameters`` for threshold tuning.
  * 20% is the held-out test slice.

All splits preserve the class ratio via stratification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


N_FOLDS = 5
RANDOM_STATE = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Stratified 5-fold split with a held-out validation slice per fold.

    The ``test_size`` / ``val_size`` arguments are kept in the signature for
    backward compatibility but are ignored — fold sizes are determined by
    ``N_FOLDS`` instead.

    Returns:
        A list of ``N_FOLDS`` tuples ``(idx_train, idx_val, idx_test)``.
    """
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros_like(y), y)):
        # Carve a validation slice out of train_val_idx, stratified by label.
        # 0.125 of 80% ≈ 10% of the full dataset.
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.125,
                random_state=random_state + fold_idx,
                stratify=y[train_val_idx],
            )
        except ValueError:
            # Fallback if a class is too rare to stratify within the fold.
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.125,
                random_state=random_state + fold_idx,
            )
        splits.append((train_idx, val_idx, test_idx))

    return splits
