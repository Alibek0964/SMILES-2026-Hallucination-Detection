"""
splitting.py — train / validation / test splits.

The dataset has 689 rows, all prompts are unique (no duplicates), and the
test.csv prompts do not overlap with the training set, so a group-aware
split reduces to plain stratification.  The block below still detects a
usable group column at runtime so the strategy stays correct if the
dataset is regenerated with overlapping prompts later.

Strategy:
    * 5-fold stratified K-fold over the label column — every sample appears
      in exactly one test slice and probe metrics are averaged across folds.
    * Inside each fold, a stratified validation slice is carved out of the
      training pool and used by `HallucinationProbe.fit_hyperparameters` to
      pick an accuracy-optimal decision threshold.
    * If the dataset ever contains a group column with repeated values
      (`prompt`, `question`, `id`, `source`, `group`), the splitter switches
      to GroupKFold to keep duplicated prompts in the same fold.

All seeds are fixed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    train_test_split,
)


N_FOLDS = 5
RANDOM_STATE = 42
VAL_FRACTION_OF_TRAIN = 0.125  # 0.125 of 80 % ≈ 10 % of the full dataset

GROUP_CANDIDATE_COLUMNS = ("group", "source", "id", "question", "prompt")


def _detect_group_column(df: pd.DataFrame | None) -> str | None:
    """Return the first column with at least one repeated value, else None."""
    if df is None:
        return None
    for col in GROUP_CANDIDATE_COLUMNS:
        if col in df.columns and df[col].nunique() < len(df):
            return col
    return None


def _stratified_val(
    train_val_idx: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a stratified validation slice from `train_val_idx`."""
    try:
        return train_test_split(
            train_val_idx,
            test_size=VAL_FRACTION_OF_TRAIN,
            random_state=seed,
            stratify=y[train_val_idx],
        )
    except ValueError:
        return train_test_split(
            train_val_idx,
            test_size=VAL_FRACTION_OF_TRAIN,
            random_state=seed,
        )


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,        # kept for backward compatibility
    val_size: float = 0.15,         # kept for backward compatibility
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Build (train, val, test) index tuples for evaluation.

    Args:
        y:            label array of shape (N,) with values in {0, 1}.
        df:           optional dataframe carrying potential group columns.
        random_state: RNG seed.

    Returns:
        A list of N_FOLDS (idx_train, idx_val, idx_test) tuples.  idx_val is
        always populated so HallucinationProbe.fit_hyperparameters can run.
    """
    y_int = np.asarray(y).astype(int)
    n = len(y_int)

    group_col = _detect_group_column(df)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    if group_col is not None:
        groups = df[group_col].astype("category").cat.codes.to_numpy()
        gkf = GroupKFold(n_splits=N_FOLDS)
        for fold_idx, (train_val_idx, test_idx) in enumerate(
            gkf.split(np.zeros(n), y_int, groups=groups),
        ):
            seed = random_state + fold_idx
            train_idx, val_idx = _stratified_val(train_val_idx, y_int, seed)
            splits.append((train_idx, val_idx, test_idx))
        return splits

    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=random_state,
    )
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        skf.split(np.zeros(n), y_int),
    ):
        seed = random_state + fold_idx
        train_idx, val_idx = _stratified_val(train_val_idx, y_int, seed)
        splits.append((train_idx, val_idx, test_idx))

    return splits


__all__ = ["split_data", "N_FOLDS", "RANDOM_STATE"]
