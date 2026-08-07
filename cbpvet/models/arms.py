"""The feature-model arms, trained ONLY on the frozen shard.

The arm contract, enforced not promised
---------------------------------------
Every arm's fitted feature list must equal exactly what the schema offers
(``training_scalars``, minus the columns an ablation arm deliberately masks).
The freeze manifest requires this as a test because the audit columns
(injection truth, split, bookkeeping) sit in the same HDF5 group as the
features, and one loose ``df[cols]`` away from a leak. ``load_matrix`` is the
single gate every arm loads through.

Invalidity handling, as pinned in the schema
--------------------------------------------
XGBoost receives NaN and routes it natively. Logistic regression and random
forest receive median-imputed values PLUS the validity bit as its own feature
(the bit is information: "this quantity is undefined here" is a fact the model
may use; a silently imputed number is a lie it cannot detect).

Rebalance and split discipline
------------------------------
Rows are filtered to ``rebalance_keep == 1`` (probe 1's carried remedy) and the
frozen ``split`` column is the only split ever used. Selection is on validation
PR-AUC only, per the ratified pin; the test split is untouched until T0.
"""

import json
import logging

import h5py
import numpy as np
import pandas as pd

from ..export import schema
from ..export.incumbent import INCUMBENT_COLS

logger = logging.getLogger("cbpvet.models")

SEED = 20260806

# Arms over the SCALAR feature space (the CNN arms went to fall under D-A).
# a  = local view only -> not a feature model; handled by the CNN branch (fall).
# b0 = the incumbent's own 17 columns, nothing else.
# b  = b0 + the core + host scalars (everything training_scalars offers).
# c'' variant for feature models = b minus the gated pair scalars.
ARM_FEATURES = {
    "b0": list(INCUMBENT_COLS),
    "b": None,          # None = full training_scalars
    "b_nopair": "mask_pair",
}


def training_columns(arm):
    full = schema.training_scalars(INCUMBENT_COLS)
    spec = ARM_FEATURES[arm]
    if spec is None:
        return full
    if spec == "mask_pair":
        return [c for c in full if c not in schema.GATED_PAIR_SCALARS]
    return list(spec)


def load_matrix(shard_path, arm, splits=("train", "val")):
    """The single feature gate. Returns X, y, groups, valid-bit matrix, meta."""
    with h5py.File(shard_path, "r") as f:
        cols = {k: f["scalars"][k][:] for k in f["scalars"].keys()}
    df = pd.DataFrame({k: v for k, v in cols.items() if v.ndim == 1})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    df = df[df.rebalance_keep == 1]
    df = df[df["split"].astype(str).isin(splits)]

    feats = training_columns(arm)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"arm {arm}: shard lacks {missing}")
    forbidden = set(feats) & (set(schema.AUDIT_COLUMNS) | set(schema.WITHHELD_SCALARS))
    if forbidden:
        raise AssertionError(f"arm {arm}: forbidden columns {forbidden}")

    X = df[feats].to_numpy(dtype=float)
    vbits = np.column_stack([
        df[f"valid_{c}"].to_numpy(dtype=float) if f"valid_{c}" in df.columns
        else np.isfinite(df[c].to_numpy(dtype=float)).astype(float)
        for c in feats])
    y = df.label.to_numpy().astype(int)
    groups = df.tic.to_numpy().astype(int)
    sp = df["split"].astype(str).to_numpy()
    return X, y, groups, vbits, {"features": feats, "split": sp, "n": len(df)}


def score_matrix(df, arm):
    """The single feature gate for SCORING (deployment, one-shot).

    Same forbidden-column contract as ``load_matrix``, but takes an in-memory
    DataFrame and applies NO split or rebalance filtering: scoring targets
    (the deployment queue, the sealed real-planet shards) have no split and
    every row must be scored. Added 2026-08-07 for the deployment run and the
    one-shot so neither builds its own ad-hoc feature path.
    """
    feats = training_columns(arm)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"arm {arm}: scoring frame lacks {missing}")
    forbidden = set(feats) & (set(schema.AUDIT_COLUMNS) | set(schema.WITHHELD_SCALARS))
    if forbidden:
        raise AssertionError(f"arm {arm}: forbidden columns {forbidden}")
    X = df[feats].to_numpy(dtype=float)
    return X, feats


def impute_for_dense(X, vbits, medians=None):
    """Median-impute + append validity bits, for LR/RF. Fit medians on train."""
    Xi = X.copy()
    if medians is None:
        medians = np.nanmedian(Xi, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    for j in range(Xi.shape[1]):
        bad = ~np.isfinite(Xi[:, j])
        Xi[bad, j] = medians[j]
    return np.hstack([Xi, vbits]), medians
