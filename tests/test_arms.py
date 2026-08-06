"""The arm contract, as the freeze manifest requires.

MANIFEST_bench-v1.md marks this test REQUIRED once arms exist: every arm's
fitted feature list must equal exactly what the schema offers (minus an
ablation arm's deliberate mask), and no audit or withheld column may ever
reach a feature matrix. The audit columns sit in the same HDF5 group as the
features, one loose ``df[cols]`` away from a leak, so the contract is
enforced here rather than promised in prose.
"""

import numpy as np
import pytest

from cbpvet.export import schema
from cbpvet.export.incumbent import INCUMBENT_COLS
from cbpvet.models import arms


def test_arm_b_is_exactly_training_scalars():
    assert arms.training_columns("b") == schema.training_scalars(INCUMBENT_COLS)


def test_arm_b0_is_exactly_the_incumbent_block():
    assert arms.training_columns("b0") == list(INCUMBENT_COLS)


def test_arm_b_nopair_masks_exactly_the_gated_pair_scalars():
    full = schema.training_scalars(INCUMBENT_COLS)
    nopair = arms.training_columns("b_nopair")
    assert nopair == [c for c in full if c not in schema.GATED_PAIR_SCALARS]
    assert set(full) - set(nopair) == set(schema.GATED_PAIR_SCALARS)


def test_no_arm_offers_forbidden_columns():
    forbidden = set(schema.AUDIT_COLUMNS) | set(schema.WITHHELD_SCALARS)
    for arm in arms.ARM_FEATURES:
        assert not set(arms.training_columns(arm)) & forbidden, arm


def test_load_matrix_rejects_a_forbidden_column(tmp_path, monkeypatch):
    import h5py

    feats = schema.training_scalars(INCUMBENT_COLS)
    n = 8
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("scalars")
        for c in feats:
            g.create_dataset(c, data=np.random.default_rng(0).normal(size=n))
        g.create_dataset("label", data=np.array([0, 1] * (n // 2)))
        g.create_dataset("tic", data=np.arange(n))
        g.create_dataset("rebalance_keep", data=np.ones(n, dtype=int))
        g.create_dataset("split", data=np.array([b"train"] * n))
        g.create_dataset("truth_pair_role", data=np.zeros(n))

    # A sane arm loads.
    X, y, groups, vbits, meta = arms.load_matrix(str(path), "b", splits=("train",))
    assert X.shape == (n, len(feats)) and meta["features"] == feats

    # An arm smuggling an audit column must be refused by the single gate.
    monkeypatch.setitem(arms.ARM_FEATURES, "evil", feats + ["truth_pair_role"])
    with pytest.raises(AssertionError):
        arms.load_matrix(str(path), "evil", splits=("train",))
