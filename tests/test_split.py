"""The split must never leak a system across sides."""
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpvet.bench import split as S

@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    rows = []
    for tic in range(1000, 1120):
        for _ in range(int(rng.integers(1, 40))):
            rows.append({"tic": tic, "label": int(rng.random() < 0.3),
                         "n_cycles_raw": int(rng.integers(0, 30))})
    ev = pd.DataFrame(rows)
    raw = pd.DataFrame({"tess_id": range(1000, 1120),
                        "period": rng.uniform(5, 40, 120),
                        "Tmag": rng.uniform(8, 15, 120)})
    return ev, raw

def test_no_tic_appears_in_two_splits(toy):
    ev, raw = toy
    tab = S.build_tic_table(ev, raw)
    asg, _ = S.assign(tab)
    sets = {s: {t for t, v in asg.items() if v == s} for s in S.TARGETS}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])

def test_every_event_matches_its_tic(toy):
    ev, raw = toy
    tab = S.build_tic_table(ev, raw)
    asg, _ = S.assign(tab)
    out, rep = S.report(ev, asg)
    assert out.split.notna().all()
    assert (out.split == out.tic.map(asg)).all()

def test_excluded_tics_are_forced_to_test(toy):
    """The real-planet hosts are the test set; they may never train."""
    ev, raw = toy
    tab = S.build_tic_table(ev, raw)
    excl = [1000, 1001, 1002]
    asg, _ = S.assign(tab, excluded_tics=excl)
    for t in excl:
        assert asg[t] == "test"
    S.verify(S.apply_split(ev, asg), asg, excluded_tics=excl)

def test_verify_catches_a_deliberately_corrupted_split(toy):
    """A guard that cannot fail is not a guard."""
    ev, raw = toy
    tab = S.build_tic_table(ev, raw)
    asg, _ = S.assign(tab)
    out = S.apply_split(ev, asg)
    out.loc[out.index[0], "split"] = "train" if out.split.iloc[0] != "train" else "test"
    with pytest.raises(AssertionError, match="disagree with their TIC"):
        S.verify(out, asg)

def test_excluded_tic_in_train_is_rejected(toy):
    ev, raw = toy
    tab = S.build_tic_table(ev, raw)
    asg, _ = S.assign(tab)
    asg[1000] = "train"
    with pytest.raises(AssertionError, match="must be test"):
        S.verify(S.apply_split(ev, asg), asg, excluded_tics=[1000])

def test_high_coverage_tier_over_allocates_to_test(toy):
    """The 10-plus tier buys test power where the MDE is set."""
    assert S.TARGETS_HIGH_COVERAGE["test"] > S.TARGETS["test"]
    assert S.coverage_tier(2) == "lt3"
    assert S.coverage_tier(5) == "3to9"
    assert S.coverage_tier(10) == "10plus"
