"""Synthetic fixtures for the one-shot script. The ONLY way 26 runs before Aug 11.

Generates, deterministically (seed fixed), under data/oneshot/fixtures/:
- a tiny synthetic "frozen shard" (HDF5, scalars group) with two fake hosts
  whose 37 offered feature columns are drawn noise, plus tic/sector/t/t_dur/
  split/rebalance_keep/label bookkeeping;
- a synthetic times table with in-window flags designed so the expected D1
  matching is hand-computable (host A: 3 events, 2 within tolerance of known
  times, 1 not; host B the candidate: 1 event, 1 match);
- a fixture hash log + declared config pointing at the REAL frozen checkpoint
  (hash-asserting the real model is part of what reviewers must exercise) but
  entirely synthetic pools;
- fixture_expected.json: the hand-computed expectations (D1/D2 counts, and
  the pass counts computed by scoring the synthetic rows with the real model
  at the real tau - computed HERE once, so a reviewer's run must reproduce
  them exactly; any disagreement is a bug by the 3B.10-iv definition).

Nothing here touches real events, sealed files, or the frozen shard.
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cbpvet.export import schema
from cbpvet.export.incumbent import INCUMBENT_COLS
from cbpvet.models import arms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "data", "oneshot", "fixtures")
SEED = 20260811

# Synthetic hosts: fake TICs that collide with nothing real.
HOST_A = 900000001      # "planet" host, mission TESS
HOST_B = 900000002      # "candidate" host


def main():
    os.makedirs(FIX, exist_ok=True)
    rng = np.random.default_rng(SEED)
    feats = schema.training_scalars(INCUMBENT_COLS)

    rows = []
    # host A: 3 events at t = 100.0, 150.0, 200.0 (t_dur 0.2 d each)
    for i, t in enumerate((100.0, 150.0, 200.0)):
        rows.append({"tic": HOST_A, "sector": 5, "t": t, "t_dur": 0.2})
    # host B (candidate): 1 event at t = 300.0
    rows.append({"tic": HOST_B, "sector": 7, "t": 300.0, "t_dur": 0.3})

    n = len(rows)
    shard_path = os.path.join(FIX, "fixture_shard.h5")
    with h5py.File(shard_path, "w") as f:
        g = f.create_group("scalars")
        for c in feats:
            g.create_dataset(c, data=rng.normal(size=n))
        # REAL schema column names (reviewers proved the fixture's invented
        # 't' column masked a KeyError the real shard would raise: the real
        # name is event_time).
        book = {"tic": np.array([r["tic"] for r in rows]),
                "sector": np.array([r["sector"] for r in rows]),
                "event_time": np.array([r["t"] for r in rows]),
                "t_dur": np.array([r["t_dur"] for r in rows]),
                "label": np.zeros(n, dtype=int),
                "split": np.array([b"test"] * n),
                "rebalance_keep": np.ones(n, dtype=int)}
        for name, data in book.items():
            if name in g:                 # t_dur is itself a training scalar
                del g[name]
            g.create_dataset(name, data=data)

    # times: host A has 3 known times - two match (within max(t_dur/2, unc)),
    # one falls far from any event; plus one out-of-window row that must be
    # EXCLUDED by the in-window filter. Host B's single time matches its event
    # only through the H7 rule (gap 0.4 d > t_dur/2 = 0.15 but < unc 0.5).
    times = [
        {"planet": "FIX-A b", "tic": HOST_A, "t_transit_btjd": 100.05,
         "t_unc_days": 0.0, "in_our_cache_window": "yes"},
        {"planet": "FIX-A b", "tic": HOST_A, "t_transit_btjd": 150.09,
         "t_unc_days": 0.0, "in_our_cache_window": "yes"},
        {"planet": "FIX-A b", "tic": HOST_A, "t_transit_btjd": 500.0,
         "t_unc_days": 0.0, "in_our_cache_window": "yes"},
        {"planet": "FIX-A b", "tic": HOST_A, "t_transit_btjd": 600.0,
         "t_unc_days": 0.0, "in_our_cache_window": "no (fixture)"},
        {"planet": "FIX-B cand", "tic": HOST_B, "t_transit_btjd": 300.4,
         "t_unc_days": 0.5, "in_our_cache_window": "yes"},
    ]
    import csv
    times_path = os.path.join(FIX, "fixture_times.csv")
    with open(times_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(times[0].keys()))
        w.writeheader()
        w.writerows(times)

    # hash-log fixture: the REAL checkpoint + tau (asserting the real model is
    # part of the review), synthetic everything else.
    import hashlib
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    real_hlog = json.load(open(os.path.join(
        REPO, "data", "bench", "full", "preshot_hashlog_2026-08-07.json")))
    hlog_path = os.path.join(FIX, "fixture_hashlog.json")
    json.dump(real_hlog, open(hlog_path, "w"), indent=2)

    cfg = {
        "run_label": "FIXTURE_RUN_NOT_REAL",
        "out_dir": os.path.join(FIX, "out"),
        "repo_root": REPO,
        "hash_log": hlog_path,
        "tau": real_hlog["operating_threshold"]["value"],
        "times_tables": [{"mission": "TESS", "path": times_path,
                          "sha256": sha(times_path)}],
        "frozen_shard": shard_path,
        "frozen_shard_hosts": [
            {"name": "FIX-A", "tic": HOST_A, "mission": "TESS",
             "kind": "planet", "expected_rows": 3},
            {"name": "FIX-B", "tic": HOST_B, "mission": "TESS",
             "kind": "candidate", "expected_rows": 1},
        ],
        "export_hosts": [],
        "kepler_hosts": [],
        "tess_train_n_cycles": [3, 5, 8, 13],
        "declared_tier": "FIXTURE",
        "in_window_only": True,
    }
    cfg_path = os.path.join(FIX, "fixture_config.json")
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    # hand-computed + model-computed expectations
    import xgboost as xgb
    import pandas as pd
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(REPO, real_hlog["scoring_model"]["path"]))
    with h5py.File(shard_path, "r") as f:
        df = {k: f["scalars"][k][:] for k in f["scalars"].keys()}
    df = pd.DataFrame({k: v for k, v in df.items() if v.ndim == 1})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    X, _ = arms.score_matrix(df, "b")
    s = model.predict_proba(X)[:, 1]
    tau = float(cfg["tau"])
    passed = (s >= tau).astype(int)
    expected = {
        "FIX-A": {"D2": 3, "D1": 2,
                  "D1_pass": int(passed[0] + passed[1]),
                  "scores": [round(float(x), 6) for x in s[:3]]},
        "FIX-B": {"D2": 1, "D1": 1, "D1_pass": int(passed[3]),
                  "scores": [round(float(s[3]), 6)]},
        "explanations": {
            "FIX-A D1=2": "events at 100.0/150.0 match times 100.05/150.09 "
                          "within t_dur/2 = 0.1; the 500.0 time matches "
                          "nothing; the 600.0 time is filtered out-of-window "
                          "so D2 = 3 not 4",
            "FIX-B D1=1": "gap 0.4 d exceeds t_dur/2 = 0.15 but is within "
                          "the published t_unc 0.5 - the H7 rule is what "
                          "matches it; a t_dur/2-only matcher returns 0 here",
        },
    }
    json.dump(expected, open(os.path.join(FIX, "fixture_expected.json"), "w"),
              indent=2)
    print(f"fixtures written under {FIX}; expected: "
          f"A D1_pass={expected['FIX-A']['D1_pass']}/2, "
          f"B D1_pass={expected['FIX-B']['D1_pass']}/1")


if __name__ == "__main__":
    main()
