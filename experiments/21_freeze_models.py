"""Complete the pinned T0 harness: model checkpoints + the dual metric.

Why this script exists (2026-08-06 final-sweep findings)
--------------------------------------------------------
The pre-compaction due-diligence sweep found three gaps between the banked T0
(`t0_core.json`, written by `20_m0_m1_t0.py`) and the ratified pins:

1. NO FROZEN-MODEL CHECKPOINT existed. Script 20 trains in memory and persists
   only numbers, but the Aug-10 pre-shot freeze, the Aug-11 one-shot's opening
   hash assert, and the deployment shortlist all need a loadable artifact.
2. THE DUAL METRIC WAS NEVER COMPUTED. The Execution-Readiness pin reads
   "...plus the dual (recall at M0-matched FP)"; script 20's own docstring
   repeats it, yet no dual field exists in t0_core.json.
3. THE PER-1,000 DENOMINATOR in the t0 block (251 real-search-negative files)
   is not the pinned one (365 test-split (TIC, sector) files). The 8.1x ratio
   is denominator-invariant; the absolute numbers need an honest restatement.

This script fixes all three WITHOUT touching the banked record: it retrains
with the identical code path, seed, and hyperparameters, ASSERTS bit-level
agreement with t0_core.json (val PR-AUCs and all eight banked FP counts at
the banked taus) before writing anything, then persists the two selected
XGBoost models plus `t0_addendum_2026-08-06.json`. If any assert fires, the
run aborts and nothing is written. t0_core.json itself is never modified.

The dual touches the same frozen test scores T0 already used; it completes
the pre-registered harness rather than performing a second model selection
(nothing here feeds back into any training or selection choice).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import logging
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import average_precision_score
import xgboost as xgb

from cbpvet.models import arms

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
SHARD = os.path.join(REPO, "data", "bench", "full", "bench_full.h5")
OUT = os.path.join(REPO, "data", "bench", "full")
MODEL_DIR = os.path.join(OUT, "models")
SEED = 20260806
STRATUM = (0.0015, 0.0030)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("21_freeze")


def load_frame(splits):
    with h5py.File(SHARD, "r") as f:
        cols = {k: f["scalars"][k][:] for k in f["scalars"].keys()}
    df = pd.DataFrame({k: v for k, v in cols.items() if v.ndim == 1})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    df = df[(df.rebalance_keep == 1) & (df["split"].isin(splits))]
    return df.reset_index(drop=True)


def op_mask(df, op):
    m = (df.snr >= 5) & (df.t_dur <= 1.0) & (df.detrend_fraction * 21 > 18)
    skye = df.skye_flag.to_numpy(dtype=float)
    m &= ~(np.isfinite(skye) & (skye != 0))
    if op == "B":
        m &= (df.best_fit_T + df.best_fit_AT) > 0
    return m.to_numpy()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    core = json.load(open(os.path.join(OUT, "t0_core.json")))
    os.makedirs(MODEL_DIR, exist_ok=True)

    models, model_meta = {}, {}
    for arm in ("b0", "b"):
        assert core["m1_grid"][arm]["selected"] == "xgboost"
        Xtr, ytr, _, _, meta_tr = arms.load_matrix(SHARD, arm, splits=("train",))
        Xva, yva, _, _, _ = arms.load_matrix(SHARD, arm, splits=("val",))
        xg = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.9,
                               eval_metric="logloss", random_state=SEED,
                               n_jobs=8).fit(Xtr, ytr)
        pr = float(average_precision_score(yva, xg.predict_proba(Xva)[:, 1]))
        banked = core["m1_grid"][arm]["scores"]["xgboost"]["val_pr_auc"]
        assert abs(pr - banked) < 1e-9, (arm, pr, banked)
        log.info("arm %s: retrained val PR-AUC %.10f == banked (checkpoint valid)",
                 arm, pr)
        models[arm] = xg
        model_meta[arm] = {"n_features": len(meta_tr["features"])}

    test = load_frame(["test"])
    pos = test[test.label == 1]
    strat = pos[(pos.depth >= STRATUM[0]) & (pos.depth <= STRATUM[1])]
    nr = test[(test.label == 0) & (test.label_source == 0)]
    n_test_files = int(core["n_test_files"])
    n_real_files = int(core["n_real_test_files"])

    addendum = {"purpose": (
        "Completes the pinned T0 harness (dual metric; pinned per-1,000 "
        "denominator; frozen model checkpoints). t0_core.json is the banked "
        "record and is unmodified; every number here was written only after "
        "the retrained models reproduced all banked val PR-AUCs and all "
        "eight banked FP counts exactly."), "seed": SEED}

    dual, restate = {}, {}
    for arm in ("b0", "b"):
        Xte, yte, _, _, _ = arms.load_matrix(SHARD, arm, splits=("test",))
        s = models[arm].predict_proba(Xte)[:, 1]
        te = test.copy()
        te["score"] = s
        p = te[te.label == 1]
        st = p[(p.depth >= STRATUM[0]) & (p.depth <= STRATUM[1])]
        neg = te[(te.label == 0) & (te.label_source == 0)]

        # Reproduce every banked FP count at the banked taus before trusting s.
        for op in ("A", "B"):
            for scope, subset in (("pooled", p), ("stratum", st)):
                cell = core["t0_matched_recall"][arm][f"OP{op}_{scope}"]
                fp = int((neg.score >= cell["tau"]).sum())
                assert fp == cell["model_fp_at_matched_recall"], (arm, op, scope)

        dual[arm], restate[arm] = {}, {}
        neg_sorted = np.sort(neg.score.to_numpy())[::-1]
        for op in ("A", "B"):
            k = core["m0"][op]["real_fp_flagged"]      # M0's FP budget
            tau_d = float(neg_sorted[k - 1])
            fp_at = int((neg.score >= tau_d).sum())
            for scope, subset in (("pooled", p), ("stratum", st)):
                cell = {
                    "m0_fp_budget": int(k),
                    "tau_dual": tau_d,
                    "model_fp_at_tau_dual": fp_at,     # == k unless score ties
                    "model_recall_at_m0_fp": float((subset.score >= tau_d).mean()),
                    "m0_recall_same_cell": core["t0_matched_recall"][arm][
                        f"OP{op}_{scope}"]["m0_recall"],
                }
                dual[arm][f"OP{op}_{scope}"] = cell

        for op in ("A", "B"):
            for scope in ("pooled", "stratum"):
                c = core["t0_matched_recall"][arm][f"OP{op}_{scope}"]
                restate[arm][f"OP{op}_{scope}"] = {
                    "model_fp_per_1000_test_files": 1000 * c["queue_fp_only"] / n_test_files,
                    "m0_fp_per_1000_test_files": 1000 * core["m0"][op]["real_fp_flagged"] / n_test_files,
                }

    addendum["dual_recall_at_m0_matched_fp"] = dual
    addendum["pinned_denominator_restatement"] = {
        "note": (
            "The pin sets denominator = test-split (TIC, sector) files "
            f"(n={n_test_files}). t0_core's t0 block divided by real-search-"
            f"negative files (n={n_real_files}); its m0 block honored the pin. "
            "The matched-recall RATIO is denominator-invariant (identical "
            "negatives both sides): headline stays 8.1x either way. Quote ONE "
            "convention per surface, never mixed: pinned 178.1 vs 21.9, or "
            "per-real-negative-file 259.0 vs 31.9 with that label."),
        "per_1000_test_files": restate,
    }
    addendum["deviations_recorded"] = [
        "tau via np.quantile linear interpolation, not the order-statistic; "
        "measured effect: identical in 7 of 8 cells, +1 model FP in b0 "
        "OPA_stratum only (conservative against the model).",
        "bootstrap resamples TIC clusters over a fixed file denominator; "
        "ratio-estimator recompute narrows the headline CI [3.98,71.81] to "
        "[4.26,67.46] (banked CI is the wider, conservative one).",
    ]

    freeze = {"frozen_at": "2026-08-06", "seed": SEED, "models": {}}
    for arm in ("b0", "b"):
        path = os.path.join(MODEL_DIR, f"arm_{arm}_xgb.json")
        models[arm].save_model(path)
        freeze["models"][arm] = {
            "path": os.path.relpath(path, REPO),
            "sha256": sha256(path),
            "selected": "xgboost",
            "val_pr_auc": core["m1_grid"][arm]["scores"]["xgboost"]["val_pr_auc"],
            "n_features": model_meta[arm]["n_features"],
            "taus_banked": {k: core["t0_matched_recall"][arm][k]["tau"]
                            for k in ("OPA_pooled", "OPA_stratum",
                                      "OPB_pooled", "OPB_stratum")},
        }
    addendum["model_freeze"] = freeze

    out_path = os.path.join(OUT, "t0_addendum_2026-08-06.json")
    with open(out_path, "w") as fh:
        json.dump(addendum, fh, indent=2)
    log.info("Wrote %s and %d model checkpoints", out_path, len(freeze["models"]))

    b = dual["b"]["OPB_stratum"]
    log.info("DUAL headline: at M0 OP-B's FP budget (%d), arm b recall %.4f "
             "vs M0's %.4f on the stratum", b["m0_fp_budget"],
             b["model_recall_at_m0_fp"], b["m0_recall_same_cell"])


if __name__ == "__main__":
    main()
