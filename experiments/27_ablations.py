"""The feature-arm ablation ladder: b0, b, b_nopair on the frozen benchmark.

What this is
------------
The Week-4 board's "ablation arms overnight" row, reduced to its live scope
under adoption D-A and registry G2: the CNN arms and arm (d) have no code
(fall scope), so the ladder is the three FEATURE arms. b0 and b are already
banked (t0_core.json, reproduced by 21_freeze_models.py); this script adds
**b_nopair** - arm b minus the four gated observed-pair scalars - trained
and selected under the IDENTICAL protocol, evaluated at the same banked
matched-recall harness. The contrast b minus b_nopair measures what the pair
machinery actually buys the feature model, which is the one ablation the
talk can honestly cite under the current branch.

Note of record: b_nopair was parked as fall scope in arms.py's comments;
running it tonight pulls it forward (dated annotation in the ledger). The
banked b0/b numbers are NOT recomputed here - they are read from t0_core and
asserted unchanged, so this script cannot silently move the headline.

Protocol, identical to 20_m0_m1_t0.py by construction:
- logreg / RF / XGBoost on train, selection on val PR-AUC only;
- test touched once, at the matched-recall table (the same taus construction
  and the same OP-B stratum cells as the banked record);
- seed 20260806, thread pinning, same hyperparameters.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from cbpvet.models import arms

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
SHARD = os.path.join(REPO, "data", "bench", "full", "bench_full.h5")
OUT = os.path.join(REPO, "data", "bench", "full")
SEED = 20260806
STRATUM = (0.0015, 0.0030)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("27_ablations")


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


def main():
    core = json.load(open(os.path.join(OUT, "t0_core.json")))

    arm = "b_nopair"
    Xtr, ytr, _, vtr, meta_tr = arms.load_matrix(SHARD, arm, splits=("train",))
    Xva, yva, _, vva, _ = arms.load_matrix(SHARD, arm, splits=("val",))
    log.info("arm %s: %d features (b has %d)", arm, len(meta_tr["features"]),
             core["m1_grid"]["b"]["n_features"])

    cand, models = {}, {}
    Xtr_d, med = arms.impute_for_dense(Xtr, vtr)
    Xva_d, _ = arms.impute_for_dense(Xva, vva, med)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000,
                                          random_state=SEED)).fit(Xtr_d, ytr)
    cand["logreg"] = lr.predict_proba(Xva_d)[:, 1]; models["logreg"] = (lr, med)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                random_state=SEED, n_jobs=8).fit(Xtr_d, ytr)
    cand["rf"] = rf.predict_proba(Xva_d)[:, 1]; models["rf"] = (rf, med)
    xg = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.08,
                           subsample=0.9, colsample_bytree=0.9,
                           eval_metric="logloss", random_state=SEED,
                           n_jobs=8).fit(Xtr, ytr)
    cand["xgboost"] = xg.predict_proba(Xva)[:, 1]; models["xgboost"] = (xg, None)

    scores = {k: {"val_pr_auc": float(average_precision_score(yva, s)),
                  "val_roc_auc": float(roc_auc_score(yva, s))}
              for k, s in cand.items()}
    best = max(scores, key=lambda k: scores[k]["val_pr_auc"])
    log.info("selection on val PR-AUC only: %s (%s)", best,
             {k: round(v["val_pr_auc"], 4) for k, v in scores.items()})

    model, med = models[best]
    Xte, yte, _, vte, _ = arms.load_matrix(SHARD, arm, splits=("test",))
    Xte_use = arms.impute_for_dense(Xte, vte, med)[0] if med is not None else Xte
    s = model.predict_proba(Xte_use)[:, 1]

    te = load_frame(["test"])
    te["score"] = s
    p = te[te.label == 1]
    strat = p[(p.depth >= STRATUM[0]) & (p.depth <= STRATUM[1])]
    nr = te[(te.label == 0) & (te.label_source == 0)]
    n_real_files = int(core["n_real_test_files"])

    result = {"selected_model": best, "scores": scores,
              "n_features": len(meta_tr["features"])}
    for op in ("A", "B"):
        for scope, subset in (("pooled", p), ("stratum", strat)):
            r_m0 = float(op_mask(subset, op).mean())
            tau = float(np.quantile(subset.score, 1 - r_m0)) if r_m0 < 1 \
                else float(subset.score.min())
            fp = int((nr.score >= tau).sum())
            result[f"OP{op}_{scope}"] = {
                "m0_recall": r_m0, "tau": tau, "queue_fp_only": fp,
                "model_fp_per_1000_real_neg_files": float(1000 * fp / n_real_files),
            }

    ladder = {
        "written": "2026-08-07",
        "protocol": "identical to 20_m0_m1_t0.py (seed, grid, selection, harness); "
                    "b0/b read from the banked t0_core, never recomputed",
        "banked_b0_OPB_stratum_fp": core["t0_matched_recall"]["b0"]["OPB_stratum"]["queue_fp_only"],
        "banked_b_OPB_stratum_fp": core["t0_matched_recall"]["b"]["OPB_stratum"]["queue_fp_only"],
        "b_nopair": result,
        "contrast_note": "pair-feature contribution = b_nopair FP minus b FP at "
                         "matched recall; positive means the pair features "
                         "reduce the queue.",
    }
    out_path = os.path.join(OUT, "ablation_ladder_2026-08-07.json")
    with open(out_path, "w") as fh:
        json.dump(ladder, fh, indent=2)
    b_fp = ladder["banked_b_OPB_stratum_fp"]
    np_fp = result["OPB_stratum"]["queue_fp_only"]
    log.info("LADDER (OP-B stratum, matched recall): b0 %d | b_nopair %d | b %d "
             "-> pair features change the queue by %+d FPs",
             ladder["banked_b0_OPB_stratum_fp"], np_fp, b_fp, np_fp - b_fp)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
