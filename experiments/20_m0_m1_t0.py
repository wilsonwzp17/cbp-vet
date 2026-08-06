"""W4.1 + W4.2 + the T0 core: M0's operating points, the M1 grid, matched recall.

M0, the incumbent as it actually operates (pins re-read from source today)
--------------------------------------------------------------------------
Direction-Memo Week-4 item 1 fixes BOTH heuristic operating points:
  OP-A = thresholds only  (snr >= 5, duration <= 1.0 d, det_dependence == 0,
         skye_flag == 0 where defined)
  OP-B = thresholds PLUS model comparison (best_fit in {T, AT})
skye_flag is undefined for injections (it is a per-sector metric of the real
search); an undefined skye is treated as PASS, matching how the incumbent's own
completeness numbers are measured on injections. Recorded, not assumed silently.

M1, per the ratified pins
-------------------------
logreg / random forest / XGBoost on the frozen shard's training scalars,
selection on validation PR-AUC ONLY; XGBoost routes NaN natively; LR/RF get
median-impute + validity bits. Arm b0 (incumbent columns only) trains alongside
as the incumbent-information baseline. The test split is used exactly once, at
the end, for the banked matched-recall table.

The matched-recall harness (the readiness pin, verbatim in effect)
------------------------------------------------------------------
recall_M0(OP) = test positives passing OP / test positives, pooled AND on the
0.15-0.30 percent depth stratum; tau = largest score with recall >= recall_M0;
FP(tau) counted on real-search test negatives; headline = manual-inspection
events per 1,000 light curves, denominator = test-split (TIC, sector) files;
plus the dual (recall at M0-matched FP); TIC-cluster bootstrap, 2,000
resamples, 2.5/97.5 percentiles. Both FP-only and FP+TP queue counts emitted.
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
STRATUM = (0.0015, 0.0030)        # the 0.15-0.30% headline depth stratum

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("20_m0m1t0")


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
    """The incumbent's operating points on shard rows."""
    m = (df.snr >= 5) & (df.t_dur <= 1.0) & (df.detrend_fraction * 21 > 18)
    skye = df.skye_flag.to_numpy(dtype=float)
    m &= ~(np.isfinite(skye) & (skye != 0))          # undefined skye = pass
    if op == "B":
        m &= (df.best_fit_T + df.best_fit_AT) > 0
    return m.to_numpy()


def main():
    # ================= M0: the incumbent on the frozen test split ==========
    test = load_frame(["test"])
    pos = test[test.label == 1]
    neg_real = test[(test.label == 0) & (test.label_source == 0)]
    stratum = pos[(pos.depth >= STRATUM[0]) & (pos.depth <= STRATUM[1])]
    n_test_files = test.groupby(["tic", "sector"]).ngroups
    n_real_files = neg_real.groupby(["tic", "sector"]).ngroups

    m0 = {}
    for op in ("A", "B"):
        mp = op_mask(pos, op); ms = op_mask(stratum, op); mn = op_mask(neg_real, op)
        m0[op] = {
            "recall_pooled": float(mp.mean()),
            "recall_stratum": float(ms.mean()),
            "n_stratum": int(len(stratum)),
            "real_fp_flagged": int(mn.sum()),
            "fp_per_1000_lc": float(1000 * mn.sum() / max(n_test_files, 1)),
        }
        log.info("M0 OP-%s: recall pooled %.4f | stratum %.4f | real FPs %d "
                 "(%.1f per 1,000 test light curves)",
                 op, m0[op]["recall_pooled"], m0[op]["recall_stratum"],
                 m0[op]["real_fp_flagged"], m0[op]["fp_per_1000_lc"])

    # ================= M1 grid on train, selected on val ====================
    results = {}
    for arm in ("b0", "b"):
        Xtr, ytr, gtr, vtr, meta_tr = arms.load_matrix(SHARD, arm, splits=("train",))
        Xva, yva, gva, vva, meta_va = arms.load_matrix(SHARD, arm, splits=("val",))
        assert meta_tr["features"] == meta_va["features"]
        cand = {}
        Xtr_d, med = arms.impute_for_dense(Xtr, vtr)
        Xva_d, _ = arms.impute_for_dense(Xva, vva, med)
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(class_weight="balanced", max_iter=2000,
                                              random_state=SEED)).fit(Xtr_d, ytr)
        cand["logreg"] = lr.predict_proba(Xva_d)[:, 1]
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=SEED, n_jobs=8).fit(Xtr_d, ytr)
        cand["rf"] = rf.predict_proba(Xva_d)[:, 1]
        xg = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.9,
                               eval_metric="logloss", random_state=SEED,
                               n_jobs=8).fit(Xtr, ytr)          # NaN-native
        cand["xgboost"] = xg.predict_proba(Xva)[:, 1]
        scores = {k: {"val_pr_auc": float(average_precision_score(yva, s)),
                      "val_roc_auc": float(roc_auc_score(yva, s))}
                  for k, s in cand.items()}
        best = max(scores, key=lambda k: scores[k]["val_pr_auc"])
        results[arm] = {"scores": scores, "selected": best,
                        "n_features": len(meta_tr["features"])}
        log.info("arm %-3s: " + " | ".join(f"{k} PR {v['val_pr_auc']:.4f}"
                 for k, v in scores.items()) + f"  -> selected {best}", arm)
        results[arm]["_models"] = {"logreg": (lr, med), "rf": (rf, med), "xgboost": (xg, None)}

    # ================= T0 core: matched recall on TEST ======================
    t0 = {}
    for arm in ("b0", "b"):
        best = results[arm]["selected"]
        model, med = results[arm]["_models"][best]
        Xte, yte, gte, vte, meta_te = arms.load_matrix(SHARD, arm, splits=("test",))
        if med is not None:
            Xte_use, _ = arms.impute_for_dense(Xte, vte, med)
        else:
            Xte_use = Xte
        s = model.predict_proba(Xte_use)[:, 1]
        te = load_frame(["test"])
        te["score"] = s
        p = te[te.label == 1]
        strat = p[(p.depth >= STRATUM[0]) & (p.depth <= STRATUM[1])]
        nr = te[(te.label == 0) & (te.label_source == 0)]
        t0[arm] = {"selected_model": best}
        for op in ("A", "B"):
            for scope, subset in (("pooled", p), ("stratum", strat)):
                r_m0 = float(op_mask(subset, op).mean())
                if not len(subset):
                    continue
                tau = float(np.quantile(subset.score, 1 - r_m0)) if r_m0 < 1 else float(subset.score.min())
                fp = int((nr.score >= tau).sum())
                # TIC-cluster bootstrap on the FP count
                rng = np.random.default_rng(SEED)
                tics = nr.tic.unique()
                bs = []
                for _ in range(2000):
                    pick = rng.choice(tics, size=len(tics), replace=True)
                    cnt = sum(int((nr[nr.tic == t].score >= tau).sum()) for t in pick)
                    bs.append(1000 * cnt / max(n_real_files, 1))
                t0[arm][f"OP{op}_{scope}"] = {
                    "m0_recall": r_m0, "tau": tau,
                    "model_fp_at_matched_recall": fp,
                    "model_fp_per_1000_lc": float(1000 * fp / max(n_real_files, 1)),
                    "m0_fp_per_1000_lc": float(1000 * op_mask(nr, op).sum() / max(n_real_files, 1)),
                    "fp_ci_2p5_97p5": [float(np.percentile(bs, 2.5)),
                                        float(np.percentile(bs, 97.5))],
                    "queue_fp_only": fp,
                    "queue_fp_plus_tp": fp + int((subset.score >= tau).sum()),
                }
        log.info("arm %s (%s): OPB stratum -> M0 FP/1000 %.1f vs model %.1f "
                 "[CI %.1f-%.1f] at matched recall %.3f",
                 arm, best,
                 t0[arm]["OPB_stratum"]["m0_fp_per_1000_lc"],
                 t0[arm]["OPB_stratum"]["model_fp_per_1000_lc"],
                 *t0[arm]["OPB_stratum"]["fp_ci_2p5_97p5"],
                 t0[arm]["OPB_stratum"]["m0_recall"])

    for arm in results:
        results[arm].pop("_models")
    out = {"m0": m0, "m1_grid": results, "t0_matched_recall": t0,
           "n_test_files": int(n_test_files), "n_real_test_files": int(n_real_files),
           "stratum": list(STRATUM), "seed": SEED,
           "skye_rule": "undefined skye treated as PASS (injections), recorded",
           "pins": "OPs per Direction-Memo W4 item 1; selection on val PR-AUC only; "
                   "harness per Execution-Readiness M0 recipe"}
    with open(os.path.join(OUT, "t0_core.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("Wrote %s", os.path.join(OUT, "t0_core.json"))


if __name__ == "__main__":
    main()
