"""W3.11: the M1-lite insurance model, the sealed pass, and shortlist v0.

Why an "insurance" model exists at all
--------------------------------------
The real benchmark is not frozen yet, so no honest model can be trained on it.
But the mentor review needs something concrete to look at, and the project needs
a fallback ranking in case the freeze slips. M1-lite is that: a deliberately
simple classifier, trained on provisional data, labelled provisional forever,
and never allowed to become the headline.

Its job is to answer one question well enough to be useful: **of the events the
search flagged, which are worth a human's attention first?**

The availability leak this design avoids
-----------------------------------------
The obvious feature list would include everything the search reports. But some
of those columns exist for one class and not the other, and a feature that is
merely PRESENT more often for positives is a perfect predictor of the label with
no physical content whatsoever.

The ratified feature list is therefore restricted to eight quantities that are
computed identically for both classes:

    depth, duration, snr, sin(2 pi phase), cos(2 pi phase),
    log10 P_bin, morph_coeff, Tmag

and explicitly EXCLUDES ``win_len``, ``det_dependence`` and ``skye_flag``.

Worth recording: in this build ``win_len`` and ``det_dependence`` ARE now
available for both classes, because injected events are routed through the same
``TransitFinderExt._process_cb_events`` that produced the real negatives, so the
original availability concern no longer applies to those two. ``skye_flag``
genuinely remains undefined for injections. The ratified eight are kept anyway,
because M1-lite is meant to be the deliberately naive insurance model and the
full M1 grid is where the richer feature set belongs.

The sealed pass
---------------
Before any model touches the real planets in anger, M1-lite scores the events
belonging to the known real-planet hosts, the scores are hashed, and the hash is
recorded. The file is then **not opened**. That is the whole point: it makes the
later one-shot honest by proving no tuning happened against those numbers, and
it costs nothing to do now.

Coverage is reduced and recorded as such: TIC 172900988 has zero cached light
curves, so it contributes no events. That is the pre-named B10 fallback, logged
rather than silently absorbed.
"""

import argparse
import hashlib
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue, time_to_phase

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
EVENTS = os.path.join(REPO, "data", "search_frozen", "out", "detected_events.txt")
DATA = os.path.join(REPO, "data")

# The ratified eight. Computed identically for both classes; no availability leak.
FEATURES = ["depth", "duration", "snr", "sin_phase", "cos_phase",
            "log_p_bin", "morph_coeff", "tmag"]

SEED = 20260804
TEST_FRACTION = 0.20
SEALED_TICS = [260128333, 319011894, 172900988]
SHORTLIST_N = 50
# Positives are subsampled to this multiple of the negatives. Without it the
# training set is 39,314 positives against 1,028 negatives, a 97.5 percent
# positive prior, which (a) makes PR-AUC meaningless because the baseline is
# already 0.9745, and (b) saturates every score into 0.98-1.00 so the shortlist
# cannot be read. It is also backwards for the deployment context: the real
# queue is almost entirely false positives. Pinned 2026-08-04.
POS_TO_NEG_RATIO = 3.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("11_m1lite")


def attach_host(df, cat, raw):
    """Add the host-level columns both classes share."""
    tics = df.tic.astype(int)
    df = df.copy()
    df["log_p_bin"] = np.log10(cat.reindex(tics).period.to_numpy())
    df["morph_coeff"] = raw.reindex(tics).morph_coeff.to_numpy()
    df["tmag"] = raw.reindex(tics).Tmag.to_numpy()
    df["sin_phase"] = np.sin(2 * np.pi * df.phase.astype(float))
    df["cos_phase"] = np.cos(2 * np.pi * df.phase.astype(float))
    return df


def build_dataset(campaign_dir, cat, raw):
    """Negatives from the real search, positives from the campaign."""
    neg = pd.read_csv(EVENTS, sep=r"\s+")
    neg.columns = [c.lower() for c in neg.columns]
    neg = neg.rename(columns={"time": "event_time"})
    # The incumbent's own filters: this is the manual queue we are triaging.
    keep = ((neg.snr >= 5) & (neg.duration <= 1.0)
            & (neg.det_dependence == 0) & (neg.skye_flag == 0))
    neg = neg[keep].copy()
    neg["label"] = 0
    log.info("Negatives: %d of %d real-search events pass the pinned filters",
             len(neg), int(keep.shape[0]))

    pos_path = os.path.join(campaign_dir, "events_all.csv")
    pos = pd.read_csv(pos_path)
    pos = pos[pos.label == 1].copy()
    n_pos_raw = len(pos)
    # THE SAME FILTERS, applied to BOTH classes. Caught 2026-08-04 by the
    # due-diligence pass: the first version filtered negatives to the incumbent's
    # queue and left positives unfiltered, so 74.5 percent of positives passed
    # the filters against 43.4 percent of negatives. "Does this event pass the
    # incumbent's filters" then separates the classes for a reason that is
    # partly real signal and partly an artefact of which class we filtered. The
    # model is only ever APPLIED to filtered queue events, so training it on an
    # asymmetry it will never see at scoring time is straightforwardly wrong.
    pos_keep = ((pos.snr >= 5) & (pos.duration <= 1.0) & (pos.det_dependence == 0))
    pos = pos[pos_keep].copy()
    log.info("Positives: %d recovered injections, %d after the SAME filters (%.1f%%)",
             n_pos_raw, len(pos), 100 * len(pos) / max(n_pos_raw, 1))
    n_target = int(POS_TO_NEG_RATIO * len(neg))
    if len(pos) > n_target:
        pos = pos.sample(n=n_target, random_state=SEED)
        log.info("Positives subsampled to %d (%.0fx the negatives) so the prior is "
                 "not 97.5%% positive; see POS_TO_NEG_RATIO", len(pos), POS_TO_NEG_RATIO)

    cols = ["tic", "sector", "event_time", "phase", "depth", "duration", "snr", "label"]
    both = pd.concat([neg[cols], pos[cols]], ignore_index=True)
    both = attach_host(both, cat, raw)
    before = len(both)
    both = both.dropna(subset=FEATURES)
    if before != len(both):
        log.info("Dropped %d rows with a missing feature (host row absent)",
                 before - len(both))
    return both


def fit_and_select(df):
    """Two candidate models, selected on validation PR-AUC, grouped by TIC."""
    X = df[FEATURES].to_numpy(dtype=float)
    y = df.label.to_numpy().astype(int)
    groups = df.tic.to_numpy()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                    random_state=SEED).split(X, y, groups))
    log.info("Split by TIC: %d train / %d val events, %d / %d distinct TICs",
             len(tr), len(va), len(set(groups[tr])), len(set(groups[va])))
    assert not (set(groups[tr]) & set(groups[va])), "TIC leaked across the split"

    candidates = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced",
                                                   max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=SEED, n_jobs=4),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(X[tr], y[tr])
        s = model.predict_proba(X[va])[:, 1]
        base = float(y[va].mean())        # PR-AUC of a random ranker
        results[name] = {"pr_auc": float(average_precision_score(y[va], s)),
                         "roc_auc": float(roc_auc_score(y[va], s)),
                         "baseline_pr_auc": base, "model": model}
        results[name]["pr_auc_lift"] = results[name]["pr_auc"] - base
        log.info("  %-7s val PR-AUC %.4f (baseline %.4f, lift %+.4f)  ROC-AUC %.4f", name,
                 results[name]["pr_auc"], base, results[name]["pr_auc_lift"],
                 results[name]["roc_auc"])
    best = max(results, key=lambda k: results[k]["pr_auc"])
    log.info("Selected %s on validation PR-AUC (selection rule pinned)", best)
    return best, results, (tr, va)


def sealed_pass(model, cat, raw, out_dir):
    """Score the real-planet hosts, hash the result, and do not open it."""
    ev = pd.read_csv(EVENTS, sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    ev = ev.rename(columns={"time": "event_time"})
    sub = ev[ev.tic.astype(int).isin(SEALED_TICS)].copy()

    coverage = {}
    for t in SEALED_TICS:
        coverage[str(t)] = int((ev.tic.astype(int) == t).sum())
    log.info("Sealed-pass coverage per TIC: %s", coverage)
    if coverage[str(172900988)] == 0:
        log.warning("TIC 172900988 contributes ZERO events (no cached light curves). "
                    "This is the pre-named B10 reduced scope, recorded not absorbed.")

    if not len(sub):
        return {"n_events": 0, "coverage": coverage, "sha256": None}

    sub = attach_host(sub, cat, raw).dropna(subset=FEATURES)
    scores = model.predict_proba(sub[FEATURES].to_numpy(dtype=float))[:, 1]
    sealed = sub[["tic", "sector", "event_time"]].copy()
    sealed["m1lite_score"] = scores

    sealed_dir = os.path.join(out_dir, "sealed")
    os.makedirs(sealed_dir, exist_ok=True)
    path = os.path.join(sealed_dir, "sealed_scores.csv")
    sealed.to_csv(path, index=False)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    log.info("SEALED: %d events written and hashed. The file is NOT inspected.", len(sealed))
    log.info("  sha256 %s", h)
    return {"n_events": int(len(sealed)), "coverage": coverage, "sha256": h,
            "path": path,
            "note": "Hash-committed before any tuning against real planets. "
                    "Contents deliberately not examined."}


def shortlist(model, cat, raw, out_dir, n=SHORTLIST_N):
    """Rank the real-search queue: what should a human look at first?"""
    ev = pd.read_csv(EVENTS, sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    ev = ev.rename(columns={"time": "event_time"})
    keep = ((ev.snr >= 5) & (ev.duration <= 1.0)
            & (ev.det_dependence == 0) & (ev.skye_flag == 0))
    q = ev[keep].copy()
    # The known real-planet hosts are excluded: the shortlist is a triage tool
    # for UNKNOWN candidates, and their events are sealed.
    q = q[~q.tic.astype(int).isin(SEALED_TICS)]
    q = attach_host(q, cat, raw).dropna(subset=FEATURES)
    q["m1lite_score"] = model.predict_proba(q[FEATURES].to_numpy(dtype=float))[:, 1]
    q = q.sort_values("m1lite_score", ascending=False)

    top = q.head(n)[["tic", "sector", "event_time", "phase", "depth", "duration",
                     "snr", "n_detrend_detections", "m1lite_score"]]
    path = os.path.join(out_dir, "shortlist_v0.csv")
    top.to_csv(path, index=False)
    log.info("Shortlist v0: top %d of %d queue events -> %s", len(top), len(q), path)
    log.info("  distinct TICs in the top %d: %d", n, top.tic.nunique())
    log.info("  score range %.4f to %.4f", top.m1lite_score.min(), top.m1lite_score.max())
    return {"n_queue": int(len(q)), "n_shortlist": int(len(top)),
            "distinct_tics": int(top.tic.nunique()),
            "score_min": float(top.m1lite_score.min()),
            "score_max": float(top.m1lite_score.max()), "path": path,
            "privacy": "TICs appear in this LOCAL file only; no shortlist TIC "
                       "may appear in any public artifact."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", default=os.path.join(DATA, "pilot_out"))
    args = ap.parse_args()

    cat = load_catalogue(CAT_PATH, TEBC=True).drop_duplicates("tess_id").set_index("tess_id")
    raw = pd.read_csv(CAT_PATH).drop_duplicates("tess_id").set_index("tess_id")

    df = build_dataset(args.campaign_dir, cat, raw)
    log.info("M1-lite dataset: %d events (%d positive), %d distinct TICs",
             len(df), int(df.label.sum()), df.tic.nunique())

    best, results, _ = fit_and_select(df)
    model = results[best]["model"]

    out_dir = os.path.join(DATA, "m1lite")
    os.makedirs(out_dir, exist_ok=True)
    sealed = sealed_pass(model, cat, raw, out_dir)
    sl = shortlist(model, cat, raw, out_dir)

    summary = {
        "status": "PROVISIONAL FOREVER. Trained on unfrozen data; never the headline.",
        "features": FEATURES,
        "excluded_features": ["win_len", "det_dependence", "skye_flag"],
        "exclusion_reason": "availability leak: a feature present for one class "
                            "predicts the label with no physical content. Note that "
                            "win_len and det_dependence ARE now available for both "
                            "classes in this build; they are kept out because "
                            "M1-lite is the deliberately naive insurance model.",
        "n_events": int(len(df)), "n_positive": int(df.label.sum()),
        "distinct_tics": int(df.tic.nunique()),
        "selected_model": best,
        "val_scores": {k: {"pr_auc": v["pr_auc"], "roc_auc": v["roc_auc"],
                           "baseline_pr_auc": v["baseline_pr_auc"],
                           "pr_auc_lift": v["pr_auc_lift"]}
                       for k, v in results.items()},
        "pos_to_neg_ratio": POS_TO_NEG_RATIO,
        "split": {"rule": "GroupShuffleSplit by TIC", "test_fraction": TEST_FRACTION,
                  "seed": SEED},
        "sealed": sealed,
        "shortlist": sl,
    }
    with open(os.path.join(out_dir, "m1lite_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote %s", os.path.join(out_dir, "m1lite_summary.json"))


if __name__ == "__main__":
    main()
