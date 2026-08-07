"""Deployment export + frozen-model ranking: TCEs -> records -> shortlist v1.

What this is
------------
Stage 2 of the deployment run (stage 1 is 23_deploy_run.py's mask + search).
It pipes the deployment TCEs through the FROZEN exporter code path - the same
``cbpvet/export`` modules, unmodified, that built bench-v1 (tool use, not an
unfreeze) - runs mono-cbp's own model comparator to fill the incumbent block,
writes a scoring shard under data/deploy_run/ (untracked, LOCAL ONLY), then
ranks every event with the hash-asserted frozen arm-b checkpoint and emits
shortlist v1.

The discipline, in order of importance:
1. CHECKPOINT HASH ASSERT FIRST. The model file must hash to exactly the
   value recorded in t0_addendum_2026-08-06.json's model_freeze block, so the
   model ranking the sky is provably the model that banked the 8.1x. Runs
   refuse to start otherwise.
2. THE EXPORTER IS USED, NEVER REIMPLEMENTED. Records come from
   EventExporter.build_record with label_source="real_search" (deployment
   events ARE real search output; their label is unknown and never used -
   only features are consumed). One code path, invariant 3.
3. SCORING USES THE SINGLE GATE (arms.score_matrix): same forbidden-column
   contract as training, no ad-hoc feature list.
4. PRIVACY: every output lands under data/ (untracked). Shortlist TICs
   appear in no public artifact, ever.

The 15-column shortlist format (pin adopted 2026-08-07 at the fork, registry
section G): tic, sector, event_time, phase, depth, duration, snr,
n_detrend_detections, skye_flag, best_fit_label, arm_b_score, rank,
coverage_tier, sector_overlap_frozen, disposition (placeholder, filled at
review). Rationale: shortlist_v0's eight observables carried forward, plus
the incumbent's best-fit label for at-a-glance triage, the frozen-model
score and rank, the two context flags the disposition review needs, and the
disposition column itself.

Disjointness pin honored downstream: sector_overlap_frozen marks (tic,
sector) files that are in the frozen search's staged set (1,945 searched;
the 1,946th file was excluded pre-search), so the headline detectability
curve can restrict to non-overlapping files and the overlap is disclosed.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import logging
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet.export import EventExporter, build_block, schema, write_shard
from cbpvet.models import arms

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
CAT_PATH = os.path.expanduser("~/mono-cbp/catalogues/TEBC_morph_05_P_7.csv")
NOISE = os.path.join(REPO, "data", "noise_screen.csv")
ADDENDUM = os.path.join(REPO, "data", "bench", "full", "t0_addendum_2026-08-06.json")
FROZEN_LIST = os.path.join(REPO, "data", "search_frozen", "staged")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("25_deploy_score")

SHORTLIST_COLS = ["tic", "sector", "event_time", "phase", "depth", "duration",
                  "snr", "n_detrend_detections", "skye_flag", "best_fit_label",
                  "arm_b_score", "rank", "coverage_tier",
                  "sector_overlap_frozen", "disposition"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def comparator_key(tic, sector, event_no):
    return f"TIC_{tic}_{sector}_{event_no}"


def load_frozen_model(arm="b"):
    add = json.load(open(ADDENDUM))
    entry = add["model_freeze"]["models"][arm]
    path = os.path.join(REPO, entry["path"])
    got = sha256(path)
    if got != entry["sha256"]:
        raise RuntimeError(
            f"checkpoint hash mismatch for arm {arm}: {got} != recorded "
            f"{entry['sha256']}. REFUSING to score with an unverified model.")
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(path)
    log.info("frozen arm-%s checkpoint verified (%s...) and loaded",
             arm, entry["sha256"][:16])
    return model, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True,
                    help="detected_events.txt from 23_deploy_run.py")
    ap.add_argument("--staged-lc", required=True,
                    help="the masked staged light curves the search ran on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--comparator-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, entry = load_frozen_model("b")     # hash-assert BEFORE any work

    ev = pd.read_csv(args.events, sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    ev = ev.rename(columns={"time": "event_time"})
    ev["label"] = 0            # unknown; never used - features only
    ev["inverted_lc"] = 0
    if args.limit:
        ev = ev.head(args.limit)
    log.info("deployment events: %d from %d TICs", len(ev), ev.tic.nunique())

    cat = load_catalogue(CAT_PATH, TEBC=True)
    raw = pd.read_csv(CAT_PATH)
    noise = pd.read_csv(NOISE)
    exp = EventExporter(args.staged_lc, cat, raw, noise_screen=noise,
                        e2_passed=True)

    records, locals_, recurs, keys, payloads = [], [], [], [], []
    t0 = _time.time()
    exp.set_system_events(ev, key=("tic",))
    skipped = 0
    for _, e in ev.iterrows():
        built = exp.build_record(e, "real_search")
        if built is None:
            skipped += 1
            continue
        rec, loc, rc = built
        rec["pair_model_version"] = 0
        rec["source_dir"] = "deploy_run"
        event_no = len(records)
        rec["event_key"] = comparator_key(int(e.tic), int(e.sector), event_no)
        records.append(rec); locals_.append(loc); recurs.append(rc)
        keys.append(rec["event_key"])
        cached = exp._cached(e.tic, e.sector)
        half = schema.local_halfwidth(float(e.duration))
        sel = np.abs(cached["time"] - float(e.event_time)) <= half
        payloads.append({"time": cached["time"][sel], "flux": cached["flux"][sel],
                         "flux_err": cached["flux_err"][sel],
                         "event_time": float(e.event_time),
                         "event_width": float(e.duration),
                         "tic": int(e.tic), "sector": int(e.sector),
                         "event_no": event_no})
    if not records:
        raise SystemExit("no records built - nothing to rank")
    log.info("built %d records (%d skipped) in %.1f s",
             len(records), skipped, _time.time() - t0)

    # ---- the incumbent block via mono-cbp's own comparator ----------------
    # Import under the real file-backed name "13_export" (not an alias): the
    # comparator pool uses spawn on macOS, and workers re-import the chunk fn's
    # module by name. An alias like "exp13" has no importable file behind it,
    # so pool.map dies with PicklingError: import of module 'exp13' failed.
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import importlib
    exp13 = importlib.import_module("13_export")
    rows, per_event = exp13.run_comparator(payloads, args.out,
                                           workers=args.comparator_workers)
    inc = build_block(rows, keys)

    for rec in records:
        rec.update({f"valid_{k}": v for k, v in exp.scalar_valid(rec).items()})
        rec["rebalance_keep"] = 1          # scoring shard: every row scored
        rec["split"] = "deploy"

    shard = os.path.join(args.out, "deploy_shard.h5")
    df = write_shard(shard, records, locals_, recurs, inc)

    # ---- rank with the frozen model through the single gate ---------------
    X, feats = arms.score_matrix(df, "b")
    assert len(feats) == entry["n_features"], (len(feats), entry["n_features"])
    scores = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["arm_b_score"] = scores

    # context flags for the shortlist
    frozen_files = set()
    if os.path.isdir(FROZEN_LIST):
        for f in os.listdir(FROZEN_LIST):
            if f.startswith("TIC_") and f.endswith(".txt"):
                p = f[:-4].split("_")
                frozen_files.add((int(p[1]), int(p[2])))
    df["sector_overlap_frozen"] = [
        int((int(t), int(s)) in frozen_files)
        for t, s in zip(df.tic, df.sector)]
    df["coverage_tier"] = df["n_cycles_raw"] if "n_cycles_raw" in df else np.nan
    # The shard stores these two pinned shortlist columns under the exporter's
    # names (t_dur, detrend_fraction = n/21); without the mapping the "in
    # ranked.columns" filter silently dropped them from the 15-column pin.
    df["duration"] = df["t_dur"]
    df["n_detrend_detections"] = np.rint(df["detrend_fraction"] * 21.0).astype(int)
    # The comparator keys its rows by 'filename' (comparator_key strings) and
    # carries the winner in 'best_fit' -- verified against incumbent.py and a
    # produced model_comparison.csv. There is no 'event_key'/'best_model' there.
    df["best_fit_label"] = ""
    if rows is not None and len(rows) and "best_fit" in rows.columns:
        bf = dict(zip(rows["filename"].astype(str), rows["best_fit"].astype(str)))
        df["best_fit_label"] = [bf.get(str(k), "") for k in df.event_key]

    ranked = df.sort_values("arm_b_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["disposition"] = ""
    full_path = os.path.join(args.out, "ranked_all.csv")
    ranked_cols = [c for c in SHORTLIST_COLS if c in ranked.columns]
    ranked[ranked_cols].to_csv(full_path, index=False)
    top = ranked.head(50)
    sl_path = os.path.join(args.out, "shortlist_v1.csv")
    top[ranked_cols].to_csv(sl_path, index=False)

    summary = {
        "built": "2026-08-07",
        "checkpoint_sha256": entry["sha256"],
        "n_events_in": int(len(ev)),
        "n_records": int(len(records)),
        "n_skipped": int(skipped),
        "n_tics": int(df.tic.nunique()),
        "comparator_s_per_event": per_event,
        "incumbent_valid_fraction": float(df.incumbent_valid.mean())
            if "incumbent_valid" in df else 0.0,
        "score_quantiles": {q: float(np.quantile(scores, float(q)))
                            for q in ("0.5", "0.9", "0.99")},
        "n_above_banked_tau": int((scores >=
            entry["taus_banked"]["OPB_stratum"]).sum()),
        # distinct (tic, sector) FILES in the frozen 1,946, as the key says;
        # df.sector_overlap_frozen.sum() would count EVENTS and overstate the
        # disclosed overlap wherever a frozen file yields more than one event.
        "sector_overlap_frozen_files": int(
            df.loc[df.sector_overlap_frozen == 1, ["tic", "sector"]]
              .drop_duplicates().shape[0]),
        "sector_overlap_frozen_events": int(df.sector_overlap_frozen.sum()),
        "shard_sha256": sha256(shard),
        "ranked_all_sha256": sha256(full_path),
        "shortlist_v1_sha256": sha256(sl_path),
        "shortlist_columns": ranked_cols,
    }
    with open(os.path.join(args.out, "deploy_score_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("shard %d rows; %d events above the banked OP-B tau; shortlist "
             "v1 written (LOCAL ONLY)", len(df), summary["n_above_banked_tau"])


if __name__ == "__main__":
    main()
