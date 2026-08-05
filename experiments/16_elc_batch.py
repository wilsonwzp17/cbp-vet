"""The ELC batch: turn ELC models into labelled positives, and measure the
DETECTED pair rate.

This closes the loop the census could not close on its own. The census measures
GEOMETRIC rates from ELC's transit-time files: how often the planet crosses both
stars. What actually matters for the benchmark is the DETECTED rate: how often
the search finds both crossings once they sit in a real, noisy light curve.

The difference is the whole point of the depth-ratio screen. A partner crossing
of the fainter star can be geometrically certain and photometrically invisible.
Kostov 2020b's observed 4-of-11 incidence is a detected rate; ELC's geometry is
not. Running the models through injection and search is what makes the two
comparable.

What this produces
------------------
1. ELC positives for the training set, with the same columns as the bank
   positives, extracted by the same ``TransitFinderExt`` code path, so probe 2
   can ask whether a classifier can tell the two provenances apart. Until these
   exist, probe 2 cannot run at all.
2. The detected pair rate, per sigma bin.
3. Per-star recovery, which is what the mentor's three-part screen predicts.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import re
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet.injection import DualInjector

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
STAGED = os.path.join(REPO, "data", "search_frozen", "staged")
EVENTS = os.path.join(REPO, "data", "search_frozen", "out", "detected_events.txt")
DATA = os.path.join(REPO, "data")

BATCH_SEED = 20260804

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# A recorded event is only a TRANSIT if the planet actually crosses the stellar
# disc. ELC writes a row for every CONJUNCTION, including ones where the planet
# misses: those carry |impact parameter| >= 1 and ingress == egress, so their
# duration is ~2e-10 d.
#
# MEASURED 2026-08-04 on the 3-host pilot: 34 percent of all recorded events,
# and 57 PERCENT OF STAR-2 EVENTS, have |b| >= 1 and are not transits. Counting
# them inflated the geometric pair rate, depressed star 2's apparent recovery
# (a non-event cannot be recovered), and caused 59 fabricated 0.02 d dips to be
# injected into real light curves via the duration floor.
IMPACT_MAX = 1.0
MIN_REAL_DURATION_D = 1e-4
# A measured "transit" depth above this is a STELLAR ECLIPSE contaminating the
# window, not the planet. Injected planet depths are drawn in [1e-3, 1.5e-2] and
# the per-host depth law multiplies by ~1.0-1.2, so a real planet crossing
# cannot exceed ~2e-2. Stellar eclipses in these systems are 0.13 to 0.5.
#
# MEASURED 2026-08-04: without this guard, 63 of 1,725 ELC pairs (3.65%) had a
# partner "depth" of 0.13-0.14 while the lead was 0.0003-0.0007, giving depth
# ratios up to 2,771 and a fitted geometric factor whose top quantile was 43.
# The cause is that transit_depth takes min(flux) inside the transit window, and
# when a planet crossing coincides with a stellar eclipse the eclipse wins.
MAX_PLANET_DEPTH = 0.05


log = logging.getLogger("16_elc_batch")


def model_transits(model_dir):
    """Planet crossings with time, star, cycle, duration and measured depth."""
    model_path = os.path.join(model_dir, "modelU.linear")
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return []
    m = np.loadtxt(model_path)
    if m.ndim != 2 or m.shape[0] < 10:
        return []
    t, f = m[:, 0], m[:, 1]

    out = []
    for fn in sorted(os.listdir(model_dir)):
        mm = re.match(r"ELC(\d+)tran(\d+)time\.dat$", fn)
        if not mm:
            continue
        body, star = int(mm.group(1)), int(mm.group(2))
        if body < 3:
            continue
        path = os.path.join(model_dir, fn)
        if os.path.getsize(path) == 0:
            continue
        arr = np.atleast_2d(np.genfromtxt(path, usecols=range(6)))
        if arr.size == 0:
            continue
        for row in arr:
            t_ev, ingress, egress = float(row[1]), float(row[4]), float(row[5])
            if abs(float(row[3])) >= IMPACT_MAX:
                continue          # planet misses the disc: not a transit
            dur = egress - ingress
            if dur < MIN_REAL_DURATION_D:
                continue          # zero-duration record, same cause
            # NO 0.02 d floor. The floor fabricated a dip for every non-transit.
            inside = np.abs(t - t_ev) <= dur / 2.0
            near = (np.abs(t - t_ev) > dur) & (np.abs(t - t_ev) <= 3 * dur)
            if inside.sum() < 2 or near.sum() < 5:
                continue
            cont = np.median(f[near])
            if not np.isfinite(cont) or cont <= 0:
                continue
            depth = float(1.0 - np.min(f[inside]) / cont)
            if not np.isfinite(depth) or depth <= 0 or depth > MAX_PLANET_DEPTH:
                continue      # stellar eclipse in the window, not the planet
            out.append({"time": t_ev, "star": star, "cycle": int(row[0]),
                        "duration": dur, "depth": depth,
                        "impact": float(row[3])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--max-models", type=int, default=None)
    args = ap.parse_args()

    root = os.path.join(DATA, "elc_models", args.tag)
    manifest = pd.read_csv(os.path.join(root, "elc_manifest.csv"))
    manifest = manifest[manifest.qa_status == "PASS"]
    if args.max_models:
        manifest = manifest.head(args.max_models)
    log.info("ELC batch over %d QA-passed models", len(manifest))

    cat = load_catalogue(CAT_PATH, TEBC=True)
    real = pd.read_csv(EVENTS, sep=r"\s+")
    out_dir = os.path.join(DATA, "elc_batch", args.tag)
    snippet_dir = os.path.join(out_dir, "snippets")
    os.makedirs(snippet_dir, exist_ok=True)

    inj = DualInjector(
        os.path.join(DATA, "profile_bank_v2", "transit_models_v2.npz"),
        real_events_df=real, rng_seed=BATCH_SEED, catalogue=cat, TEBC=False,
        provenance="elc", run_id=f"elcbatch_{args.tag}", load_models=False)

    t0 = _time.time()
    n_models, n_skipped = 0, 0
    for _, row in manifest.iterrows():
        model_dir = os.path.join(root, f"TIC{int(row.tic)}_m{int(row.model_id.split('_')[1]):03d}")
        if not os.path.isdir(model_dir):
            n_skipped += 1
            continue
        transits = model_transits(model_dir)
        if not transits:
            n_skipped += 1
            continue
        # Inject into every cached sector of this host that overlaps the model
        # window; a transit outside a given sector simply lands on no data and
        # is dropped by inject_fixed_events.
        tic = int(row.tic)
        files = sorted(f for f in os.listdir(STAGED) if f.startswith(f"TIC_{tic}_"))
        for fname in files:
            inj.inject_fixed_events(
                os.path.join(STAGED, fname), transits,
                model_meta={"model_id": row.model_id, "sigma": float(row.sigma),
                            "sigma_bin": row.sigma_bin},
                snippet_dir=snippet_dir)
        n_models += 1

    ev, te = inj.events_frame(), inj.tests_frame()
    if not len(te):
        raise SystemExit("No ELC injections landed on data")
    ev.to_csv(os.path.join(out_dir, "events_all.csv"), index=False)
    te.to_csv(os.path.join(out_dir, "tests_all.csv"), index=False)

    # ---- the detected pair rate ----------------------------------------
    te["key"] = te.model_idx.astype(str) + "_" + te.tic.astype(str) + "_" + \
                te.sector.astype(str) + "_" + te.elc_cycle.astype(str)
    grp = te.groupby("key").agg(
        n_transits=("recovered", "size"), n_recovered=("recovered", "sum"),
        n_stars=("elc_star", "nunique"),
        stars_recovered=("elc_star", lambda s: s[te.loc[s.index, "recovered"] == 1].nunique()),
        sigma_bin=("sigma", "first"))
    geometric_pairs = grp[grp.n_stars >= 2]
    detected_pairs = geometric_pairs[geometric_pairs.stars_recovered >= 2]
    det_rate = len(detected_pairs) / max(len(geometric_pairs), 1)

    per_star = te.groupby("elc_star").recovered.agg(["size", "mean"])

    log.info("Injected %d transits across %d models (%d skipped) in %.1f s",
             len(te), n_models, n_skipped, _time.time() - t0)
    log.info("  overall per-transit recovery: %.4f", te.recovered.mean())
    for s, r in per_star.iterrows():
        log.info("    star %d: %d transits, recovery %.4f", int(s), int(r["size"]), r["mean"])
    log.info("  conjunctions with BOTH stars crossed (geometric): %d", len(geometric_pairs))
    log.info("  of those, BOTH crossings detected: %d  -> DETECTED PAIR RATE %.4f",
             len(detected_pairs), det_rate)
    log.info("  inversion rate: %.4f (target 0.5)", te.inverted_lc.mean())

    summary = {
        "tag": args.tag, "n_models": n_models, "n_skipped": n_skipped,
        "n_transits_injected": int(len(te)), "n_events_flagged": int(len(ev)),
        "n_positive_events": int((ev.label == 1).sum()),
        "per_transit_recovery": float(te.recovered.mean()),
        "per_star_recovery": {str(int(s)): {"n": int(r["size"]), "recovery": float(r["mean"])}
                              for s, r in per_star.iterrows()},
        "geometric_pairs": int(len(geometric_pairs)),
        "detected_pairs": int(len(detected_pairs)),
        "detected_pair_rate": float(det_rate),
        "inversion_rate": float(te.inverted_lc.mean()),
        "distinct_tics": int(te.tic.nunique()),
        "note": "DETECTED rates. Comparable with Kostov 2020b's observed 4-of-11 "
                "incidence; the census's geometric rates are not.",
    }
    with open(os.path.join(out_dir, "elc_batch_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote %s", os.path.join(out_dir, "elc_batch_summary.json"))


if __name__ == "__main__":
    main()
