"""The calibration census: does the synthetic bank's pair model match real geometry?

What this answers
-----------------
The bank injects a partner transit at a rate, a spacing, and a depth ratio that
we CHOSE, from a model fitted to twelve observed spacings. ELC, by contrast,
integrates the actual orbits and reports exactly when each star was crossed. So
ELC is the ground truth against which the bank's assumptions can be checked.

Three comparisons, each with a pre-named threshold (Execution-Readiness B5):

    pair rate      off by more than 10 percentage points absolute
    spacing        KS p < 0.01 with the median off by more than 20 percent
    depth ratio    median off by more than 30 percent

Any trip fires the pre-named pair-subset re-injection rather than a debate.

How a conjunction is counted
----------------------------
``iwriteeclipse`` writes one file per (body, star) pair, and each row carries a
CYCLE number. Rows sharing a cycle belong to the same conjunction, so:

    conjunction  = one cycle number, across all star files
    pair         = a conjunction with at least one crossing of EACH star, whose
                   minimum cross-star separation lies inside the per-system
                   label window [max(0.15 d, duration), 0.5 x P_bin]
    same-star multiple = two or more crossings of the SAME star in one cycle,
                   which is Kostov 2020b equation (2)'s distinct signature

An important scope note, learned the hard way
---------------------------------------------
These are GEOMETRIC rates. ELC records a crossing whether or not it would be
detectable. Kostov 2020b's observed 4-of-11 incidence is a DETECTED rate, and
the two must never be compared directly: a partner can cross geometrically and
be far too shallow to find. The detected rate requires running these models
through injection and search, which is a separate step.

The bank's own PAIR_RATE is a generation choice about geometry, so comparing it
against the geometric census IS well posed. Its cited source, Chen and Kipping
Scenario A at sigma = 1e-3, gives f2 = 0.367, and f2 is itself a geometric
probability. But C&K do NOT condition on the binary being eclipsing, whereas
every host here is an eclipsing binary and therefore edge-on by selection, which
makes a double crossing far more likely. That is the hypothesis this census
tests.
"""

import argparse
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cbpvet.injection import pair_model

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
DATA = os.path.join(REPO, "data")

# Pre-named disagreement thresholds (Execution-Readiness B5).
THRESH_RATE_PP = 0.10
THRESH_SPACING_KS_P = 0.01
THRESH_SPACING_MEDIAN_FRAC = 0.20
THRESH_RATIO_MEDIAN_FRAC = 0.30

# The bank's assumed pair rate, for the record.
BANK_PAIR_RATE = 0.37
CK_F2_AT_1E3 = 0.367

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

MIN_REAL_DURATION_D = 1e-4

log = logging.getLogger("12_census")


def read_tran_files(model_dir):
    """All planet crossings in one model, with star identity and cycle."""
    events = []
    for fn in sorted(os.listdir(model_dir)):
        m = re.match(r"ELC(\d+)tran(\d+)time\.dat$", fn)
        if not m:
            continue
        body, star = int(m.group(1)), int(m.group(2))
        if body < 3:                       # bodies 1 and 2 are the stars
            continue
        path = os.path.join(model_dir, fn)
        if os.path.getsize(path) == 0:
            continue
        # The trailing UTC string breaks loadtxt; take the six numeric columns.
        arr = np.atleast_2d(np.genfromtxt(path, usecols=range(6)))
        if arr.size == 0:
            continue
        for row in arr:
            if not np.isfinite(row[1]):
                continue
            b = abs(float(row[3]))
            dur = float(row[5]) - float(row[4])
            if b >= IMPACT_MAX or dur < MIN_REAL_DURATION_D:
                continue          # conjunction without a disc crossing
            events.append({"body": body, "star": star, "cycle": int(row[0]),
                           "time": float(row[1]), "impact": b,
                           "ingress": float(row[4]), "egress": float(row[5])})
    return events


def transit_depth(model_dir, ev):
    """Depth of one crossing, against a local continuum outside it."""
    path = os.path.join(model_dir, "modelU.linear")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.nan
    m = np.loadtxt(path)
    if m.ndim != 2:
        return np.nan
    t, f = m[:, 0], m[:, 1]
    dur = max(ev["egress"] - ev["ingress"], 1e-3)
    inside = np.abs(t - ev["time"]) <= dur / 2.0
    near = (np.abs(t - ev["time"]) > dur) & (np.abs(t - ev["time"]) <= 3 * dur)
    if inside.sum() < 2 or near.sum() < 5:
        return np.nan
    cont = np.median(f[near])
    if not np.isfinite(cont) or cont <= 0:
        return np.nan
    d = float(1.0 - np.min(f[inside]) / cont)
    if not np.isfinite(d) or d <= 0 or d > MAX_PLANET_DEPTH:
        return np.nan          # stellar eclipse in the window, not the planet
    return d


def census(model_root, manifest):
    conjunctions, pairs, spacings, ratios, same_star = [], [], [], [], []
    for _, row in manifest.iterrows():
        if row.get("qa_status") not in ("PASS",):
            continue
        model_dir = os.path.join(model_root, f"TIC{int(row.tic)}_m{int(row.model_id.split('_')[1]):03d}")
        if not os.path.isdir(model_dir):
            continue
        events = read_tran_files(model_dir)
        if not events:
            continue
        p_bin = float(row.p_bin)
        by_cycle = {}
        for e in events:
            by_cycle.setdefault(e["cycle"], []).append(e)

        for cyc, evs in by_cycle.items():
            stars = {e["star"] for e in evs}
            rec = {"tic": int(row.tic), "model_id": row.model_id,
                   "sigma": float(row.sigma), "sigma_bin": row.sigma_bin,
                   "cycle": cyc, "n_events": len(evs), "n_stars": len(stars)}
            # same-star multiple: two or more crossings of one star in a cycle
            for s in stars:
                if sum(1 for e in evs if e["star"] == s) >= 2:
                    rec["same_star_multiple"] = 1
                    same_star.append(rec | {"star": s})
                    break
            else:
                rec["same_star_multiple"] = 0

            is_pair = 0
            if len(stars) >= 2:
                s1 = [e for e in evs if e["star"] == 1]
                s2 = [e for e in evs if e["star"] == 2]
                if s1 and s2:
                    dts = [abs(a["time"] - b["time"]) for a in s1 for b in s2]
                    dt = min(dts)
                    dur = np.median([e["egress"] - e["ingress"] for e in evs])
                    lo, hi = pair_model.label_window(p_bin, dur)
                    if hi > lo and lo <= dt <= hi:
                        is_pair = 1
                        spacings.append({"dt": dt, "dt_over_pbin": dt / p_bin,
                                         "p_bin": p_bin, "sigma_bin": row.sigma_bin})
                        d1 = transit_depth(model_dir, s1[0])
                        d2 = transit_depth(model_dir, s2[0])
                        if np.isfinite(d1) and np.isfinite(d2) and d1 > 0:
                            ratios.append({"ratio": d2 / d1, "tic": int(row.tic),
                                           "sigma_bin": row.sigma_bin})
                    else:
                        rec["outside_window"] = 1
            rec["is_pair"] = is_pair
            conjunctions.append(rec)
            if is_pair:
                pairs.append(rec)
    return (pd.DataFrame(conjunctions), pd.DataFrame(spacings),
            pd.DataFrame(ratios), pd.DataFrame(same_star))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    root = os.path.join(DATA, "elc_models", args.tag)
    manifest = pd.read_csv(os.path.join(root, "elc_manifest.csv"))
    conj, spac, rat, same = census(root, manifest)
    if not len(conj):
        raise SystemExit("No conjunctions found; nothing to census")

    rate = float(conj.is_pair.mean())
    log.info("Conjunctions with >= 1 crossing: %d over %d models",
             len(conj), manifest.qa_status.eq("PASS").sum())
    log.info("  GEOMETRIC pair rate: %.4f", rate)
    for b, g in conj.groupby("sigma_bin"):
        log.info("    sigma %-5s n=%4d  pair rate %.4f  same-star multiples %.4f",
                 b, len(g), g.is_pair.mean(), g.same_star_multiple.mean())
    log.info("  same-star multiples overall: %.4f", float(conj.same_star_multiple.mean()))

    # ---- comparison 1: pair rate ---------------------------------------
    d_rate = abs(rate - BANK_PAIR_RATE)
    trip_rate = d_rate > THRESH_RATE_PP

    # ---- comparison 2: spacing, against the bank's own sampler ----------
    rng = np.random.default_rng(args.seed)
    ks_p, med_frac, trip_spacing = np.nan, np.nan, False
    if len(spac) >= 8:
        drawn = []
        for _, r in spac.iterrows():
            # Draw what the bank WOULD have used for a system like this one.
            v = pair_model.sample_dt_over_pbin(rng, np.log10(20.0), size=1)[0]
            drawn.append(v)
        drawn = np.asarray(drawn)
        obs = spac.dt_over_pbin.to_numpy()
        ks = stats.ks_2samp(obs, drawn)
        ks_p = float(ks.pvalue)
        med_frac = float(abs(np.median(obs) - np.median(drawn)) / max(np.median(drawn), 1e-9))
        trip_spacing = bool(ks_p < THRESH_SPACING_KS_P and med_frac > THRESH_SPACING_MEDIAN_FRAC)
        log.info("  spacing dt/P_bin: ELC median %.4f, bank median %.4f, "
                 "KS p=%.3g, median off %.1f%%",
                 np.median(obs), np.median(drawn), ks_p, 100 * med_frac)

    # ---- comparison 3: depth ratio -------------------------------------
    # Compare against what the BANK would have drawn for these same hosts, not
    # against 1.0. Comparing to 1.0 silently assumes identical stars and makes
    # the threshold meaningless: it was my own bug, caught 2026-08-04.
    ratio_frac, trip_ratio, bank_med = np.nan, False, np.nan
    if len(rat) >= 5:
        obs_r = rat.ratio.to_numpy()
        obs_r = obs_r[np.isfinite(obs_r)]
        noise = pd.read_csv(os.path.join(DATA, "noise_screen.csv")).set_index("tic")
        drawn = []
        for _, rr in rat.iterrows():
            t = int(rr.tic)
            dr = float(noise.loc[t, "depth_ratio"]) if t in noise.index else 1.0
            drawn.append(float(pair_model.pair_depth_ratio(rng, dr, size=1)[0]))
        drawn = np.asarray(drawn)
        if len(obs_r):
            bank_med = float(np.median(drawn))
            ratio_frac = float(abs(np.median(obs_r) - bank_med) / max(bank_med, 1e-9))
            trip_ratio = ratio_frac > THRESH_RATIO_MEDIAN_FRAC
            log.info("  depth ratio star2/star1: ELC median %.4f, bank median %.4f, "
                     "off %.1f%% over %d pairs",
                     np.median(obs_r), bank_med, 100 * ratio_frac, len(obs_r))
            log.info("    ELC quantiles p10/p50/p90: %.3f / %.3f / %.3f",
                     *np.percentile(obs_r, [10, 50, 90]))
            log.info("    bank quantiles p10/p50/p90: %.3f / %.3f / %.3f",
                     *np.percentile(drawn, [10, 50, 90]))

    trips = {"pair_rate": bool(trip_rate), "spacing": bool(trip_spacing),
             "depth_ratio": bool(trip_ratio)}
    out = {
        "tag": args.tag,
        "n_models_passed": int(manifest.qa_status.eq("PASS").sum()),
        "n_conjunctions": int(len(conj)),
        "n_pairs": int(len(spac)),
        "geometric_pair_rate": rate,
        "pair_rate_by_sigma_bin": {str(b): float(g.is_pair.mean())
                                   for b, g in conj.groupby("sigma_bin")},
        "same_star_multiple_rate": float(conj.same_star_multiple.mean()),
        "bank_assumed_pair_rate": BANK_PAIR_RATE,
        "pair_rate_delta": d_rate,
        "spacing_ks_p": ks_p,
        "spacing_median_frac_off": med_frac,
        "depth_ratio_frac_off_from_bank": ratio_frac,
        "depth_ratio_bank_median": bank_med,
        "depth_ratio_elc_median": float(np.median(rat.ratio.to_numpy())) if len(rat) else None,
        "thresholds": {"rate_pp": THRESH_RATE_PP, "ks_p": THRESH_SPACING_KS_P,
                       "spacing_median_frac": THRESH_SPACING_MEDIAN_FRAC,
                       "ratio_median_frac": THRESH_RATIO_MEDIAN_FRAC},
        "trips": trips,
        "any_trip": any(trips.values()),
        "scope": "GEOMETRIC rates from ELC tran files. NOT comparable to Kostov "
                 "2020b's observed 4-of-11 detected incidence; the detected rate "
                 "requires running these models through injection and search.",
    }
    path = os.path.join(DATA, "elc_models", "census_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    conj.to_csv(os.path.join(DATA, "elc_models", "census_conjunctions.csv"), index=False)
    log.info("Wrote %s", path)

    print("\n" + "=" * 72)
    print("CALIBRATION CENSUS (geometric)")
    print("=" * 72)
    print(f"  ELC geometric pair rate      : {rate:.4f}  over {len(conj)} conjunctions")
    print(f"  bank assumed PAIR_RATE       : {BANK_PAIR_RATE:.4f}"
          f"  (cited: C&K Scenario A f2(1e-3) = {CK_F2_AT_1E3})")
    print(f"  |delta|                      : {d_rate:.4f}   threshold {THRESH_RATE_PP}")
    for k, v in trips.items():
        print(f"    {k:12s} {'TRIP' if v else 'ok'}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
