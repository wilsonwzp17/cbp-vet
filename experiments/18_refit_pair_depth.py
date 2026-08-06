"""Refit the bank's pair depth-ratio model on the full ELC batch.

Why a refit was always going to be needed
-----------------------------------------
The partner-to-lead depth ratio has two parts:

    d2 / d1  =  (host surface-brightness ratio)  x  (a geometric factor)

The first part is physical and the catalogue measures it: for a detached
eclipsing binary both eclipses block the same area, the smaller star's disc, so
the eclipse depth ratio IS the surface-brightness ratio. That anchor was always
right.

The second part is the chord geometry, and it is the part that has to be
measured. The planet's path across the SMALLER star is often grazing while its
path across the larger one is not, so the factor spans orders of magnitude.

History of this model, kept so the change is auditable:

1. `lognormal(0, 0.15)` clipped `[0.5, 2.0]`. Invented, never validated, could
   not reach below 0.47. Every synthetic partner was far easier to detect than
   reality.
2. Refit on **30 pairs from 3 hosts**. Overshot in the other direction: the
   census then measured ELC 0.3526 against the bank's 0.1830, off 92.7 percent.
3. This refit: after the grazing and stellar-eclipse guards, **1,399 pairs
   across 124 hosts** survive (of 150 hosts generated; 26 contribute no clean
   cross-star pair) — 47x the sample of the 30-pair fit.

The quantity fitted is the geometric factor alone, `g = (d2/d1) / host_ratio`,
so the physical anchor stays where it belongs and only the measured part is
resampled. Grazing non-transits are excluded: ELC writes a row for every
conjunction including ones where the planet misses the disc, and counting those
is what produced the retracted pair-rate error.
"""

import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
DATA = os.path.join(REPO, "data")
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

N_QUANTILES = 40

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("18_refit")


def transit_depth(t, f, ev):
    dur = max(ev["egress"] - ev["ingress"], 1e-6)
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


def collect(tag="full"):
    root = os.path.join(DATA, "elc_models", tag)
    manifest = pd.read_csv(os.path.join(root, "elc_manifest.csv"))
    manifest = manifest[manifest.qa_status == "PASS"]
    noise = pd.read_csv(os.path.join(DATA, "noise_screen.csv")).set_index("tic")

    rows = []
    for _, r in manifest.iterrows():
        d = os.path.join(root, f"TIC{int(r.tic)}_m{int(r.model_id.split('_')[1]):03d}")
        mp = os.path.join(d, "modelU.linear")
        if not os.path.isdir(d) or not os.path.exists(mp) or os.path.getsize(mp) == 0:
            continue
        m = np.loadtxt(mp)
        if m.ndim != 2 or m.shape[0] < 10:
            continue
        t, f = m[:, 0], m[:, 1]

        by_cycle = {}
        for fn in os.listdir(d):
            mm = re.match(r"ELC(\d+)tran(\d+)time\.dat$", fn)
            if not mm or int(mm.group(1)) < 3:
                continue
            path = os.path.join(d, fn)
            if os.path.getsize(path) == 0:
                continue
            arr = np.atleast_2d(np.genfromtxt(path, usecols=range(6)))
            if arr.size == 0:
                continue
            for row in arr:
                if abs(float(row[3])) >= IMPACT_MAX:      # planet misses the disc
                    continue
                if float(row[5]) - float(row[4]) < MIN_REAL_DURATION_D:
                    continue
                by_cycle.setdefault(int(row[0]), []).append(
                    {"star": int(mm.group(2)), "time": float(row[1]),
                     "ingress": float(row[4]), "egress": float(row[5])})

        host_ratio = float(noise.loc[int(r.tic), "depth_ratio"]) if int(r.tic) in noise.index else np.nan
        if not np.isfinite(host_ratio) or host_ratio <= 0:
            continue
        for cyc, evs in by_cycle.items():
            s1 = [e for e in evs if e["star"] == 1]
            s2 = [e for e in evs if e["star"] == 2]
            if not s1 or not s2:
                continue
            d1, d2 = transit_depth(t, f, s1[0]), transit_depth(t, f, s2[0])
            if not (np.isfinite(d1) and np.isfinite(d2)) or d1 <= 0 or d2 <= 0:
                continue
            rows.append({"tic": int(r.tic), "cycle": cyc, "d1": d1, "d2": d2,
                         "ratio": d2 / d1, "host_ratio": host_ratio,
                         "g": (d2 / d1) / host_ratio})
    return pd.DataFrame(rows)


def main():
    df = collect("full")
    log.info("Collected %d real ELC pairs across %d hosts", len(df), df.tic.nunique())
    if len(df) < 100:
        raise SystemExit("Too few pairs to refit")

    g = df.g.to_numpy()
    g = g[np.isfinite(g) & (g > 0)]
    # Winsorise the top 1 percent. g > 1 is real physics (a grazing LEAD crossing
    # gives a small d1, so d2/d1 is large), but the sampler's output is clipped
    # to DEPTH_RATIO_CLIP = (1e-3, 2.0) anyway, so any g above ~2/host_ratio is
    # functionally identical to the clip. Carrying a raw p98.75 of 11.2 in the
    # pinned array would only look alarming without changing a single draw.
    g_cap = float(np.percentile(g, 99))
    n_capped = int((g > g_cap).sum())
    g = np.minimum(g, g_cap)
    log.info("winsorised %d of %d values at p99 = %.4f", n_capped, len(g), g_cap)
    qs = np.linspace(0.5 / N_QUANTILES, 1 - 0.5 / N_QUANTILES, N_QUANTILES)
    quant = np.quantile(g, qs)

    log.info("geometric factor g = (d2/d1) / host_depth_ratio, n=%d", len(g))
    for p in (5, 10, 25, 50, 75, 90, 95):
        log.info("   p%-3d %.5f", p, float(np.percentile(g, p)))
    log.info("observed d2/d1 directly: p10 %.4f  p50 %.4f  p90 %.4f",
             *np.percentile(df.ratio, [10, 50, 90]))

    out = {
        "n_pairs": int(len(df)), "n_hosts": int(df.tic.nunique()),
        "n_quantiles": N_QUANTILES,
        "quantiles": [round(float(x), 6) for x in quant],
        "g_percentiles": {f"p{p}": float(np.percentile(g, p))
                          for p in (5, 10, 25, 50, 75, 90, 95)},
        "ratio_percentiles": {f"p{p}": float(np.percentile(df.ratio, p))
                              for p in (10, 50, 90)},
        "supersedes": "30-pair 3-host fit of 2026-08-04 which overshot "
                      "(census measured ELC 0.3526 vs bank 0.1830, 92.7% off)",
        "grazing_excluded": True,
        "stellar_eclipse_guard": "MAX_PLANET_DEPTH = 0.05",
        "winsorised_at_p99": g_cap,
        "n_winsorised": n_capped,
    }
    path = os.path.join(DATA, "elc_models", "pair_depth_ratio_fit.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    df.to_csv(os.path.join(DATA, "elc_models", "pair_depth_pairs_full.csv"), index=False)
    log.info("Wrote %s", path)

    print("\nELC_GEOMETRIC_FACTOR = np.array([")
    for i in range(0, N_QUANTILES, 8):
        print("    " + ", ".join(f"{x:.5f}" for x in quant[i:i + 8]) + ",")
    print("])")


if __name__ == "__main__":
    main()
