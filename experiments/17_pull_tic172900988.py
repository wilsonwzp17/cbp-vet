"""B10: pull TIC 172900988, the second TESS circumbinary planet.

Why it matters
--------------
The one-shot's TESS arm is tiny. Kostov 2020b's own accounting gives about eight
detectable TESS events: six from TOI-1338 b and two from TIC 172900988 b. Losing
the second host drops the pool from ~8 to 6, which on counts that small is a
material loss of the only TESS evidence we have.

The problem: **TIC 172900988 has zero cached light curves** and is absent from
the frozen TEBC catalogue, so nothing in the pipeline can touch it. Verified
twice: the search flagged 0 events for it, and the sealed pass covered only
TOI-1338 and the mono-cbp candidate.

The ephemeris, and where it comes from
--------------------------------------
Read directly from Kostov et al. 2021 (arXiv:2105.08614, in ``papers/``):

    "it was not difficult to find a suitable ephemeris (P = 19.658025,
     T0 = 3878.33860, where the time convention is BJD - 2,455,000)"

TESS times are BTJD = BJD - 2,457,000, so **T0 = 1878.33860 BTJD**, which falls
inside sector 21 and matches the paper's title, "Detected in One Sector".

Note that mono-cbp's Table 2 rounds the period to 19.70 d. The paper's
19.658025 is used here, because a 0.04 d error accumulates to a phase error of
0.002 per cycle and the eclipse mask is only ~0.03 wide in phase.

Eclipse positions and widths are MEASURED from the phase-folded light curve
rather than derived from the paper's MCMC parameters. The paper reports
``sqrt(e) cos w`` style parameters across six competing photodynamical
solutions, and propagating those into a phase offset compounds several
uncertainties when the quantity we actually need, where the eclipses sit in
phase and how wide they are, is directly visible in the data.
"""

import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import time_to_phase

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
DATA = os.path.join(REPO, "data")
STAGING = os.path.join(DATA, "tic172900988")

TIC = 172900988
# Kostov et al. 2021, section 3; time convention BJD - 2,455,000.
P_BIN = 19.658025
T0_BJD_2455000 = 3878.33860
T0_BTJD = T0_BJD_2455000 - 2000.0     # BTJD = BJD - 2,457,000
Q_BIN = 0.97                          # mono-cbp Table 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("17_pull")


def download(sectors=None):
    import lightkurve as lk
    os.makedirs(STAGING, exist_ok=True)
    res = lk.search_lightcurve(f"TIC {TIC}", mission="TESS", author="TESS-SPOC")
    log.info("MAST offers %d TESS-SPOC products", len(res))
    got = []
    for row in res:
        sec = int(row.table["sequence_number"][0]) if hasattr(row, "table") else None
        try:
            lc = row.download(flux_column="sap_flux", quality_bitmask="hard")
        except Exception as exc:
            log.warning("  sector %s download failed: %s", sec, type(exc).__name__)
            continue
        if lc is None:
            continue
        sector = int(lc.meta.get("SECTOR", sec or -1))
        t = np.asarray(lc.time.value, float)
        f = np.asarray(lc.flux.value, float)
        fe = (np.asarray(lc.flux_err.value, float)
              if lc.flux_err is not None else np.full_like(f, np.nan))
        # mono-cbp's crowding correction, the same one the frozen cache used.
        crowd = lc.meta.get("CROWDSAP")
        flfr = lc.meta.get("FLFRCSAP")
        if crowd is not None and flfr is not None:
            med = np.nanmedian(f)
            f = (f - med * (1 - crowd)) / flfr
            fe = fe / flfr
        med = np.nanmedian(f)
        f, fe = f / med, fe / med
        ph = time_to_phase(t, period=P_BIN, t0=T0_BTJD)
        keep = ~np.isnan(t * f * ph)
        arr = np.column_stack([t[keep], f[keep], fe[keep], ph[keep]])
        path = os.path.join(STAGING, f"TIC_{TIC}_{sector:02d}.txt")
        np.savetxt(path, arr, header="TIME FLUX FLUX_ERR PHASE ECL_MASK")
        got.append({"sector": sector, "n": int(keep.sum()),
                    "t0": float(t[keep].min()), "t1": float(t[keep].max()),
                    "crowdsap": crowd, "flfrcsap": flfr, "path": path})
        log.info("  sector %2d: %6d cadences, BTJD %.2f to %.2f, CROWDSAP %s",
                 sector, int(keep.sum()), t[keep].min(), t[keep].max(), crowd)
    return got


def measure_eclipses():
    """Find the two eclipses in the phase-folded light curve."""
    files = sorted(f for f in os.listdir(STAGING) if f.endswith(".txt"))
    ph, fl = [], []
    for f in files:
        a = np.loadtxt(os.path.join(STAGING, f), skiprows=1)
        if a.ndim == 2 and a.shape[0] > 10:
            ph.append(a[:, 3]); fl.append(a[:, 1])
    ph, fl = np.concatenate(ph), np.concatenate(fl)
    nb = 400
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nb - 1)
    prof = np.array([np.median(fl[idx == b]) if (idx == b).any() else np.nan
                     for b in range(nb)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    base = np.nanmedian(prof)
    depth = base - prof

    # Primary: the deepest bin. By construction of T0 it should sit near phase 0.
    i1 = int(np.nanargmax(depth))
    prim_pos = float(centres[i1])
    prim_depth = float(depth[i1])
    # Secondary: the deepest bin at least 0.15 in phase away, wrapped.
    d = np.abs(((centres - prim_pos + 0.5) % 1.0) - 0.5)
    far = d > 0.15
    i2 = int(np.nanargmax(np.where(far, depth, -np.inf)))
    sec_pos = float(centres[i2])
    sec_depth = float(depth[i2])

    def width_at(i, dep, frac=0.10):
        """Full width where the dip still exceeds ``frac`` of its peak depth.

        NOT half-max. Half-max biases against the DEEPER eclipse: a deeper dip
        has a higher half-max threshold, so it measures narrower even when the
        two eclipses last the same time. Measured here 2026-08-04: half-max gave
        primary 0.0025 against secondary 0.0150, when simple geometry
        ((R1 + R2) / (pi a)) x P predicts about 0.0155 for both.

        This width feeds the eclipse MASK, and an under-wide mask leaves eclipse
        wings unmasked, which the search then flags as candidate events. That is
        mono-cbp's single largest false-positive class, 1,084 of its 1,647, so
        erring wide is the safe direction.
        """
        thr = dep * frac
        # Walk with WRAPAROUND. Phase is a circle, and this binary's primary
        # eclipse sits at phase 0.9988, i.e. the last bin of 400. A non-wrapping
        # walk stops dead at the array edge and reports a truncated width:
        # measured 2026-08-04, that gave primary 0.0050 against secondary
        # 0.0250 when geometry predicts both near 0.0154. The primary is the
        # DEEPEST eclipse in the data, so under-measuring its mask is the worst
        # possible place for this bug.
        n_left = 0
        while n_left < nb // 2:
            j = (i - n_left - 1) % nb
            if not (np.isfinite(depth[j]) and depth[j] > thr):
                break
            n_left += 1
        n_right = 0
        while n_right < nb // 2:
            j = (i + n_right + 1) % nb
            if not (np.isfinite(depth[j]) and depth[j] > thr):
                break
            n_right += 1
        return float((n_left + n_right + 1) / nb)

    return {
        "prim_pos": prim_pos, "prim_width": width_at(i1, prim_depth),
        "prim_depth": prim_depth,
        "sec_pos": sec_pos, "sec_width": width_at(i2, sec_depth),
        "sec_depth": sec_depth,
        "depth_ratio": float(sec_depth / prim_depth) if prim_depth > 0 else np.nan,
        "n_cadences": int(len(ph)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    got = [] if args.skip_download else download()
    if not os.path.isdir(STAGING) or not os.listdir(STAGING):
        log.error("PULL FAILED and no staged files exist. Pre-named fallback: seal "
                  "the TOI-1338-only pass and record the reduced scope; the TESS "
                  "D1 pool drops from ~8 to 6 events as a logged tier degradation.")
        raise SystemExit(1)

    ecl = measure_eclipses()
    log.info("Measured eclipses: primary phase %.4f width %.4f depth %.5f",
             ecl["prim_pos"], ecl["prim_width"], ecl["prim_depth"])
    log.info("                   secondary phase %.4f width %.4f depth %.5f",
             ecl["sec_pos"], ecl["sec_width"], ecl["sec_depth"])
    log.info("                   depth ratio %.4f", ecl["depth_ratio"])
    if abs(((ecl["prim_pos"] + 0.5) % 1.0) - 0.5) > 0.05:
        log.warning("Primary eclipse is %.4f in phase, not near 0. The ephemeris "
                    "or the time convention may be off.", ecl["prim_pos"])

    row = {"tess_id": TIC, "period": P_BIN, "bjd0": T0_BTJD,
           "prim_pos": ecl["prim_pos"], "prim_width": ecl["prim_width"],
           "sec_pos": ecl["sec_pos"], "sec_width": ecl["sec_width"],
           "Tmag": np.nan, "morph_coeff": np.nan, "q_bin": Q_BIN,
           "source": "Kostov et al. 2021 ephemeris; eclipses measured from data"}
    pd.DataFrame([row]).to_csv(os.path.join(DATA, "catalogue_tic172900988.csv"), index=False)

    with open(os.path.join(DATA, "tic172900988_pull.json"), "w") as fh:
        json.dump({"tic": TIC, "period": P_BIN, "t0_btjd": T0_BTJD,
                   "t0_bjd_2455000": T0_BJD_2455000, "q_bin": Q_BIN,
                   "sectors": got, "eclipses": ecl,
                   "ephemeris_source": "Kostov et al. 2021 (arXiv:2105.08614), "
                                       "section 3: P = 19.658025, T0 = 3878.33860 "
                                       "in BJD - 2,455,000",
                   "note": "mono-cbp Table 2 rounds the period to 19.70 d; the "
                           "paper value is used because a 0.04 d error moves the "
                           "phase by 0.002 per cycle against a ~0.03-wide mask"},
                  fh, indent=2, default=str)
    log.info("Wrote the supplemental catalogue row and the pull record.")


if __name__ == "__main__":
    main()
