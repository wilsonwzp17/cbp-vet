"""Per-system out-of-eclipse noise + host-eligibility screen (W3.3c).

The measurement the mentor specified on 2026-07-29: mask the eclipses, fit a
smooth polynomial, normalize, take a robust RMS. Runs over the frozen
1,946-file TEBC-snapshot cache, aggregates per system, and joins the catalog
eclipse depths so the three-part host screen can be counted at any threshold.

Outputs data/noise_screen.csv with one row per system:
  tic, n_files, rms_cadence (median over files, robust), prim_depth, sec_depth,
  depth_ratio (sec/prim, 2g fit; polyfit fallback), prim_pos, sec_pos,
plus a printed summary of screen counts at a grid of thresholds. The specific
threshold that gates E2 is pinned in the freeze manifest, not here.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "lc_cache"
CAT = Path.home() / "mono-cbp" / "catalogues" / "TEBC_morph_05_P_7.csv"
OUT = REPO / "data" / "noise_screen.csv"

MASK_WIDTH_FACTOR = 1.6   # mask phases within this multiple of the catalog half-width
MIN_OOE_POINTS = 200      # skip files with fewer usable out-of-eclipse points
POLY_ORDER = 2            # the mentor's "fit a quadratic or something"
CHUNK_GAP_D = 0.75        # a gap longer than this splits the file into chunks
MIN_CHUNK_POINTS = 24     # skip chunks too short to detrend


def robust_rms(x: np.ndarray) -> float:
    """1.4826 * MAD: insensitive to residual eclipse points and flares."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def file_rms(path: Path, prim_pos, prim_w, sec_pos, sec_w) -> float | None:
    """Per-chunk quadratic detrend (v2, the estimator of record per Amendment-1):
    detrending locally per contiguous chunk tracks the slow trends the pipeline's
    own detrender would remove, so the residual scatter is the noise the search
    actually fights, not the file-scale trend."""
    d = np.loadtxt(path)
    if d.ndim != 2 or d.shape[0] < MIN_OOE_POINTS:
        return None
    t, f, ph = d[:, 0], d[:, 1], d[:, 3]
    keep = np.ones(len(t), bool)
    for pos, w in ((prim_pos, prim_w), (sec_pos, sec_w)):
        if pos is None or not np.isfinite(pos):
            continue
        half = MASK_WIDTH_FACTOR * (w if (w and np.isfinite(w) and w > 0) else 0.02)
        dist = np.abs(((ph - pos) + 0.5) % 1.0 - 0.5)
        keep &= dist > half
    if keep.sum() < MIN_OOE_POINTS:
        return None
    tt, ff = t[keep], f[keep]
    med = np.median(ff)
    if med <= 0:
        return None
    # split into contiguous chunks at gaps, detrend each with a quadratic
    edges = np.where(np.diff(tt) > CHUNK_GAP_D)[0] + 1
    resids = []
    for chunk_t, chunk_f in zip(np.split(tt, edges), np.split(ff, edges)):
        if len(chunk_t) < MIN_CHUNK_POINTS:
            continue
        c = np.polyfit(chunk_t - chunk_t.mean(), chunk_f, POLY_ORDER)
        resids.append(chunk_f - np.polyval(c, chunk_t - chunk_t.mean()))
    if not resids:
        return None
    return float(robust_rms(np.concatenate(resids) / med))


def main() -> None:
    cat = pd.read_csv(CAT)
    cat = cat.drop_duplicates(subset="tess_id", keep="first").set_index("tess_id")

    files = defaultdict(list)
    for p in sorted(CACHE.glob("TIC_*.txt")):
        m = re.match(r"TIC_(\d+)_(\d+)\.txt", p.name)
        if m:
            files[int(m.group(1))].append(p)

    rows = []
    for tic, paths in files.items():
        if tic not in cat.index:
            continue
        r = cat.loc[tic]
        # depths: prefer the double-Gaussian fit, fall back to polyfit
        pd_2g, sd_2g = r.get("prim_depth_2g"), r.get("sec_depth_2g")
        prim_depth = pd_2g if np.isfinite(pd_2g) else r.get("prim_depth_pf")
        sec_depth = sd_2g if (sd_2g is not None and np.isfinite(sd_2g)) else r.get("sec_depth_pf")
        rmss = []
        for p in paths:
            try:
                v = file_rms(p, r.get("prim_pos_2g"), r.get("prim_width_2g"),
                             r.get("sec_pos_2g"), r.get("sec_width_2g"))
            except Exception:
                v = None
            if v is not None:
                rmss.append(v)
        if not rmss:
            continue
        ratio = (sec_depth / prim_depth) if (sec_depth and prim_depth and prim_depth > 0
                                             and np.isfinite(sec_depth)) else np.nan
        rows.append(dict(tic=tic, n_files=len(rmss), rms_cadence=float(np.median(rmss)),
                         prim_depth=prim_depth, sec_depth=sec_depth, depth_ratio=ratio,
                         prim_pos=r.get("prim_pos_2g"), sec_pos=r.get("sec_pos_2g")))

    df = pd.DataFrame(rows).sort_values("tic")
    df.to_csv(OUT, index=False)

    n = len(df)
    meas = df[np.isfinite(df.depth_ratio)]
    print(f"systems measured: {n}  (with measurable secondary: {len(meas)})")
    print(f"rms_cadence: median {df.rms_cadence.median()*100:.3f}%  "
          f"p25 {df.rms_cadence.quantile(.25)*100:.3f}%  p75 {df.rms_cadence.quantile(.75)*100:.3f}%")
    # transit-scale noise credit: a ~6 h transit at 30-min cadence is ~12 points
    sqrt_n = np.sqrt(12.0)
    print("\nscreen counts (provisional thresholds; manifest pins the final rule)")
    print("ref_depth = assumed primary-transit depth; partner depth = ratio * ref_depth")
    for ref_depth in (0.005, 0.003):
        for snr_min in (5.0, 3.0):
            det = meas[(meas.depth_ratio * ref_depth) /
                       (meas.rms_cadence / sqrt_n) >= snr_min]
            phys = det[det.depth_ratio >= 0.33]
            strong = phys[phys.prim_depth >= 0.30]
            print(f"  ref {ref_depth*100:.1f}%  SNR>={snr_min:.0f}:  "
                  f"partner-detectable {len(det):4d}   N_physical {len(phys):4d}   "
                  f"N_strong {len(strong):4d}")


if __name__ == "__main__":
    main()
