#!/usr/bin/env python
"""
03_pull_goodsn_pilot.py -- pull and plot the best-characterized good-S/N EBs.

Starting set: the good-S/N systems from the validated-EB target list, piloted
on the 28 that also have TEBC eclipse ephemerides and Gaia DR3 orbital
solutions (best-characterized first).

Cache-aware: earlier sectors are already in data/lc_cache, so this script
queries MAST for the current sector list per system and downloads only what
the cache lacks, which is mostly the newer sectors (~60+).

Per system it produces:
  - new cache files (TESS-SPOC via mono-cbp's own lc_to_txt recipe;
    QLP-only sectors go to a separate lc_cache_qlp/ so the mono-cbp cache stays pure)
  - one PNG per sector + one stitched overview PNG (data/goodsn_pilot/plots/)
  - a row in the status CSV (data/goodsn_pilot/pilot_status.csv)

Usage:
    conda activate mono-cbp
    python experiments/03_pull_goodsn_pilot.py            # the 28 best-characterized
    python experiments/03_pull_goodsn_pilot.py --tics data/goodsn_in_tebc.txt
Env overrides: MONO_CBP_DIR, CBP_DATA_DIR (as in 01_download_lightcurves.py).
"""
from __future__ import annotations

import argparse
import csv
import os
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

import lightkurve as lk
from mono_cbp.utils.data import load_catalogue, get_row, lc_to_txt
from mono_cbp.utils.eclipses import time_to_phase

MONO_CBP = Path(os.environ.get("MONO_CBP_DIR", Path.home() / "mono-cbp"))
CATALOGUE = MONO_CBP / "catalogues" / "TEBC_morph_05_P_7.csv"
DATA_ROOT = Path(os.environ.get("CBP_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CACHE = DATA_ROOT / "lc_cache"          # mono-cbp-recipe TESS-SPOC files (bulk download)
CACHE_QLP = DATA_ROOT / "lc_cache_qlp"  # QLP-only sectors, kept separate on purpose
OUT = DATA_ROOT / "goodsn_pilot"
PLOTS = OUT / "plots"
EYEBALL_SECTOR = 60  # sectors >= this get the by-eye flag (not yet searched by prior work)


def cached_sectors(tic: int) -> set[int]:
    out = set()
    for d in (CACHE, CACHE_QLP):
        if d.exists():
            for f in d.glob(f"TIC_{tic}_*.txt"):
                try:
                    out.add(int(f.stem.split("_")[2]))
                except (IndexError, ValueError):
                    pass
    return out


def save_qlp(row, lc, sector: int, tic: int) -> None:
    """Minimal saver for QLP products (no CROWDSAP/FLFRCSAP, so lc_to_txt can't run)."""
    CACHE_QLP.mkdir(parents=True, exist_ok=True)
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    fe = (np.asarray(lc.flux_err.value, float)
          if lc.flux_err is not None else np.full_like(f, np.nan))
    med = np.nanmedian(f)
    f, fe = f / med, fe / med
    ph = time_to_phase(t, period=row["period"], t0=row["bjd0"])
    keep = ~np.isnan(t * f * ph)
    arr = np.column_stack([t[keep], f[keep], fe[keep], ph[keep]])
    np.savetxt(CACHE_QLP / f"TIC_{tic}_{sector:02d}.txt", arr,
               header="TIME FLUX FLUX_ERR PHASE ECL_MASK")


def load_cached(tic: int, sector: int):
    for d in (CACHE, CACHE_QLP):
        p = d / f"TIC_{tic}_{sector:02d}.txt"
        if p.exists():
            a = np.loadtxt(p, skiprows=1, ndmin=2)
            src = "TESS-SPOC" if d == CACHE else "QLP"
            return a[:, 0], a[:, 1], src
    return None, None, None


def plot_system(tic: int, sectors: list[int], row) -> None:
    sysdir = PLOTS / f"TIC_{tic}"
    sysdir.mkdir(parents=True, exist_ok=True)
    stitched = []
    for s in sorted(sectors):
        t, f, src = load_cached(tic, s)
        if t is None or len(t) == 0:
            continue
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(t, f, ".", ms=2, color="navy")
        flag = "  [EYEBALL: post-search data]" if s >= EYEBALL_SECTOR else ""
        ax.set_title(f"TIC {tic}  S{s:02d}  ({src}, P_bin={row['period']:.3f} d){flag}")
        ax.set_xlabel("BTJD"); ax.set_ylabel("norm. flux")
        fig.tight_layout()
        fig.savefig(sysdir / f"S{s:02d}.png", dpi=110)
        plt.close(fig)
        stitched.append((s, t, f, src))
    if stitched:
        fig, ax = plt.subplots(figsize=(16, 3.6))
        for s, t, f, src in stitched:
            ax.plot(t, f, ".", ms=1.5,
                    color="crimson" if s >= EYEBALL_SECTOR else "navy")
        ax.set_title(f"TIC {tic} -- all {len(stitched)} sectors "
                     f"(red = S>={EYEBALL_SECTOR}, the new data)  P_bin={row['period']:.3f} d")
        ax.set_xlabel("BTJD"); ax.set_ylabel("norm. flux")
        fig.tight_layout()
        fig.savefig(PLOTS / f"TIC_{tic}_overview.png", dpi=110)
        plt.close(fig)


def process(tic: int, cat, writer) -> None:
    row = get_row(cat, tic)
    have = cached_sectors(tic)
    res = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    tbl = res.table
    # sector + author for every product MAST currently offers
    products = {}
    for r in tbl:
        try:
            sec = int(str(r["mission"]).split()[-1])
        except (ValueError, IndexError):
            continue
        products.setdefault(sec, set()).add(str(r["author"]))
    available = set(products)
    new = sorted(available - have)
    got_spoc, got_qlp, failed = [], [], []
    for s in new:
        try:
            if "TESS-SPOC" in products[s]:
                sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS",
                                          sector=s, author="TESS-SPOC")
                lc = sr.download(flux_column="sap_flux", quality_bitmask="hard")
                if lc is not None:
                    lc_to_txt(cat, lc, output_path=str(CACHE))
                    got_spoc.append(s)
                    continue
            if "QLP" in products[s]:
                sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS",
                                          sector=s, author="QLP")
                lc = sr.download()
                if lc is not None:
                    save_qlp(row, lc, s, tic)
                    got_qlp.append(s)
                    continue
            failed.append(s)  # only 2-min SPOC or nothing usable
        except Exception:
            failed.append(s)
    all_sectors = sorted(have | set(got_spoc) | set(got_qlp))
    plot_system(tic, all_sectors, row)
    writer.writerow({
        "tic": tic, "P_bin_d": f"{row['period']:.4f}",
        "sectors_available": len(available), "cached_before": len(have),
        "new_spoc": ",".join(map(str, got_spoc)) or "-",
        "new_qlp": ",".join(map(str, got_qlp)) or "-",
        "unfetched": ",".join(map(str, failed)) or "-",
        "eyeball_sectors": ",".join(str(s) for s in all_sectors if s >= EYEBALL_SECTOR) or "-",
        "total_now": len(all_sectors),
    })
    print(f"TIC {tic}: {len(available)} avail | {len(have)} cached | "
          f"+{len(got_spoc)} SPOC +{len(got_qlp)} QLP | miss {len(failed)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tics", default=str(DATA_ROOT / "goodsn_golden28.txt"))
    args = ap.parse_args()
    tics = [int(l) for l in open(args.tics) if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    cat = load_catalogue(str(CATALOGUE), TEBC=True)
    fields = ["tic", "P_bin_d", "sectors_available", "cached_before", "new_spoc",
              "new_qlp", "unfetched", "eyeball_sectors", "total_now"]
    with open(OUT / "pilot_status.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for tic in tics:
            try:
                process(tic, cat, writer)
            except Exception as e:
                print(f"TIC {tic}: FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            fh.flush()
    print("\nPilot done. Plots in", PLOTS, "| status CSV in", OUT / "pilot_status.csv")


if __name__ == "__main__":
    main()
