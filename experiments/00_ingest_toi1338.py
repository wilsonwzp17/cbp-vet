#!/usr/bin/env python
"""
00_ingest_toi1338.py -- validate the data path on TOI-1338.

Goal: prove we can rebuild a mono-cbp input light curve from raw MAST data and
run the mono-cbp search end-to-end on it, recovering the same reference
monotransit that the bundled example data yields. Everything downstream
(bulk downloads, injection campaigns) rests on this working.

Target: TIC 260128333 = TOI-1338, sector 6.
Reference event (mono-cbp/examples/results/detected_events.txt, S6 strong event):
    t_event ~ 1483.9075 BTJD, depth ~ 0.311%, duration ~ 0.292 d, SNR ~ 14.62

What the script does:
  1. Read the binary ephemeris + eclipse params from the TEBC catalogue row.
  2. Download the TESS-SPOC FFI light curve for S6 via lightkurve, and rebuild
     flux exactly the way mono-cbp's own utils.data.lc_to_txt does it
     (SAP flux, CROWDSAP/FLFRCSAP crowding correction, normalise, phase).
  3. Save it as an NPZ in the mono-cbp input schema.
  4. Validate the reconstruction against the shipped example NPZ (flux/phase).
  5. Run EclipseMasker -> TransitFinder on (a) our rebuilt NPZ and (b) a copy of
     the shipped NPZ as a control, and check both recover the reference event.

Run:
    conda activate mono-cbp
    python experiments/00_ingest_toi1338.py
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# --- mono-cbp library (canonical helpers we reuse rather than re-implement) ---
import lightkurve as lk
from mono_cbp import MonoCBPPipeline
from mono_cbp.utils.data import load_catalogue, get_row
from mono_cbp.utils.eclipses import time_to_phase, get_eclipse_mask

# ----------------------------------------------------------------------------
# Configuration (no magic constants buried in the logic below)
# ----------------------------------------------------------------------------
MONO_CBP_DIR = Path(os.environ.get("MONO_CBP_DIR", Path.home() / "mono-cbp"))
CATALOGUE_PATH = MONO_CBP_DIR / "catalogues" / "TEBC_morph_05_P_7.csv"
SECTOR_TIMES_PATH = MONO_CBP_DIR / "catalogues" / "sector_times.csv"
BUNDLED_NPZ = MONO_CBP_DIR / "data" / "TIC_260128333_06.npz"

TIC = 260128333
SECTOR = 6

# Where scratch data/outputs go (under cbp-vet/data/, which is git-ignored)
HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "data" / "ingest_toi1338"
DIR_MAST = WORK / "data_from_mast"          # our reconstruction
DIR_CONTROL = WORK / "data_bundled_control" # copy of the shipped NPZ
OUT_MAST = WORK / "out_from_mast"
OUT_CONTROL = WORK / "out_control"

# Reference event to recover
REF_TIME = 1483.9075     # BTJD
REF_DEPTH = 0.00311      # fractional (0.311 %)
REF_DURATION = 0.2917    # days
REF_SNR = 14.62

# Acceptance tolerances
TIME_TOL = max(0.15, REF_DURATION / 2.0)   # "recovered" = flagged within t_dur/2
MIN_SNR_ACCEPT = 8.0                        # must be a strong, unambiguous detection

# Search config: match the example notebook's search stage, plots off for speed.
SEARCH_CONFIG = {
    "transit_finding": {
        "edge_cutoff": 0.0,
        "mad_threshold": 3.0,
        "detrending_method": "cb",          # cosine + biweight (the paper's default)
        "generate_vetting_plots": False,
        "generate_skye_plots": False,
        "generate_event_snippets": True,
        "save_event_snippets": False,
        "cadence_minutes": 30,
        "cosine": {"win_len_max": 12, "win_len_min": 1, "fap_threshold": 1e-2, "poly_order": 2},
        "biweight": {"win_len_max": 3, "win_len_min": 1},
        "filters": {"min_snr": 5, "max_duration_days": 1, "det_dependence_threshold": 18},
    }
}


def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


def get_ephemeris():
    """Return the catalogue row (period, bjd0, prim/sec eclipse params) for TIC."""
    cat = load_catalogue(str(CATALOGUE_PATH), TEBC=True)
    row = get_row(cat, TIC)
    if row is None:
        raise SystemExit(f"TIC {TIC} not found in {CATALOGUE_PATH}")
    print(f"  period = {row['period']:.6f} d   bjd0 = {row['bjd0']:.4f} BTJD")
    print(f"  prim_pos = {row['prim_pos']:.4f}  prim_width = {row['prim_width']:.4f}")
    print(f"  sec_pos  = {row['sec_pos']:.4f}  sec_width  = {row['sec_width']:.4f}")
    return row


def download_and_build(row):
    """Rebuild the S6 light curve from raw MAST, mirroring utils.data.lc_to_txt.

    Returns a dict with keys time/flux/flux_err/phase/eclipse_mask, or raises.
    """
    search = lk.search_lightcurve(
        f"TIC {TIC}", mission="TESS", sector=SECTOR, author="TESS-SPOC"
    )
    print(f"  MAST search returned {len(search)} product(s)")
    if len(search) == 0:
        raise RuntimeError("No TESS-SPOC light curve found for this TIC/sector.")
    lc = search.download(flux_column="sap_flux", quality_bitmask="hard")
    if lc is None:
        raise RuntimeError("MAST download returned None.")

    time = np.asarray(lc.time.value, dtype=float)
    raw_flux = np.asarray(lc.flux.value, dtype=float)
    raw_flux_err = np.asarray(lc.flux_err.value, dtype=float)

    # Crowding / flux-fraction correction (CROWDSAP, FLFRCSAP), as mono-cbp does.
    crowdsap = lc.meta.get("CROWDSAP")
    flfrcsap = lc.meta.get("FLFRCSAP")
    print(f"  CROWDSAP = {crowdsap}   FLFRCSAP = {flfrcsap}")
    raw_flux[raw_flux == 0] = np.nan
    raw_flux_err[raw_flux == 0] = np.nan
    if crowdsap is not None and flfrcsap is not None:
        median_flux = np.nanmedian(raw_flux)
        excess = (1.0 - crowdsap) * median_flux
        flux = (raw_flux - excess) / flfrcsap
        flux_err = raw_flux_err / flfrcsap
    else:
        print("  WARNING: CROWDSAP/FLFRCSAP missing -> skipping crowding correction")
        flux, flux_err = raw_flux, raw_flux_err

    # Normalise to median 1.
    flux_err = flux_err / np.nanmedian(flux)
    flux = flux / np.nanmedian(flux)

    phase = time_to_phase(time, period=row["period"], t0=row["bjd0"])

    # Drop NaNs (mono-cbp filters on time*flux*flux_err*phase).
    keep = ~np.isnan(time * flux * flux_err * phase)
    time, flux, flux_err, phase = time[keep], flux[keep], flux_err[keep], phase[keep]

    eclipse_mask = np.logical_or(
        get_eclipse_mask(phase, row["prim_pos"], row["prim_width"]),
        get_eclipse_mask(phase, row["sec_pos"], row["sec_width"]),
    )

    cad = np.nanmedian(np.diff(time)) * 1440.0
    print(f"  built N={len(time)} cadences, {cad:.1f} min cadence, "
          f"span {time.min():.3f}..{time.max():.3f} BTJD, "
          f"in-eclipse frac {eclipse_mask.mean():.3f}")
    return {"time": time, "flux": flux, "flux_err": flux_err,
            "phase": phase, "eclipse_mask": eclipse_mask}


def compare_to_bundled(recon: dict) -> None:
    """Report how closely the reconstruction matches the shipped example NPZ."""
    if not BUNDLED_NPZ.exists():
        print("  (bundled NPZ not found; skipping comparison)")
        return
    b = np.load(BUNDLED_NPZ)
    bt, bf = np.asarray(b["time"], float), np.asarray(b["flux"], float)
    rt, rf = recon["time"], recon["flux"]
    print(f"  bundled N={len(bt)}   reconstructed N={len(rt)}")

    # Match on nearest timestamp (within half a cadence) and compare flux.
    order = np.argsort(rt)
    rt_s, rf_s = rt[order], rf[order]
    idx = np.searchsorted(rt_s, bt)
    idx = np.clip(idx, 0, len(rt_s) - 1)
    dt = np.abs(rt_s[idx] - bt)
    matched = dt < (0.0104)  # ~15 min = half of 30-min cadence
    n_match = int(matched.sum())
    if n_match:
        df = np.abs(rf_s[idx][matched] - bf[matched])
        finite = np.isfinite(df)
        print(f"  matched {n_match}/{len(bt)} timestamps; "
              f"median |Δflux| = {np.nanmedian(df[finite]):.2e}, "
              f"95th pct = {np.nanpercentile(df[finite], 95):.2e}")
    else:
        print("  WARNING: no timestamps matched — check time system/quality mask")


def run_search(data_dir: Path, out_dir: Path) -> list[dict]:
    """Run mask+find on a directory holding one NPZ; return event dicts for our TIC/sector."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = MonoCBPPipeline(
        catalogue_path=str(CATALOGUE_PATH),
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        sector_times_path=str(SECTOR_TIMES_PATH),
        TEBC=True,
        config=SEARCH_CONFIG,
    )
    pipe.run(vet_candidates=False, injection_retrieval=False,
             mask_eclipses_kwargs={"force": True})
    r = pipe.transit_finder.results
    events = []
    for i in range(len(r["event_times"])):
        if str(r["tics"][i]) == str(TIC) and str(r["sectors"][i]) == str(SECTOR):
            events.append({
                "time": float(r["event_times"][i]),
                "phase": float(r["event_phases"][i]),
                "depth": float(r["event_depths"][i]),
                "duration": float(r["event_durations"][i]),
                "snr": float(r["event_snrs"][i]),
            })
    return sorted(events, key=lambda e: e["time"])


def assess(events: list[dict], label: str) -> bool:
    """Print events and decide whether the reference event was recovered."""
    print(f"\n  [{label}] {len(events)} TCE(s) for TIC {TIC} S{SECTOR}:")
    for e in events:
        near = abs(e["time"] - REF_TIME) < TIME_TOL
        flag = " <== reference" if near else ""
        print(f"    t={e['time']:.4f}  depth={e['depth']*100:.3f}%  "
              f"dur={e['duration']:.3f}d  SNR={e['snr']:.2f}{flag}")
    hits = [e for e in events
            if abs(e["time"] - REF_TIME) < TIME_TOL and e["snr"] >= MIN_SNR_ACCEPT]
    if hits:
        e = max(hits, key=lambda x: x["snr"])
        dsnr = e["snr"] - REF_SNR
        print(f"  RECOVERED: t={e['time']:.4f} (Δt={e['time']-REF_TIME:+.4f} d), "
              f"SNR={e['snr']:.2f} (ref {REF_SNR}, Δ={dsnr:+.2f}), "
              f"depth={e['depth']*100:.3f}% (ref {REF_DEPTH*100:.3f}%)")
        return True
    print(f"  NOT RECOVERED within Δt<{TIME_TOL:.3f} d and SNR>={MIN_SNR_ACCEPT}")
    return False


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    for d in (DIR_MAST, DIR_CONTROL):
        d.mkdir(parents=True, exist_ok=True)

    banner("1. Ephemeris from TEBC catalogue")
    row = get_ephemeris()

    banner("2-4. Rebuild S6 light curve from MAST + validate vs bundled")
    mast_ok = True
    try:
        recon = download_and_build(row)
        np.savez(DIR_MAST / f"TIC_{TIC}_{SECTOR:02d}.npz", **recon)
        print(f"  saved {DIR_MAST / f'TIC_{TIC}_{SECTOR:02d}.npz'}")
        compare_to_bundled(recon)
    except Exception as exc:  # noqa: BLE001
        mast_ok = False
        print(f"  MAST rebuild FAILED: {type(exc).__name__}: {exc}")
        print("  (continuing with the bundled control so the run still reports)")

    # Control: copy the shipped NPZ (never mutate the original in place).
    shutil.copy(BUNDLED_NPZ, DIR_CONTROL / f"TIC_{TIC}_{SECTOR:02d}.npz")

    banner("5. Run mono-cbp search (mask -> detrend -> find)")
    control_pass = mast_pass = False
    print("\n-- control: shipped example NPZ --")
    control_pass = assess(run_search(DIR_CONTROL, OUT_CONTROL), "control")
    if mast_ok:
        print("\n-- reconstruction: rebuilt-from-MAST NPZ --")
        mast_pass = assess(run_search(DIR_MAST, OUT_MAST), "from-MAST")

    banner("SUMMARY")
    print(f"  control (shipped NPZ) recovers reference event : {'PASS' if control_pass else 'FAIL'}")
    if mast_ok:
        print(f"  rebuilt-from-MAST recovers reference event     : {'PASS' if mast_pass else 'FAIL'}")
    else:
        print(f"  rebuilt-from-MAST recovers reference event     : SKIPPED (download failed)")

    overall = control_pass and (mast_pass if mast_ok else False)
    print(f"\n  ACCEPTANCE (ingestion head-start): {'PASS' if overall else 'INCOMPLETE'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
