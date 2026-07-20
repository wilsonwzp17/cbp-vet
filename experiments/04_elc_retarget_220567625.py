#!/usr/bin/env python
"""
04_elc_retarget_220567625.py -- first forward-model retarget onto a real system.

Target: TIC 220567625 (good S/N, TEBC ephemeris, and a Gaia DR3 SB1 orbit).
  TEBC eclipse ephemeris:  P = 19.426601 d, Tconj(primary) = 1344.87755 BTJD,
                           secondary eclipse at phase 0.4921 (so ecosw ~ -0.0124)
  Gaia DR3 SB1 orbit:      P = 19.426588 +/- 0.0008 d (agrees to 6 decimals),
                           ecc = 0.284314 +/- 0.0026 (real spectroscopic value)
  Cached data: sectors 1, 3, 28, 29, 30 (~790 d span, ~40 binary cycles).

The exercise: copy the Kepler-16 template, change the binary
period + Tconj + eccentricity, run, and check the model's eclipses land on the
real ones. Stellar masses/radii/temps stay at Kepler-16 values, so DEPTHS will
not match -- TIMING is the test. Time axis = BTJD everywhere (the trap).

omega convention is resolved empirically: |cos(omega)| is fixed by the observed
secondary phase; the script tries omega = 92.53 deg first, checks where the
model secondary lands, and flips to 267.47 deg if wrong.

Run:  conda activate mono-cbp && python experiments/04_elc_retarget_220567625.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(os.environ.get("CBP_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
ELC90 = DATA / "elc" / "ELC90"
WORK = DATA / "elc" / "retarget_220567625"
CACHE = DATA / "lc_cache"
OUTDIR = DATA / "goodsn_pilot"

TIC = 220567625
P_BIN = 19.426601          # TEBC (Gaia SB1 agrees: 19.426588 +/- 0.0008)
TCONJ = 1344.87755         # BTJD, TEBC bjd0 (primary eclipse)
ECC = 0.284314             # Gaia SB1
SEC_PHASE_OBS = 0.4920801828187696   # TEBC sec_pos_2g
ECOSW = (np.pi / 2) * (SEC_PHASE_OBS - 0.5)   # ~ -0.01244
# ELC has a compiled-in light-curve length cap (Nmaxphase = 189,012 points) and
# exits 0 (!) with only a stdout message when exceeded. So instead of one model
# spanning the 1,000-day data gap, run two windows covering the actual sectors
# and stitch. (Windows also dodge the Nbody=2 span segfault; Nbody stays 3.)
WINDOWS = [(1320.0, 2145.0), (3150.0, 3210.0)]   # S1-S30, then the new S68-S69
# ELC outputs on an internal ~0.0039569-d grid but ALLOCATES span/step points
# from this input value, so step must be <= 0.00396 or allocation < request
# ("Nmaxphase is too small"). The template's 0.002 is safe; coarser is not.
TIME_STEP = 0.002
T_START = WINDOWS[0][0]   # Tref anchor
SEPAR = 48.251791 * (P_BIN / 41.079297) ** (2.0 / 3.0)  # Kepler-16 a scaled by P^(2/3)


def patch_template(omega_deg: float, t0: float, t1: float) -> str:
    src = (ELC90 / "ELC_Kepler16.inp").read_text().splitlines()
    ecosw = ECC * np.cos(np.radians(omega_deg))
    esinw = ECC * np.sin(np.radians(omega_deg))
    repl = {
        "t_start (if itime=2)":        f"  {t0:.8f}",
        "t_end   (if itime=2)":        f"  {t1:.8f}",
        "time step in days (if itime=2)": f" {TIME_STEP:.8f}",
        "Tref for dynamical integrator": f"     {T_START:.6f}",
        "Period (days), tag pe":       f"    {P_BIN:.9f}",
        "T0 (time of periastron passage), tag T0": "     0.000000000000000",
        "Tconj (time of primary eclipse), tag Tc": f"   {TCONJ:.9f}",
        "eccentricity, tag ec":        f"{ECC:.13f}",
        "argument of periaston in degrees, tag ar": f"  {omega_deg:.10f}",
        "e*cos(omega), tag oc":        f"{ecosw: .17f}",
        "e*sin(omega), tag os":        f"{esinw: .17f}",
        "separ (semimajor axis in solar radii), tag se": f"   {SEPAR:.6f}",
        # Keep Nbody=3 exactly as the template has it. Findings from failed
        # variants (known quirks): Nbody=2 segfaults on spans > ~805 d;
        # touching Iseason or P1incl or P1ratrad makes ELC exit silently with no
        # outputs (fixed-sequence parse sensitivity). So the Kepler-16 planet
        # stays in the model; its phantom transits are masked in analysis by
        # ephemeris matching instead.
    }
    out = []
    for line in src:
        for key, val in repl.items():
            if key in line:
                line = f"{val}   {key}"
                break
        out.append(line)
    return "\n".join(out) + "\n"


def run_elc(omega_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Run one ELC model per window and stitch. Verifies output exists: ELC can
    exit 0 while failing (e.g. the Nmaxphase cap prints to stdout, no files)."""
    ts, fs = [], []
    for (t0, t1) in WINDOWS:
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        for f in ("ELC", "ELC.atm"):
            shutil.copy(ELC90 / f, WORK / f)
        (WORK / "ELC.inp").write_text(patch_template(omega_deg, t0, t1))
        r = subprocess.run(["./ELC"], cwd=WORK, capture_output=True,
                           text=True, timeout=900)
        mfile = WORK / "modelU.linear"
        if r.returncode != 0 or not mfile.exists():
            raise RuntimeError(
                f"ELC failed for window {t0}-{t1} (exit {r.returncode}): "
                f"{(r.stdout or '').strip().splitlines()[-2:]}")
        m = np.loadtxt(mfile)
        ts.append(m[:, 0])
        fs.append(m[:, 1] / np.median(m[:, 1]))
    return np.concatenate(ts), np.concatenate(fs)


def eclipse_minima(t, f, thresh=0.995):
    """Times of local minima below thresh, one per contiguous dip."""
    below = f < thresh
    times = []
    i = 0
    while i < len(f):
        if below[i]:
            j = i
            while j < len(f) and below[j]:
                j += 1
            seg = slice(i, j)
            times.append(t[seg][np.argmin(f[seg])])
            i = j
        else:
            i += 1
    return np.array(times)


def main() -> None:
    # --- run with omega guess 1; flip if the secondary lands wrong ---
    for omega in (92.53, 267.47):
        t, f = run_elc(omega)
        dips = eclipse_minima(t, f)
        phases = ((dips - TCONJ) / P_BIN) % 1.0
        prim = dips[(phases < 0.1) | (phases > 0.9)]
        sec = dips[(np.abs(phases - 0.5) < 0.15)]
        sec_phase_model = np.median(((sec - TCONJ) / P_BIN) % 1.0) if len(sec) else np.nan
        print(f"omega={omega}: {len(prim)} primaries, {len(sec)} secondaries, "
              f"model sec phase = {sec_phase_model:.4f} (obs {SEC_PHASE_OBS:.4f})")
        if len(sec) and abs(sec_phase_model - SEC_PHASE_OBS) < 0.01:
            print(f"  -> omega = {omega} deg accepted")
            break

    # --- classify model dips by ephemeris matching; phantoms = neither series ---
    def nearest_dist(times, series_t0):
        cyc = np.round((times - series_t0) / P_BIN)
        return np.abs(times - (series_t0 + cyc * P_BIN))
    sec_t0 = TCONJ + SEC_PHASE_OBS * P_BIN
    d_prim = nearest_dist(dips, TCONJ)
    d_sec = nearest_dist(dips, sec_t0)
    prim = dips[(d_prim < 0.3) & (d_prim <= d_sec)]
    sec = dips[(d_sec < 0.3) & (d_sec < d_prim)]
    phantom = dips[(d_prim >= 0.3) & (d_sec >= 0.3)]  # the template planet's transits
    print(f"dips: {len(prim)} primaries, {len(sec)} secondaries, "
          f"{len(phantom)} phantom (template-planet) dips masked")

    cycles = np.round((prim - TCONJ) / P_BIN)
    resid_min = (prim - (TCONJ + cycles * P_BIN)) * 24 * 60
    print(f"model primary-eclipse residuals vs ephemeris: "
          f"median {np.median(np.abs(resid_min)):.2f} min, max {np.max(np.abs(resid_min)):.2f} min "
          f"over {len(prim)} eclipses / {t.max()-t.min():.0f} d")

    # NaN out the model around phantom dips so the overlay shows only the binary.
    # Only mask phantoms well clear of real eclipses, so the mask can never erase
    # a genuine model eclipse from the figure.
    for tp in phantom:
        if min(nearest_dist(np.array([tp]), TCONJ)[0],
               nearest_dist(np.array([tp]), sec_t0)[0]) > 1.0:
            f[np.abs(t - tp) < 0.35] = np.nan

    # --- overlay on the real cached sectors ---
    files = sorted(CACHE.glob(f"TIC_{TIC}_*.txt"))
    fig, axes = plt.subplots(len(files), 1, figsize=(13, 2.6 * len(files)))
    for ax, fp in zip(np.atleast_1d(axes), files):
        d = np.loadtxt(fp, skiprows=1, ndmin=2)
        td, fd = d[:, 0], d[:, 1]
        ax.plot(td, fd, ".", ms=2.5, color="navy", label="TESS (cached)")
        m = (t >= td.min() - 1) & (t <= td.max() + 1)
        # depths differ by construction (Kepler-16 stars); rescale model dip depth to the data
        fm = f[m]
        data_depth = 1 - np.nanmin(fd)
        model_depth = 1 - np.nanmin(fm) if len(fm) else 1
        fscaled = 1 - (1 - fm) * (data_depth / model_depth if model_depth > 0 else 1)
        ax.plot(t[m], fscaled, "-", lw=1, color="crimson", alpha=0.8,
                label="ELC model (timing; depth rescaled)")
        sec_lbl = fp.stem.split("_")[2]
        ax.set_title(f"TIC {TIC}  S{sec_lbl}", fontsize=10)
        ax.set_ylabel("flux")
        ax.legend(fontsize=7, loc="lower right")
    axes = np.atleast_1d(axes)
    axes[-1].set_xlabel("BTJD")
    fig.suptitle(f"First ELC retarget: TIC {TIC} -- P={P_BIN} d, ecc={ECC} (Gaia), "
                 f"Tconj={TCONJ} (TEBC). Model eclipses vs real data.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = OUTDIR / f"elc_retarget_TIC{TIC}.png"
    fig.savefig(out, dpi=130)
    print(f"overlay figure: {out}")


if __name__ == "__main__":
    main()
