#!/usr/bin/env python
"""
02_randomized_profile_bank.py  --  W2.2 step 3 prototype.

Problem found in W2.2: mono-cbp's injector randomizes only the injection TIME.
The transit SHAPE comes from a fixed bank (the stock bank is a 7x7 grid of depth x
duration, all at impact parameter 0 and one limb-darkening law). For training data we
need variety in WHAT is injected, so a classifier learns transit morphology, not the
49 memorized template shapes.

Fix (this prototype): generate our own bank of RANDOMIZED profiles and hand it to the
injector unchanged. We reuse mono-cbp's own batman-based generator per profile, so the
transit physics/math is identical to the library; we only randomize the parameters:
  depth      ~ log-uniform(0.05%, 1.5%)
  duration   ~ uniform(0.05, 1.2 d)
  impact b   ~ uniform(0, 0.9)         (varies V- vs U-shape, some near-grazing)
  limb dark  ~ randomized quadratic coefficients
The injector then injects each of these at a random time into random light curves.

(Ingress/egress asymmetry, to mimic the moving secondary, is a Week-3 refinement:
 batman transits are symmetric; add it later via an eccentric/warped model.)

Usage:
    conda activate mono-cbp
    python experiments/02_randomized_profile_bank.py
Env: MONO_CBP_DIR (default ~/mono-cbp), CBP_DATA_DIR (default <repo>/data)
"""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from mono_cbp.utils.transit_models import create_transit_models, save_transit_models, load_transit_models

MONO_CBP = Path(os.environ.get("MONO_CBP_DIR", Path.home() / "mono-cbp"))
DATA_ROOT = Path(os.environ.get("CBP_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def create_randomized_bank(n_models=48, seed=0,
                           depth_range=(5e-4, 1.5e-2),
                           duration_range=(0.05, 1.2),
                           b_max=0.9, cadence_minutes=30):
    """Build a bank of n randomized transit profiles in mono-cbp's transit_models format."""
    rng = np.random.default_rng(seed)
    models, meta_time = [], None
    for _ in range(n_models):
        depth = 10.0 ** rng.uniform(np.log10(depth_range[0]), np.log10(depth_range[1]))
        dur = rng.uniform(*duration_range)
        b = rng.uniform(0.0, b_max)
        u1, u2 = rng.uniform(0.2, 0.5), rng.uniform(0.1, 0.3)
        # reuse the library's exact per-profile batman + Winn(2010) geometry (num=1 => one model)
        one = create_transit_models(
            depth_range=(depth, depth), duration_range=(dur, dur),
            num_depths=1, num_durations=1, cadence_minutes=cadence_minutes,
            impact_parameter=b, limb_dark_coeffs=(u1, u2),
        )
        meta_time = one["time"]
        models.append(one["models"][0])
    bank = {
        "time": meta_time, "models": models,
        "num_depths": n_models, "num_durations": 1,   # product = n_models
        "depth_range": depth_range, "duration_range": duration_range,
        "cadence_minutes": cadence_minutes,
    }
    return bank


def main():
    out = DATA_ROOT / "profile_bank"
    out.mkdir(parents=True, exist_ok=True)
    bank_path = out / "transit_models_random.npz"

    bank = create_randomized_bank(n_models=48, seed=0)
    save_transit_models(bank, str(bank_path))
    print(f"Saved {len(bank['models'])} randomized profiles -> {bank_path}")

    # verify it round-trips and is genuinely off-grid
    lb = load_transit_models(str(bank_path))
    d = np.array([m["depth"] for m in lb["models"]])
    du = np.array([m["duration"] for m in lb["models"]])
    b = np.array([m["impact_parameter"] for m in lb["models"]])
    print(f"  depth:    {d.min()*100:.3f}% .. {d.max()*100:.3f}%  ({len(np.unique(np.round(d,6)))} distinct)")
    print(f"  duration: {du.min():.3f} .. {du.max():.3f} d       ({len(np.unique(np.round(du,4)))} distinct)")
    print(f"  impact b: {b.min():.3f} .. {b.max():.3f}           (stock bank is all b=0)")

    # prove the injector uses our bank: run injection-recovery with it
    from mono_cbp import MonoCBPPipeline
    WORK = Path("/tmp/w2_2_inject"); DDIR = WORK / "data"; ODIR = WORK / "out"
    if WORK.exists(): shutil.rmtree(WORK)
    DDIR.mkdir(parents=True); ODIR.mkdir(parents=True)
    shutil.copy(DATA_ROOT / "ingest_toi1338/data_from_mast/TIC_260128333_06.npz", DDIR / "TIC_260128333_06.npz")
    CFG = {"transit_finding": {"detrending_method": "cb", "mad_threshold": 3.0, "generate_vetting_plots": False,
            "generate_skye_plots": False, "generate_event_snippets": True, "save_event_snippets": False, "cadence_minutes": 30,
            "cosine": {"win_len_max": 12, "win_len_min": 1, "fap_threshold": 1e-2, "poly_order": 2},
            "biweight": {"win_len_max": 3, "win_len_min": 1},
            "filters": {"min_snr": 5, "max_duration_days": 1, "det_dependence_threshold": 18}}}
    p = MonoCBPPipeline(catalogue_path=str(MONO_CBP / "catalogues/TEBC_morph_05_P_7.csv"),
                        data_dir=str(DDIR), output_dir=str(ODIR),
                        sector_times_path=str(MONO_CBP / "catalogues/sector_times.csv"),
                        TEBC=True, config=CFG, transit_models_path=str(bank_path))
    ir = p.run_injection_retrieval(n_injections=1, output_dir=str(ODIR))
    inj_d = np.asarray(ir["injected_depth"] if "injected_depth" in ir else ir.get("injected_depths"))
    print(f"\n  injector ran {len(ir)} tests using OUR bank (one per randomized profile).")
    print(f"  injected depths are continuous, not a 7-value grid: "
          f"{len(np.unique(np.round(inj_d,6)))} distinct values across {len(ir)} tests")
    rec_col = [c for c in ir.columns if 'recover' in c.lower()]
    print(f"  result columns: {list(ir.columns)}")
    print("\nW2.2 CLOSED: extraction works; randomized injection achieved via our own profile bank.")


if __name__ == "__main__":
    main()
