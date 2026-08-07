"""Deployment-native detectability injections, per adoptions H1/H2/H5.

What this produces
------------------
The detectability measurement for the talk's deployment slide: injection
tests into the SAME eclipse-masked staged copies the deployment search
processed, under the SAME 50/50 dual-inversion harness as the benchmark
(H2: a fourth provenance, "deploy_native" on injector rows only), with the
radius-axis inputs (model-frame sqrt(depth)) recorded per test. Three
programs, per H1:

1. PER-SYSTEM (headline): the golden 28 minus TIC 166532356 (zero disjoint
   files - H1a), each on its DISJOINT (tic, sector) files only (files NOT in
   the frozen 1,946), N_PER_SYSTEM injections each.
2. AGGREGATE (headline): disjoint non-golden files, N_AGGREGATE injections.
3. ALL-63 SECONDARY: every staged file including frozen-overlap ones,
   N_SECONDARY injections, overlap disclosed (the disjointness pin).

Discipline encoded from the adoptions:
- H1b: the first 100 injections are timed and the 2.5 s/injection gate is
  re-asserted; failing it flips the run to aggregate-only (the adoption's
  own text) and records the flip.
- H1d: a 90-minute wall tripwire kills remaining per-system programs and
  jumps to the aggregate (consistent with the Aug-11 casualty order).
- H2: p_invert = 0.5 with the per-test mix gate |rate - 0.5| <= 0.02 per
  program; the native-path mask list is the deployment search's OWN
  detected_events.txt; outputs live in a fresh dir, never merged into
  bench-v1 shards.
- H5: the per-file masking regime (ecl_mask column absent on 4-column
  30-min files -> injector sees None; sub-30-min files rebin with
  catalog-recomputed masks) matches the benchmark campaign's own behavior
  BY DESIGN; it is disclosed in the summary, not shimmed.

Outputs (data/deploy_run/detectability/, campaign format so the standard
export path can consume them): tests_all.csv, events_all.csv, snippets/,
campaign_summary.json (pins include pair_model v3 + p_invert + programs).
The vetter-pass leg (export recovered events with label_source
bank_injection per H2, score with the frozen checkpoint) runs afterwards
via 25-style tooling; this script is injections only.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
import re
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet import physics
from cbpvet.injection import DualInjector, pair_model
from cbpvet.injection.epoch_sampler import HistogramEpochSampler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
HIST = os.path.join(REPO, "data", "search_frozen", "out", "joint_phase_gap_hist.npz")
DEPLOY_EVENTS = os.path.join(REPO, "data", "deploy_run", "search", "detected_events.txt")
BANK = os.path.join(REPO, "data", "profile_bank_v2", "transit_models_v2.npz")
NOISE = os.path.join(REPO, "data", "noise_screen.csv")
STAGED = os.path.join(REPO, "data", "deploy_staged", "lc")
FROZEN_STAGED = os.path.join(REPO, "data", "search_frozen", "staged")
GOLDEN = os.path.join(REPO, "data", "goodsn_golden28.txt")
OUT = os.path.join(REPO, "data", "deploy_run", "detectability")

SEED = 20260807
N_PER_SYSTEM = 150
N_AGGREGATE = 2000
N_SECONDARY = 1000
PAIR_RATE = 0.37
GATE_S_PER_INJ = 2.5
TRIPWIRE_S = 90 * 60
EXCLUDED_GOLDEN = 166532356        # zero disjoint files (H1a)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("28_detect")


def load_bank_models(indices):
    with np.load(BANK, allow_pickle=True) as data:
        return {int(i): {"flux": data[f"model_{i}_flux"],
                         "depth": float(data[f"model_{i}_depth"]),
                         "duration": float(data[f"model_{i}_duration"])}
                for i in indices}


def host_table(cat_df, raw_df, noise_df):
    hosts = {}
    raw_i = raw_df.drop_duplicates("tess_id").set_index("tess_id")
    noise_i = noise_df.set_index("tic")
    for tic in cat_df.tess_id.unique():
        rr = raw_i.loc[tic]
        _, _, e = physics.eclipse_eccentricity(
            rr.get("prim_pos_2g"), rr.get("sec_pos_2g"),
            rr.get("prim_width_2g"), rr.get("sec_width_2g"))
        if not np.isfinite(e):
            e = 0.0
        dr = np.nan
        if tic in noise_i.index:
            dr = float(noise_i.loc[tic, "depth_ratio"])
        if not np.isfinite(dr) or dr <= 0:
            dr = 1.0
        row = cat_df[cat_df.tess_id == tic].iloc[0]
        hosts[int(tic)] = {"p_bin": float(row["period"]), "e": float(e),
                           "depth_ratio": float(np.clip(dr, 0.05, 1.0))}
    return hosts


def main():
    os.makedirs(os.path.join(OUT, "snippets"), exist_ok=True)
    rng = np.random.default_rng(SEED)

    files = sorted(f for f in os.listdir(STAGED) if f.endswith(".txt"))
    frozen = {f for f in os.listdir(FROZEN_STAGED) if f.endswith(".txt")}
    golden = {int(l.strip()) for l in open(GOLDEN) if l.strip()}

    def tic_of(f):
        return int(re.match(r"TIC_(\d+)_", f).group(1))

    disjoint = [f for f in files if f not in frozen]
    per_system = {}
    for f in disjoint:
        t = tic_of(f)
        if t in golden and t != EXCLUDED_GOLDEN:
            per_system.setdefault(t, []).append(f)
    agg_files = [f for f in disjoint if tic_of(f) not in golden]
    log.info("files: %d staged, %d disjoint; per-system TICs %d (files %d); "
             "aggregate files %d; secondary pool %d",
             len(files), len(disjoint), len(per_system),
             sum(len(v) for v in per_system.values()), len(agg_files), len(files))

    cat = load_catalogue(CAT_PATH, TEBC=True).drop_duplicates("tess_id")
    raw = pd.read_csv(CAT_PATH)
    noise = pd.read_csv(NOISE)
    hosts = host_table(cat, raw, noise)
    deploy_ev = pd.read_csv(DEPLOY_EVENTS, sep=r"\s+")
    sampler = HistogramEpochSampler(HIST, rng, catalogue=cat)

    inj = DualInjector(
        BANK, real_events_df=deploy_ev, rng_seed=SEED, epoch_sampler=sampler,
        catalogue=cat, TEBC=False, p_invert=0.5,
        provenance="deploy_native", run_id="detectability_2026-08-07",
        load_models=False)

    # Build the full injection plan up front so model loading is one pass.
    plan = []          # (program, file, tic)
    for t, fl in sorted(per_system.items()):
        for k in range(N_PER_SYSTEM):
            plan.append((f"per_system_{t}", fl[k % len(fl)], t))
    for k in range(N_AGGREGATE):
        f = agg_files[int(rng.integers(len(agg_files)))]
        plan.append(("aggregate", f, tic_of(f)))
    for k in range(N_SECONDARY):
        f = files[int(rng.integers(len(files)))]
        plan.append(("secondary_all63", f, tic_of(f)))

    midx_all = rng.choice(16384, size=len(plan), replace=False) \
        if len(plan) <= 16384 else rng.integers(0, 16384, size=len(plan))
    models = load_bank_models(sorted(set(int(i) for i in midx_all)))
    log.info("plan: %d injections (%d per-system, %d aggregate, %d secondary); "
             "%d distinct bank models loaded",
             len(plan), sum(1 for p in plan if p[0].startswith("per_system")),
             N_AGGREGATE, N_SECONDARY, len(models))

    t_start = _time.perf_counter()
    timed_100 = []
    program_of_test = []
    flip_to_aggregate = False
    tripwire_fired = False
    n_done = 0
    for k, (program, fname, tic) in enumerate(plan):
        if flip_to_aggregate and program.startswith("per_system"):
            continue
        if not tripwire_fired and program.startswith("per_system") and \
                _time.perf_counter() - t_start > TRIPWIRE_S:
            tripwire_fired = True
            log.warning("90-min tripwire: killing remaining per-system, "
                        "running aggregate (H1d)")
        if tripwire_fired and program.startswith("per_system"):
            continue

        host = hosts.get(tic)
        model = models[int(midx_all[k])]
        partner = None
        if host and rng.random() < PAIR_RATE:
            p_p = float(physics.draw_planet_period(rng, host["p_bin"],
                                                   host["e"], size=1)[0])
            dt = pair_model.pair_spacing(rng, host["p_bin"], p_p,
                                         model["duration"], size=1)
            if dt is not None:
                partner = {"dt": float(dt[0]),
                           "ratio": float(pair_model.pair_depth_ratio(
                               rng, host["depth_ratio"], size=1)[0]),
                           "duration_ratio": float(
                               pair_model.pair_duration_ratio(rng, size=1)[0])}
        t0 = _time.perf_counter()
        inj.process_file(os.path.join(STAGED, fname), model["flux"],
                         model["depth"], model["duration"],
                         model_idx=int(midx_all[k]),
                         snippet_dir=os.path.join(OUT, "snippets"),
                         partner=partner)
        dt_s = _time.perf_counter() - t0
        program_of_test.append(program)
        n_done += 1
        if n_done <= 100:
            timed_100.append(dt_s)
            if n_done == 100:
                mean_s = float(np.mean(timed_100))
                log.info("H1b gate re-assert at 100 injections: %.3f "
                         "s/injection vs %.1f gate", mean_s, GATE_S_PER_INJ)
                if mean_s > GATE_S_PER_INJ:
                    flip_to_aggregate = True
                    log.warning("GATE EXCEEDED: flipping to aggregate-only "
                                "per H1's own text; recorded, not silent")
        if n_done % 1000 == 0:
            log.info("%d/%d injections, %.1f min elapsed", n_done, len(plan),
                     (_time.perf_counter() - t_start) / 60)

    tests = inj.tests_frame()
    events = inj.events_frame()
    # tests_frame rows align 1:1 with successful process_file calls
    assert len(tests) == len(program_of_test), (len(tests), len(program_of_test))
    tests = tests.copy()
    tests["program"] = program_of_test
    tests["rp_over_rstar_model"] = np.sqrt(tests["depth_model"].astype(float)) \
        if "depth_model" in tests.columns else np.nan
    tests.to_csv(os.path.join(OUT, "tests_all.csv"), index=False)
    events.to_csv(os.path.join(OUT, "events_all.csv"), index=False)

    mix = {}
    for prog, grp in tests.groupby("program" if "program" in tests else []):
        r = float(grp.inverted_lc.mean())
        mix[prog] = {"n": int(len(grp)), "inverted_rate": r,
                     "gate_pass": bool(abs(r - 0.5) <= 0.02)}
    prog_prefix_mix = {}
    for prefix in ("per_system", "aggregate", "secondary_all63"):
        grp = tests[tests.program.str.startswith(prefix)]
        if len(grp):
            r = float(grp.inverted_lc.mean())
            prog_prefix_mix[prefix] = {"n": int(len(grp)), "inverted_rate": r,
                                       "gate_pass": bool(abs(r - 0.5) <= 0.02)}

    total_s = _time.perf_counter() - t_start
    summary = {
        "run": "detectability_2026-08-07",
        "adoptions": "H1 (per-system golden 27 + aggregate), H2 (dual 50/50, "
                     "deploy_native on injector rows only), H5 (masking regime "
                     "disclosed, not shimmed)",
        "pins": {"seed": SEED, "p_invert": 0.5, "pair_rate": PAIR_RATE,
                 "pair_model": pair_model.config_pins(),
                 "n_per_system": N_PER_SYSTEM, "n_aggregate": N_AGGREGATE,
                 "n_secondary": N_SECONDARY,
                 "native_mask_list": "deployment search detected_events.txt",
                 "excluded_golden_tic_zero_disjoint": EXCLUDED_GOLDEN},
        "timing": {"total_s": round(total_s, 1),
                   "mean_s_per_injection_first100": round(float(np.mean(timed_100)), 4),
                   "gate_2p5s_pass": bool(np.mean(timed_100) <= GATE_S_PER_INJ),
                   "flip_to_aggregate_fired": flip_to_aggregate,
                   "tripwire_90min_fired": tripwire_fired},
        "n_tests": int(len(tests)),
        "n_recovered": int(tests.recovered.sum()),
        "n_events": int(len(events)),
        "inversion_mix_by_program_class": prog_prefix_mix,
        "masking_regime_disclosure": (
            "Staged .txt files carry 4 data columns (no ecl_mask column): "
            ">=30-min-cadence files inject with ecl_mask=None while "
            "sub-30-min files rebin with catalog-recomputed masks - "
            "IDENTICAL to the benchmark campaign's behavior on the same "
            "format (H5); eclipse cadences are additionally already excised "
            "in place by the masker."),
        "disjointness": {
            "n_staged_files": len(files), "n_disjoint": len(disjoint),
            "headline_programs_disjoint_only": True,
            "secondary_includes_frozen_overlap": True},
    }
    with open(os.path.join(OUT, "campaign_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("DONE: %d tests (%d recovered) in %.1f min; mix gates: %s",
             len(tests), int(tests.recovered.sum()), total_s / 60,
             {k: v["gate_pass"] for k, v in prog_prefix_mix.items()})


if __name__ == "__main__":
    main()
