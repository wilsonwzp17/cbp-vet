"""THE ONE-SHOT: the sealed evaluation against the real planets. Runs ONCE, Aug 11.

What this is
------------
The single permitted look at how the frozen model scores the known real
circumbinary planets (D1 primary denominator) and the manual candidate. It
is written and adversarially reviewed on 2026-08-07 against SYNTHETIC
FIXTURES ONLY; it must never be executed against real inputs before the
morning of Aug 11 (ground rule 3 of the final-run brief). There is no human
branch inside: every choice is read from the declared config, every
precondition is an assert, and the outputs are a pure function of the
frozen inputs.

The discipline, in execution order (Amendment-1 3B.10 i-viii + adoption H7):
1. OPENING ASSERTS, before anything is read: the checkpoint file hashes to
   the value in the pre-shot hash log (data/bench/full/
   preshot_hashlog_2026-08-07.json); the operating threshold equals the
   hash log's value; each declared times table hashes to its declared
   sha256; the frozen shard hashes are NOT recomputed here (158 MB) but the
   declared per-TIC row counts are asserted after load - a mismatch means
   the declaration is stale and the run ABORTS with nothing consumed (fix
   the declaration by dated annotation per the degradation ladder, then
   run; since nothing was scored, that abort is not a second evaluation).
2. POOLS. TESS: the two frozen-cache hosts' records are read from the
   FROZEN bench-v1 shard (their real_search rows have been there, sealed by
   the test split, since the freeze); the third TESS host's 21 events are
   exported fresh through the FROZEN exporter code path (identical chain to
   25_deploy_score, validated by the deployment pre-flight). Kepler
   (declared tier per adoption H4): each host's staged masked quarter files
   + its catalog row (from the declared config) go through the same frozen
   exporter; per 3B.10-vi the recurrence input is CYCLE-SUBSAMPLED to the
   TESS training n_cycles distribution by an input-level cadence filter
   (the exporter itself is never modified): keep every cadence within the
   local window of the event, keep all cadences of n* randomly selected
   OTHER binary cycles (n* inverse-CDF-drawn from the frozen train-split
   TESS n_cycles_raw distribution, config seed), drop the rest. The
   uncapped export runs as the labelled secondary. A KS statistic between
   the subsampled Kepler n* draw and the TESS train distribution is
   recorded in the output.
3. SCORE ONCE: arms.score_matrix (the single gate) + the hash-verified
   arm-b checkpoint + the frozen threshold. No calibration exists
   (temperature: none, per the hash log).
4. D1/D2 MATCHING per planet: known transit times from the transcribed
   PUBLIC tables (tess/kepler_transit_times.csv), filtered to rows whose
   in-window flag is set; D2 = known times inside our data windows; D1 =
   flagged events within tolerance of a known time, tolerance = max(
   t_dur/2, published per-row t_unc) per adoption H7 (3B.10-i's t_dur/2
   alone would fail day-precision literature times for reasons that are
   the literature's precision, not the pipeline's). Every D1 row carries
   the upper-bound sentence (3B.10-ii). TESS numbers are COUNTS with
   Wilson intervals, never percentages (3B.10-iii). The TIC-319011894
   candidate is scored and reported as replication-not-confirmation (its
   "known time" is itself a mono-cbp detection in the same data).
5. OUTPUTS: per-planet rows + per-mission rollup + the candidate row,
   SHA-256 hashed, written once. The bug rule (3B.10-iv) binds the humans
   around this script, not the script: a bug is a crash, NaN, empty
   output, shape mismatch, or disagreement with a pre-committed fixture -
   never a defect inferred from the result's value; at most one corrected
   second evaluation.

Fixtures: experiments/26_fixtures.py generates the synthetic config +
shard + times tables with hand-computable expected outputs; reviewers and
CI exercise THIS script only through them until Aug 11.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import logging
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("26_one_shot")

Z = 1.96


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k, n):
    """Closed-form Wilson interval, z = 1.96."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_shard_rows(shard_path, tics):
    with h5py.File(shard_path, "r") as f:
        cols = {k: f["scalars"][k][:] for k in f["scalars"].keys()}
    df = pd.DataFrame({k: v for k, v in cols.items() if v.ndim == 1})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df[df.tic.astype(int).isin([int(t) for t in tics])].reset_index(drop=True)


def subsample_cycles(time, phase, event_time, event_phase, t_dur, period,
                     n_star, rng):
    """3B.10-vi as an input-level cadence filter; the exporter is untouched.

    Keeps: cadences within the local window of the event (its own cycle's
    data must stay intact), plus all cadences of n_star randomly selected
    OTHER cycles. Returns a boolean keep-mask over the cadence array.
    """
    t = np.asarray(time, dtype=float)
    cyc = np.floor((t - float(event_time)) / float(period) + 0.5).astype(int)
    local = np.abs(t - float(event_time)) <= max(3.0 * float(t_dur), 0.75)
    others = np.unique(cyc[(cyc != 0) & ~local])
    if len(others) > n_star:
        chosen = set(rng.choice(others, size=n_star, replace=False).tolist())
    else:
        chosen = set(others.tolist())
    keep = local | (cyc == 0) | np.isin(cyc, list(chosen))
    return keep


def export_host(host_cfg, out_dir, comparator_workers, subsample=None):
    """One host's flagged events through the FROZEN exporter (25's chain)."""
    from mono_cbp.utils import load_catalogue
    from cbpvet.export import EventExporter, build_block, schema, write_shard

    if "events_sha256" in host_cfg:
        got = sha256(host_cfg["events"])
        if got != host_cfg["events_sha256"]:
            raise RuntimeError(f"{host_cfg['name']}: events file hash {got} "
                               f"!= declared {host_cfg['events_sha256']}; ABORT")
    ev = pd.read_csv(host_cfg["events"], sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    ev = ev.rename(columns={"time": "event_time"})
    ev["label"] = 0
    ev["inverted_lc"] = 0

    if host_cfg.get("catalogue_row"):
        cat = pd.DataFrame([host_cfg["catalogue_row"]])
        raw = cat.copy()
    else:
        cat = load_catalogue(os.path.expanduser(host_cfg["catalogue"]), TEBC=True)
        raw = pd.read_csv(os.path.expanduser(host_cfg["catalogue"]))
    noise = pd.read_csv(os.path.join(REPO, "data", "noise_screen.csv")) \
        if host_cfg.get("use_noise", True) else None

    records, locals_, recurs, keys, payloads = [], [], [], [], []
    if subsample is not None and "subsample_seed" not in host_cfg:
        raise RuntimeError(f"{host_cfg['name']}: subsample_seed must be "
                           "DECLARED (3B.10-vi requires the manifest seed)")
    rng = np.random.default_rng(host_cfg.get("subsample_seed", 0))
    exp = EventExporter(host_cfg["staged_lc"], cat, raw, noise_screen=noise,
                        e2_passed=True)
    exp.set_system_events(ev, key=("tic",))
    skipped_missing = 0
    for _, e in ev.iterrows():
        use_exp = exp
        if subsample is not None:
            # 3B.10-vi as a REAL input-level cadence filter (reviewers proved
            # local_flux_source is a no-op for the recurrence path): write
            # per-event filtered COPIES of every staged file of this host and
            # export the event from that scratch staging, so _all_sectors and
            # the recurrence view genuinely see the subsampled cadences. The
            # frozen exporter code stays untouched.
            cached = exp._cached(e.tic, e.sector)
            if cached is None:
                skipped_missing += 1
                continue
            ev_dir = os.path.join(out_dir, "subsampled", host_cfg["name"],
                                  str(len(records)))
            os.makedirs(ev_dir, exist_ok=True)
            n_star = int(subsample["draw"](rng))
            import glob
            for fpath in sorted(glob.glob(os.path.join(
                    host_cfg["staged_lc"], f"TIC_{int(e.tic)}_*.txt"))):
                arr = np.loadtxt(fpath, skiprows=1)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                keep = subsample_cycles(arr[:, 0], None, float(e.event_time),
                                        None, float(e.duration),
                                        float(host_cfg["p_bin"]), n_star, rng)
                with open(fpath) as fh:
                    header = fh.readline()
                with open(os.path.join(ev_dir, os.path.basename(fpath)), "w") as fh:
                    fh.write(header)
                    np.savetxt(fh, arr[keep], fmt="%.10g")
            use_exp = EventExporter(ev_dir, cat, raw, noise_screen=noise,
                                    e2_passed=True)
            use_exp.set_system_events(ev, key=("tic",))
        cached = use_exp._cached(e.tic, e.sector)
        if cached is None:                      # reviewer M2
            skipped_missing += 1
            continue
        built = use_exp.build_record(e, "real_search")
        if built is None:
            continue
        rec, loc, rc = built
        rec["pair_model_version"] = 0
        rec["source_dir"] = f"oneshot_{host_cfg['name']}"
        event_no = len(records)
        rec["event_key"] = f"TIC_{int(e.tic)}_{int(e.sector)}_{event_no}"
        records.append(rec); locals_.append(loc); recurs.append(rc)
        keys.append(rec["event_key"])
        cached = use_exp._cached(e.tic, e.sector)
        half = schema.local_halfwidth(float(e.duration))
        sel = np.abs(cached["time"] - float(e.event_time)) <= half
        payloads.append({"time": cached["time"][sel], "flux": cached["flux"][sel],
                         "flux_err": cached["flux_err"][sel],
                         "event_time": float(e.event_time),
                         "event_width": float(e.duration),
                         "tic": int(e.tic), "sector": int(e.sector),
                         "event_no": event_no})
    if not records:
        raise RuntimeError(f"{host_cfg['name']}: no records built")

    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import importlib
    exp13 = importlib.import_module("13_export")
    rows, _ = exp13.run_comparator(payloads, out_dir, workers=comparator_workers)
    inc = build_block(rows, keys)
    for rec in records:
        rec.update({f"valid_{k}": v for k, v in exp.scalar_valid(rec).items()})
        rec["rebalance_keep"] = 1
        rec["split"] = "oneshot"
    shard = os.path.join(out_dir, f"oneshot_{host_cfg['name']}.h5")
    return write_shard(shard, records, locals_, recurs, inc)


def match_d1(events_df, times_df, tol_rule):
    """D1: flagged events within tolerance of a known transit time."""
    matched_idx = set()
    matched_times = set()
    for ti, trow in times_df.iterrows():
        t_known = float(trow["t_transit_btjd"])
        t_unc = float(trow.get("t_unc_days", 0.0) or 0.0)
        for ei, erow in events_df.iterrows():
            tol = max(float(erow["t_dur"]) / 2.0, t_unc) \
                if tol_rule == "H7" else float(erow["t_dur"]) / 2.0
            if abs(float(erow["event_time"]) - t_known) <= tol:
                matched_idx.add(ei)
                matched_times.add(ti)
    return matched_idx, matched_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="the declared config (data/oneshot/declared_config.json "
                         "for the real run; the fixture config for review)")
    ap.add_argument("--comparator-workers", type=int, default=8)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. opening asserts ------------------------------------------------
    hlog = json.load(open(cfg["hash_log"]))
    ck = hlog["scoring_model"]
    got = sha256(os.path.join(cfg.get("repo_root", REPO), ck["path"]))
    if got != ck["sha256"]:
        raise RuntimeError(f"checkpoint hash {got} != hash log {ck['sha256']}; ABORT")
    tau = float(hlog["operating_threshold"]["value"])
    if not (tau == float(cfg["tau"])):      # NaN-safe exact equality
        raise RuntimeError("declared tau differs from the hash log; ABORT")
    if cfg.get("frozen_shard_sha256"):
        got = sha256(cfg["frozen_shard"])
        if got != cfg["frozen_shard_sha256"]:
            raise RuntimeError(f"frozen shard hash {got} != declared; ABORT")
    if not str(cfg.get("frozen_shard", "")).find("fixtures") >= 0:
        if not cfg.get("armed"):
            raise RuntimeError(
                "config targets NON-FIXTURE data but is not armed: true. "
                "The real one-shot config must set armed (the Aug-11 guard).")
    for tbl in cfg["times_tables"]:
        got = sha256(tbl["path"])
        if got != tbl["sha256"]:
            raise RuntimeError(f"times table {tbl['path']} hash mismatch; ABORT")
    log.info("opening asserts PASS: checkpoint %s..., tau %.6f, %d times tables",
             ck["sha256"][:12], tau, len(cfg["times_tables"]))

    import xgboost as xgb
    from cbpvet.models import arms
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(cfg.get("repo_root", REPO), ck["path"]))

    # ---- 2. pools ----------------------------------------------------------
    pools = []          # (host_name, mission, kind, df)
    for h in cfg["frozen_shard_hosts"]:
        df = load_shard_rows(cfg["frozen_shard"], [h["tic"]])
        if len(df) != int(h["expected_rows"]):
            raise RuntimeError(
                f"{h['name']}: frozen shard has {len(df)} rows, declaration "
                f"says {h['expected_rows']}; declaration stale, ABORT")
        pools.append((h["name"], h["mission"], h.get("kind", "planet"), df))
    for h in cfg.get("export_hosts", []):
        df = export_host(h, out_dir, args.comparator_workers)
        if "expected_rows" in h and len(df) != int(h["expected_rows"]):
            raise RuntimeError(f"{h['name']}: exported {len(df)} rows vs "
                               f"declared {h['expected_rows']}; ABORT")
        pools.append((h["name"], h["mission"], h.get("kind", "planet"), df))
    ks_records = {}
    for h in cfg.get("kepler_hosts", []):
        train_dist = np.asarray(cfg["tess_train_n_cycles"], dtype=float)
        def draw(rng, d=train_dist):
            return int(rng.choice(d))     # true empirical draw (reviewer M3)
        df = export_host(h, out_dir, args.comparator_workers,
                         subsample={"draw": draw})
        if "expected_rows" in h and len(df) != int(h["expected_rows"]):
            raise RuntimeError(f"{h['name']}: exported {len(df)} rows vs "
                               f"declared {h['expected_rows']}; ABORT")
        pools.append((h["name"], "Kepler", "planet", df))
        df_unc = export_host({**h, "name": h["name"] + "_uncapped"}, out_dir,
                             args.comparator_workers)
        pools.append((h["name"] + "_uncapped", "Kepler", "secondary", df_unc))
        if "n_cycles_raw" in df.columns:
            from scipy import stats
            ks = stats.ks_2samp(df["n_cycles_raw"].to_numpy(dtype=float),
                                train_dist)
            ks_records[h["name"]] = {"ks_stat": float(ks.statistic),
                                     "p": float(ks.pvalue)}

    # ---- 3. score once -----------------------------------------------------
    missions = [t["mission"] for t in cfg["times_tables"]]
    if len(missions) != len(set(missions)):
        raise RuntimeError("duplicate mission in times_tables; ABORT")
    times = {t["mission"]: pd.read_csv(t["path"]) for t in cfg["times_tables"]}
    results, mission_roll = [], {}
    for name, mission, kind, df in pools:
        X, feats = arms.score_matrix(df, "b")
        if len(feats) != int(ck["n_features"]):
            raise RuntimeError(f"{name}: {len(feats)} features vs hash log "
                               f"{ck['n_features']}; ABORT")
        s = model.predict_proba(X)[:, 1]
        df = df.copy()
        df["score"] = s
        df["passed"] = (s >= tau).astype(int)

        tt = times.get(mission)
        if tt is None and kind == "planet":     # reviewer M1: never silent D1=0
            raise RuntimeError(f"{name}: no times table declared for mission "
                               f"{mission}; a planet pool cannot be evaluated")
        planet_rows = tt[tt.tic.astype(int).isin(df.tic.astype(int).unique())] \
            if tt is not None else pd.DataFrame()
        if cfg.get("in_window_only", True) and len(planet_rows) and \
                "in_our_cache_window" in planet_rows.columns:
            planet_rows = planet_rows[planet_rows.in_our_cache_window.astype(str)
                                      .str.startswith("yes")]
        matched_idx, matched_times = match_d1(df, planet_rows, tol_rule="H7")
        d1 = df.loc[sorted(matched_idx)] if matched_idx else df.iloc[0:0]
        k = int(d1.passed.sum())
        n = int(len(d1))
        lo, hi = wilson(k, n)
        row = {
            "host": name, "mission": mission, "kind": kind,
            "n_flagged_events": int(len(df)),
            "D2_known_times_in_windows": int(len(planet_rows)),
            "D1_flagged_at_known_times": n,
            "D1_vetter_pass": k,
            "wilson_95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "n_flagged_passing_tau_total": int(df.passed.sum()),
            "upper_bound_sentence": (
                "D1 contains only events our search recovered, so this is "
                "performance on the search's easiest real events, an upper "
                "bound."),
        }
        results.append(row)
        if kind == "planet":
            mr = mission_roll.setdefault(mission, {"D1": 0, "pass": 0, "D2": 0})
            mr["D1"] += n; mr["pass"] += k; mr["D2"] += row["D2_known_times_in_windows"]

    for m, mr in mission_roll.items():
        lo, hi = wilson(mr["pass"], mr["D1"])
        mr["wilson_95"] = [round(lo, 4), round(hi, 4)] if lo is not None else None
        mr["reporting"] = ("counts with Wilson intervals, never percentages"
                           if m == "TESS" else "counts and fraction")

    out = {
        "run": cfg.get("run_label", "one_shot"),
        "executed_under": "3B.10 i-viii + H4 tier + H7 tolerance; no human branch",
        "checkpoint_sha256": ck["sha256"],
        "tau": tau,
        "temperature": "none (hash log)",
        "declared_tier": cfg.get("declared_tier"),
        "per_host": results,
        "per_mission": mission_roll,
        "kepler_ncycles_ks": ks_records,
        "notes": [
            "TIC 319011894 rows (kind=candidate) demonstrate replication, not "
            "independent confirmation: the matched time is itself a mono-cbp "
            "detection in the same data.",
            "D1 semantics, pre-declared: D1 counts FLAGGED EVENTS matching a "
            "known time (an event matching two times counts once; two events "
            "matching one time count twice - a duplicate detection at one "
            "transit enlarges the denominator).",
            "Per-row D1 tolerance = max(t_dur/2, published t_unc) per "
            "adoption H7.",
        ],
    }
    res_path = os.path.join(out_dir, "oneshot_results.json")
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    digest = sha256(res_path)
    with open(os.path.join(out_dir, "oneshot_results.sha256"), "w") as fh:
        fh.write(digest + "\n")
    log.info("ONE-SHOT COMPLETE. results %s sha256 %s", res_path, digest[:16])


if __name__ == "__main__":
    main()
