"""Run the seven probes against an exported shard, and compute the MDE.

The probes must pass BEFORE the dataset is allowed to freeze. A probe that
predicts the label from a quantity that has nothing to do with planets means the
benchmark is measuring the manufacturing process rather than the science.

Every probe result carries its own sample size, and probe 1 additionally
carries its standard error, so an underpowered check reports as underpowered
rather than quietly passing.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet.bench import probes
from cbpvet.export.incumbent import INCUMBENT_COLS
from cbpvet.export import schema

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("14_probes")


def load_shard(path):
    with h5py.File(path, "r") as h5:
        local = h5["local"][:]
        recur = h5["recurrence"][:]
        cols = {k: h5["scalars"][k][:] for k in h5["scalars"].keys()}
    df = pd.DataFrame({k: v for k, v in cols.items() if v.ndim == 1})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df, local, recur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=os.path.join(REPO, "data", "bench", "pilot", "bench_pilot.h5"))
    ap.add_argument("--scale", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--raw", action="store_true",
                    help="ignore rebalance_keep and probe the unremedied set")
    args = ap.parse_args()

    df, local, recur = load_shard(args.shard)
    log.info("Shard %s: %d events, %d positive, %d distinct TICs",
             args.shard, len(df), int((df.label == 1).sum()), df.tic.nunique())
    # Probe 1's pre-named remedy is carried in the shard as a FLAG rather than
    # applied by deletion, so the probes must honour it. Without this the gate
    # is evaluated on the unremedied set and fails by construction.
    n_before = len(df)
    if "rebalance_keep" in df.columns and not args.raw:
        keep = df.rebalance_keep.to_numpy().astype(bool)
        df, local, recur = df[keep].reset_index(drop=True), local[keep], recur[keep]
        log.info("Honouring rebalance_keep: %d of %d rows (%.1f%%)",
                 len(df), n_before, 100 * len(df) / n_before)
    elif args.raw:
        log.warning("--raw: probing the UNREMEDIED set; probe 1 will fail by construction")
    if args.scale == "pilot":
        log.warning("PILOT SCALE: split is a provisional GroupShuffleSplit by TIC. "
                    "Every result below is labelled provisional and does not "
                    "certify anything for the freeze.")

    labels = df.label.to_numpy().astype(int)
    groups = df.tic.to_numpy()

    # At full scale, certify on the shard's FROZEN split (train fits, val
    # scores), exactly the partition the arms will use. The probes module
    # docstring promised this; before 2026-08-06 the code drew its own
    # GroupShuffleSplit at every scale.
    split_masks = None
    certified_split = "provisional GroupShuffleSplit(seed 0)"
    if args.scale == "full" and "split" in df.columns:
        sp = df["split"].astype(str).to_numpy()
        tr_idx = np.flatnonzero(sp == "train")
        va_idx = np.flatnonzero(sp == "val")
        if len(tr_idx) and len(va_idx):
            split_masks = (tr_idx, va_idx)
            certified_split = "frozen shard split (train fits, val scores)"
            log.info("Probes certify on the FROZEN split: %d train / %d val events",
                     len(tr_idx), len(va_idx))
    results = []

    # ---- probe 1: inversion balance ------------------------------------
    results.append(probes.probe_1_inversion(labels, df.inverted_lc.to_numpy()))

    # ---- probe 2: provenance -------------------------------------------
    scalar_cols = [c for c in schema.CORE_SCALARS + list(INCUMBENT_COLS)
                   + schema.HOST_SCALARS if c in df.columns]
    X = df[scalar_cols].to_numpy(dtype=float)
    results.append(probes.probe_2_provenance(
        X, df.label_source.to_numpy(), labels, groups, split_masks=split_masks))

    # ---- probe 3: eclipse-phase distance --------------------------------
    cat = load_catalogue(CAT_PATH, TEBC=True).drop_duplicates("tess_id").set_index("tess_id")
    idx = df.tic.astype(int)
    have = idx.isin(cat.index)
    pd_vals = np.full(len(df), np.nan)
    sub = cat.reindex(idx[have])
    pd_vals[have.to_numpy()] = probes.phase_distance(
        df.phase.to_numpy()[have.to_numpy()],
        sub.prim_pos.to_numpy(), sub.prim_width.to_numpy(),
        sub.sec_pos.to_numpy(), sub.sec_width.to_numpy())
    df["phase_distance"] = pd_vals

    for pnum, (col, name) in probes.SINGLE_COLUMN_PROBES.items():
        if col not in df.columns:
            results.append({"probe": pnum, "name": name, "value": None,
                            "pass": None, "note": f"column {col} absent"})
            continue
        results.append(probes.probe_single_column(
            pnum, df[col].to_numpy(), labels, groups, name,
            reported_only=pnum in probes.REPORTED_ONLY, split_masks=split_masks))

    # ---- probe 4: host identity ----------------------------------------
    results.append(probes.probe_4_host_identity(local, df.tic.to_numpy().astype(int)))

    results.sort(key=lambda r: r["probe"])

    # ---- reported-only DIAGNOSTIC screens (not the seven ratified probes) --
    # Added 2026-08-06 after the stress test found a pair scalar that equalled
    # the label with no probe watching. Two families: (a) each gated pair
    # scalar as a single-column screen; (b) validity patterns - can
    # isfinite(x) alone predict the label? Neither family GATES tonight;
    # promoting any of them to a gate is a tag decision.
    diagnostics = []
    for i, col in enumerate(schema.GATED_PAIR_SCALARS):
        if col in df.columns:
            d = probes.probe_single_column(
                100 + i, df[col].to_numpy(dtype=float), labels, groups,
                f"DIAG pair scalar: {col}", reported_only=True,
                split_masks=split_masks)
            diagnostics.append(d)
    for i, col in enumerate(schema.CONDITIONAL_SCALARS):
        if col in df.columns:
            fin = np.isfinite(df[col].to_numpy(dtype=float)).astype(float)
            if 0 < fin.mean() < 1:
                d = probes.probe_single_column(
                    200 + i, fin, labels, groups,
                    f"DIAG validity pattern: isfinite({col})", reported_only=True,
                    split_masks=split_masks)
                diagnostics.append(d)

    # ---- MDE -------------------------------------------------------------
    n_tics = int(df.tic.nunique())
    pbar = float((df.label == 1).mean())
    mde_val = probes.mde(pbar, n_tics)
    target = 0.10

    print("\n" + "=" * 78)
    print(f"SEVEN PROBES  ({args.scale} scale; certified on: {certified_split})")
    print("=" * 78)
    for r in results:
        v = r.get("value")
        vs = "n/a  " if v is None else f"{v:.4f}"
        gate = r.get("gate")
        gs = "reported-only" if r.get("reported_only") else (f"<= {gate:.4f}" if gate else "n/a")
        if r.get("pass") is None:
            verdict = "SKIP"
        else:
            verdict = "PASS" if r["pass"] else "FAIL"
        print(f"  probe {r['probe']}  {r['name'][:44]:46s} {vs}  gate {gs:14s} {verdict}")
        if r.get("note"):
            print(f"            note: {r['note']}")
        if r.get("underpowered"):
            print(f"            UNDERPOWERED: 2 standard errors = {2*r['standard_error']:.4f} "
                  f"exceeds the gate; this run cannot certify probe 1")
    print("-" * 78)
    print(f"  MDE: {mde_val:.4f} absolute on {n_tics} distinct TICs at pbar={pbar:.3f}")
    print(f"       target effect {target:.2f}. "
          f"{'ADEQUATE' if mde_val <= target else 'EXCEEDS TARGET: pre-named response is to widen the stratum'}")
    print("  DIAGNOSTIC SCREENS (reported-only, never gated tonight):")
    for d in diagnostics:
        v = d.get("value"); fv = d.get("auc_folded")
        vs = " n/a " if v is None else f"{v:.4f}"
        fs = "" if fv is None else f"  folded {fv:.4f}"
        print(f"    {d['name'][:52]:54s} {vs}{fs}")
    gated = [r for r in results if r.get("pass") is not None]
    n_fail = sum(1 for r in gated if not r["pass"])
    print(f"  {len(gated)} probes evaluated, {n_fail} failing, "
          f"{len(results)-len(gated)} skipped")
    print("=" * 78 + "\n")

    out = os.path.join(os.path.dirname(args.shard), "probe_results.json")
    with open(out, "w") as fh:
        json.dump({"scale": args.scale, "shard": args.shard, "provisional": args.scale == "pilot",
                   "rebalance_applied": bool("rebalance_keep" in df.columns and not args.raw),
                   "n_events_before_rebalance": int(n_before),
                   "n_events": int(len(df)), "n_tics": n_tics, "pbar": pbar,
                   "mde_absolute": mde_val, "mde_target": target,
                   "certified_split": certified_split,
                   "probes": results, "diagnostics": diagnostics}, fh, indent=2, default=str)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
