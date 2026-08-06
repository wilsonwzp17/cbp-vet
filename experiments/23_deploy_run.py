"""The deployment search driver: mask + find over a staged directory.

What this is
------------
The Aug-9 deployment run's engine, written and SMOKE-TESTED on 2026-08-06 so
the overnight fork starts from a proven script instead of a recipe. It is the
same mask -> find chain the frozen search (experiment 07) ran, parameterized,
because the deployment TCEs must be produced by the exact code path every
benchmark event went through (invariant 3): ``EclipseMasker`` on staged
COPIES, then ``TransitFinderExt`` so ``n_detrend_detections`` is captured
(the deployment events will be exported through the frozen exporter, whose
schema expects it).

Usage (each run gets its own staged dir and out dir; staging is consumed):
    python experiments/23_deploy_run.py --data-dir data/deploy_staged/lc \
        --catalogue ~/mono-cbp/catalogues/TEBC_morph_05_P_7.csv \
        --sector-times data/deploy_staged/sector_times_extended.csv \
        --out data/deploy_run/out --label full63

    # TIC 172900988 (W3.11 leftover): scratch copies of the B10 pull,
    # the one-row supplemental catalogue, same extended sector times.

Hard rules carried from experiment 07, enforced not assumed:
- the masker rewrites files IN PLACE, so --data-dir must never point into a
  cache; this script asserts the dir is not one of the known caches;
- triage BEFORE masking: files with <2 data rows crash masker._load_txt
  (verified on the frozen set); they are quarantined, enumerated, and capped;
- the Skye IndexError: max staged sector must be covered by the sector-times
  rows, asserted before the finder runs (the whole point of B11);
- every input file is SHA-256'd after masking, so the run's inputs are pinned
  in the output manifest.

Privacy: shortlist TICs and any candidate identities never leave local disk;
this script writes only under data/ (untracked).
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.eclipse_masking import EclipseMasker

from cbpvet.search.finder_ext import TransitFinderExt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN_DATA_DIRS = {"lc_cache", "lc_cache_qlp", "tic172900988"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("23_deploy")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def triage(data_dir, out_dir, max_excluded=5):
    """Quarantine files the masker would crash on, before it runs."""
    qdir = os.path.join(out_dir, "staged_excluded")
    os.makedirs(qdir, exist_ok=True)
    excluded = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        try:
            arr = np.loadtxt(os.path.join(data_dir, fname), skiprows=1)
        except Exception as exc:
            excluded.append({"file": fname,
                             "reason": f"loadtxt failed: {type(exc).__name__}"})
            continue
        if arr.ndim != 2 or arr.shape[0] < 2:
            excluded.append({
                "file": fname,
                "reason": f"only {0 if arr.ndim != 2 else arr.shape[0]} data "
                          "rows; masker._load_txt indexes 2-D"})
    for rec in excluded:
        shutil.move(os.path.join(data_dir, rec["file"]),
                    os.path.join(qdir, rec["file"]))
        log.warning("EXCLUDED %s (%s)", rec["file"], rec["reason"])
    if len(excluded) > max_excluded:
        raise RuntimeError(f"{len(excluded)} exclusions exceeds the cap "
                           f"{max_excluded}: systemic, investigate first.")
    return excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--sector-times", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tebc", action="store_true",
                    help="catalogue is TEBC-format (pos/width derived from "
                         "2g/pf columns); omit for a supplemental catalogue "
                         "that carries prim/sec pos+width directly")
    args = ap.parse_args()

    data_dir = os.path.realpath(args.data_dir)   # realpath: a symlink must not hide a cache
    out_dir = os.path.abspath(args.out)
    parts = {os.path.basename(data_dir.rstrip("/")),
             os.path.basename(os.path.dirname(data_dir))}
    if parts & FORBIDDEN_DATA_DIRS:              # raise, not assert: survives python -O
        raise RuntimeError(
            f"--data-dir {data_dir} looks like a cache; the masker rewrites "
            "in place. Stage COPIES and point at the staging dir.")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in sorted(os.listdir(data_dir)) if f.endswith(".txt")]
    if not files:
        raise RuntimeError(f"no .txt light curves in {data_dir}")
    sectors = [int(f.rsplit("_", 1)[1][:-4]) for f in files]
    st = pd.read_csv(args.sector_times)
    if max(sectors) > len(st):                    # raise, not assert: survives python -O
        raise RuntimeError(
            f"max staged sector {max(sectors)} exceeds sector-times rows "
            f"{len(st)}; the Skye metric would IndexError (B11).")
    if list(st.Sector) != list(range(1, len(st) + 1)):
        raise RuntimeError(
            "sector-times rows must be contiguous from 1 (values[sector-1] indexing)")

    excluded = triage(data_dir, out_dir)
    t0 = time.time()
    log.info("masking %d files in place (staged copies) ...",
             len(files) - len(excluded))
    EclipseMasker(catalogue=args.catalogue, data_dir=data_dir,
                  TEBC=args.tebc).mask_all()
    t_mask = time.time() - t0

    t0 = time.time()
    finder = TransitFinderExt(catalogue=args.catalogue,
                              sector_times=args.sector_times, TEBC=args.tebc)
    finder.process_directory(data_dir,
                             output_file=os.path.join(out_dir,
                                                      "detected_events.txt"))
    t_find = time.time() - t0

    ev_path = os.path.join(out_dir, "detected_events.txt")
    ev = pd.read_csv(ev_path, sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    if len(ev):
        assert "n_detrend_detections" in ev.columns, \
            "finder_ext column missing; the export schema needs it"
    else:
        log.info("zero events flagged; finder_ext writes the extra column "
                 "only when events exist")

    hashes = {f: sha256(os.path.join(data_dir, f))
              for f in sorted(os.listdir(data_dir)) if f.endswith(".txt")}
    meta = {
        "label": args.label,
        "data_dir": data_dir,
        "catalogue": os.path.abspath(args.catalogue),
        "sector_times": os.path.abspath(args.sector_times),
        "n_files_searched": len(files) - len(excluded),
        "excluded": excluded,
        "n_events": int(len(ev)),
        "n_tics_with_events": int(ev.tic.nunique()) if len(ev) else 0,
        "runtime_s": {"mask": round(t_mask, 1), "find": round(t_find, 1)},
        "masked_input_sha256": hashes,
        "detected_events_sha256": sha256(ev_path),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("%s: %d files -> %d events (%d TICs); mask %.1fs find %.1fs",
             args.label, meta["n_files_searched"], meta["n_events"],
             meta["n_tics_with_events"], t_mask, t_find)


if __name__ == "__main__":
    main()
