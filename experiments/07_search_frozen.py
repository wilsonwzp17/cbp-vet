"""W3.5: run the mono-cbp search over the frozen 1,946 light curves.

This produces four never-re-run artifacts:

1. ``detected_events.txt``  the real-search negatives, plus the trailing
   ``N_DETREND_DETECTIONS`` column that only ``TransitFinderExt`` records.
2. ``event_snippets/``      per-event flux windows the exporter reads.
3. ``joint_phase_gap_hist.npz``  the 2D (eclipse-phase-distance, gap-distance)
   histogram the injection campaign importance-samples epochs from, so that
   injected positives land where real negatives actually live.
4. ``staged_sha256.txt``    hashes of the masked inputs, taken AFTER masking.

Two landmines this script exists to avoid (both verified on disk):

* The cached light curves carry a five-name header over four data columns, so
  they have no ``ECL_MASK``. Searching them as-is runs with NO eclipse masking,
  and every stellar eclipse becomes a "candidate" event.
* ``EclipseMasker.mask_file`` rewrites files IN PLACE. Pointing it at
  ``data/lc_cache`` would permanently mutate the frozen cache, and symlinking
  would do the same through the link. Therefore: copy-stage, never link.

Usage
-----
    python experiments/07_search_frozen.py                 # full run, ~23 min
    python experiments/07_search_frozen.py --limit 12      # smoke test
    python experiments/07_search_frozen.py --hist-only     # rebuild histogram
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.eclipse_masking import EclipseMasker
from mono_cbp.utils import load_catalogue

from cbpvet.search import TransitFinderExt

# ---------------------------------------------------------------- paths / pins
REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
SECTOR_TIMES = os.path.join(MONO, "catalogues", "sector_times.csv")
CACHE = os.path.join(REPO, "data", "lc_cache")
WORK = os.path.join(REPO, "data", "search_frozen")
STAGED = os.path.join(WORK, "staged")
OUT = os.path.join(WORK, "out")

# Filters applied before the histogram, from mono_cbp/config/defaults.py.
MIN_SNR = 5
MAX_DURATION_DAYS = 1.0
# det_dependence == 0 means the event survived MORE than det_dependence_threshold
# (18) of the 21 biweight windows, i.e. it is the detrending-robust class.
KEEP_DET_DEPENDENCE = 0
KEEP_SKYE_FLAG = 0

# Histogram binning (pinned; the campaign's epoch sampler reads these edges).
PHASE_BINS = np.linspace(0.0, 0.5, 11)          # 10 bins, phase distance to nearest eclipse
GAP_EDGES = np.array([0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 30], dtype=float)
GAP_THRESHOLD_DAYS = 0.5                        # a break > this counts as a gap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("07_search")


def frozen_pairs(catalogue_csv):
    """Return the frozen (tic, sector, filename) list: catalog pairs present in cache.

    The catalog's ``sectors`` column is a comma-separated string. One row per
    tess_id after de-duplication; 1,946 of the 2,068 cached files are catalog
    pairs, the other 122 being newer sector-60-plus pulls that stay out.
    """
    cat = pd.read_csv(catalogue_csv).drop_duplicates("tess_id")
    present = set(os.listdir(CACHE))
    pairs, missing = [], []
    for _, row in cat.iterrows():
        tic = int(row.tess_id)
        raw = str(row.sectors)
        for tok in raw.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok or tok == "nan":
                continue
            sector = int(float(tok))
            fname = f"TIC_{tic}_{sector:02d}.txt"
            (pairs if fname in present else missing).append((tic, sector, fname))
    return pairs, missing


def stage(pairs, limit=None):
    """Copy the frozen files into a scratch dir. Copies, never symlinks."""
    os.makedirs(STAGED, exist_ok=True)
    subset = pairs[:limit] if limit else pairs
    copied = 0
    for _, _, fname in subset:
        dst = os.path.join(STAGED, fname)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(CACHE, fname), dst)
            copied += 1
    log.info("Staged %d files into %s (%d newly copied)", len(subset), STAGED, copied)
    on_disk = [f for f in os.listdir(STAGED) if f.endswith(".txt")]
    if len(on_disk) != len(subset):
        raise RuntimeError(
            f"Staging dir holds {len(on_disk)} files but {len(subset)} were requested. "
            "Clear data/search_frozen/staged and re-run; a stale dir would silently "
            "change the frozen denominator."
        )
    return subset


def triage(max_excluded=5):
    """Quarantine unusable staged files BEFORE masking, and record exactly which.

    ``masker._load_txt`` does ``data[:, 0]`` on the result of ``np.loadtxt``. A
    file with zero or one data rows loads 1-D and raises IndexError, killing the
    whole 23-minute run. One such file exists in the frozen set
    (TIC_124912666_12.txt: header only, zero rows, empty in the original cache
    too), so it cannot be searched by anything and is excluded here.

    Exclusions change the frozen denominator, so they are enumerated to disk and
    capped: more than ``max_excluded`` means something systemic, not one bad
    download, and the run stops rather than quietly shrinking the benchmark.
    """
    excluded_dir = os.path.join(WORK, "staged_excluded")
    os.makedirs(excluded_dir, exist_ok=True)
    excluded = []
    for fname in sorted(os.listdir(STAGED)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(STAGED, fname)
        try:
            arr = np.loadtxt(path, skiprows=1)
        except Exception as exc:  # unreadable
            excluded.append({"file": fname, "reason": f"loadtxt failed: {type(exc).__name__}"})
            continue
        if arr.ndim != 2 or arr.shape[0] < 2:
            excluded.append({
                "file": fname,
                "reason": f"only {0 if arr.ndim != 2 else arr.shape[0]} data rows; "
                          "masker._load_txt indexes 2-D and would raise IndexError",
            })
    for rec in excluded:
        shutil.move(os.path.join(STAGED, rec["file"]), os.path.join(excluded_dir, rec["file"]))
    with open(os.path.join(OUT, "excluded_files.json"), "w") as fh:
        json.dump(excluded, fh, indent=2)
    if excluded:
        log.warning("EXCLUDED %d unusable file(s) from the frozen set:", len(excluded))
        for rec in excluded:
            log.warning("  %s  (%s)", rec["file"], rec["reason"])
    if len(excluded) > max_excluded:
        raise RuntimeError(
            f"{len(excluded)} files excluded, above the cap of {max_excluded}. "
            "That is systemic, not one bad download. Investigate before running."
        )
    return excluded


def hash_dir(directory, out_path):
    """SHA256 every staged file AFTER masking, so the frozen inputs are pinned."""
    lines = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".txt"):
            continue
        h = hashlib.sha256()
        with open(os.path.join(directory, fname), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {fname}")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Wrote %d hashes to %s", len(lines), out_path)


def check_masked(sample_n=5):
    """Assert masking actually happened: masked files carry 5 data columns."""
    files = sorted(f for f in os.listdir(STAGED) if f.endswith(".txt"))[:sample_n]
    for fname in files:
        arr = np.loadtxt(os.path.join(STAGED, fname), skiprows=1, max_rows=5)
        if arr.shape[1] != 5:
            raise RuntimeError(
                f"{fname} has {arr.shape[1]} data columns after masking, expected 5 "
                "(TIME FLUX FLUX_ERR PHASE ECL_MASK). The search would run UNMASKED."
            )
    log.info("Mask check passed on %d files (5 data columns present)", len(files))


def build_histogram(events_path, catalogue_csv):
    """Joint (phase-distance, gap-distance) histogram over filtered real events."""
    ev = pd.read_csv(events_path, sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    n_all = len(ev)

    keep = (
        (ev["snr"].astype(float) >= MIN_SNR)
        & (ev["duration"].astype(float) <= MAX_DURATION_DAYS)
        & (ev["det_dependence"].astype(int) == KEEP_DET_DEPENDENCE)
    )
    if "skye_flag" in ev.columns:
        keep &= ev["skye_flag"].astype(int) == KEEP_SKYE_FLAG
    ev = ev[keep].copy()
    log.info("Histogram input: %d of %d events pass the pinned filters", len(ev), n_all)

    cat = load_catalogue(catalogue_csv, TEBC=True).drop_duplicates("tess_id").set_index("tess_id")

    # --- phase distance to the nearer eclipse edge -------------------------
    d_phi = np.full(len(ev), np.nan)
    for i, (tic, ph) in enumerate(zip(ev["tic"].astype(int), ev["phase"].astype(float))):
        if tic not in cat.index:
            continue
        row = cat.loc[tic]
        dists = []
        for pos_key, w_key in (("prim_pos", "prim_width"), ("sec_pos", "sec_width")):
            pos, width = row[pos_key], row[w_key]
            if pd.isna(pos) or pd.isna(width):
                continue
            centre = abs(((ph - pos + 0.5) % 1.0) - 0.5)
            dists.append(max(0.0, centre - width / 2.0))
        if dists:
            d_phi[i] = min(dists)
    ev["d_phi"] = d_phi

    # --- time distance to the nearest data edge or intra-sector gap --------
    d_gap = np.full(len(ev), np.nan)
    time_cache = {}
    for i, (tic, sector, t_ev) in enumerate(
        zip(ev["tic"].astype(int), ev["sector"].astype(int), ev["time"].astype(float))
    ):
        key = (tic, sector)
        if key not in time_cache:
            path = os.path.join(STAGED, f"TIC_{tic}_{sector:02d}.txt")
            if not os.path.exists(path):
                time_cache[key] = None
            else:
                t = np.loadtxt(path, skiprows=1, usecols=0)
                t = np.sort(t[np.isfinite(t)])
                edges = [t[0], t[-1]]
                dt = np.diff(t)
                for j in np.where(dt > GAP_THRESHOLD_DAYS)[0]:
                    edges.extend([t[j], t[j + 1]])
                time_cache[key] = np.asarray(edges)
        edges = time_cache[key]
        if edges is not None:
            d_gap[i] = np.min(np.abs(edges - t_ev))
    ev["d_gap"] = d_gap

    ok = np.isfinite(ev["d_phi"]) & np.isfinite(ev["d_gap"])
    log.info("Both coordinates computable for %d of %d filtered events", int(ok.sum()), len(ev))
    sub = ev[ok]

    H, _, _ = np.histogram2d(
        sub["d_phi"].to_numpy(), sub["d_gap"].to_numpy(), bins=[PHASE_BINS, GAP_EDGES]
    )
    npz_path = os.path.join(OUT, "joint_phase_gap_hist.npz")
    np.savez(
        npz_path, H=H, phase_edges=PHASE_BINS, gap_edges=GAP_EDGES,
        n_events=len(sub), n_filtered=len(ev), n_all=n_all,
    )
    ev.to_csv(os.path.join(OUT, "detected_events_filtered.csv"), index=False)
    log.info("Wrote %s  (H sum = %d, %d empty bins of %d)",
             npz_path, int(H.sum()), int((H == 0).sum()), H.size)
    return H, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stage only N files (smoke test)")
    ap.add_argument("--hist-only", action="store_true", help="rebuild the histogram only")
    ap.add_argument("--force-restage", action="store_true")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    events_path = os.path.join(OUT, "detected_events.txt")

    if args.hist_only:
        build_histogram(events_path, CAT_PATH)
        return

    if args.force_restage and os.path.isdir(STAGED):
        shutil.rmtree(STAGED)

    pairs, missing = frozen_pairs(CAT_PATH)
    log.info("Frozen catalog pairs present in cache: %d (missing %d)", len(pairs), len(missing))
    if not args.limit and len(pairs) != 1946:
        raise RuntimeError(f"Expected 1946 frozen pairs, reconstructed {len(pairs)}")
    max_sector = max(s for _, s, _ in (pairs[:args.limit] if args.limit else pairs))
    n_sector_rows = len(pd.read_csv(SECTOR_TIMES, comment="#"))
    if max_sector > n_sector_rows:
        raise RuntimeError(
            f"Max staged sector {max_sector} exceeds sector_times rows {n_sector_rows}; "
            "the Skye metric indexes values[sector-1] and would raise IndexError."
        )

    subset = stage(pairs, limit=args.limit)
    excluded = triage()
    n_searchable = len(subset) - len(excluded)
    log.info("Searchable frozen files: %d of %d catalog pairs", n_searchable, len(subset))

    t0 = _time.time()
    EclipseMasker(catalogue=CAT_PATH, data_dir=STAGED, TEBC=True).mask_all()
    log.info("Masking took %.1f s", _time.time() - t0)
    check_masked()
    hash_dir(STAGED, os.path.join(OUT, "staged_sha256.txt"))

    t0 = _time.time()
    finder = TransitFinderExt(catalogue=CAT_PATH, sector_times=SECTOR_TIMES, TEBC=True)
    df = finder.process_directory(STAGED, output_file="detected_events.txt", output_dir=OUT)
    elapsed = _time.time() - t0
    log.info("Search took %.1f s over %d files (%.2f s/file); %d events",
             elapsed, len(subset), elapsed / max(len(subset), 1), len(df))

    with open(os.path.join(OUT, "run_meta.json"), "w") as fh:
        json.dump(
            {
                "n_catalog_pairs": len(subset),
                "n_excluded": len(excluded),
                "excluded_files": [r["file"] for r in excluded],
                "n_files_searched": n_searchable,
                "n_events_all": int(len(df)),
                "search_seconds": round(elapsed, 1),
                "min_snr": MIN_SNR,
                "max_duration_days": MAX_DURATION_DAYS,
                "det_dependence_kept": KEEP_DET_DEPENDENCE,
                "skye_flag_kept": KEEP_SKYE_FLAG,
                "catalogue": CAT_PATH,
                "sector_times": SECTOR_TIMES,
            },
            fh,
            indent=2,
        )

    build_histogram(events_path, CAT_PATH)


if __name__ == "__main__":
    main()
