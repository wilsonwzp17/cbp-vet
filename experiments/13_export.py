"""W3.1: export every event, from every provenance, into frozen HDF5 shards.

This is the step that turns "we ran a search and a campaign" into "we have a
dataset". After it, nothing downstream ever touches a light curve again: the
models, the seven probes, the head-to-head table and the one-shot all read
shards.

Order of operations matters and is enforced here:

1. Real-search negatives are read from ``detected_events.txt``, and their local
   views are re-cut from the staged masked files at the WIDE half-width, because
   the snippets the finder saved use its narrower stock window and mixing widths
   across classes would be a provenance tell.
2. Campaign events are read from ``events_all.csv``; their local views come from
   the saved wide snippets, which is the only place the injected flux exists.
3. The incumbent's 17 columns are computed and joined in BEFORE writing, by
   outer join on filename. After the freeze, arm (b) cannot be repaired.
4. Everything is written with a ``scalar_valid`` bit vector, so undefined
   scalars are declared rather than imputed.

Usage
-----
    python experiments/13_export.py --limit 400          # pilot export, timed
    python experiments/13_export.py                      # full export
    python experiments/13_export.py --no-comparator      # skip the slow step
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import sys
import time as _time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.model_comparison import ModelComparator
from mono_cbp.utils import load_catalogue

from cbpvet.export import EventExporter, build_block, schema, write_shard
from cbpvet.injection.rebalance import rebalance_negatives
from cbpvet.bench import split as splitmod
from cbpvet.export.incumbent import INCUMBENT_COLS

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
STAGED = os.path.join(REPO, "data", "search_frozen", "staged")
EVENTS = os.path.join(REPO, "data", "search_frozen", "out", "detected_events.txt")
NOISE = os.path.join(REPO, "data", "noise_screen.csv")
OUT = os.path.join(REPO, "data", "bench")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("13_export")


def load_real_events(limit=None):
    df = pd.read_csv(EVENTS, sep=r"\s+")
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"time": "event_time"})
    df["label"] = 0
    df["inverted_lc"] = 0
    if limit:
        df = df.head(limit)
    log.info("Real-search events: %d", len(df))
    return df


def load_campaign_events(campaign_dir, limit=None):
    path = os.path.join(campaign_dir, "events_all.csv")
    if not os.path.exists(path):
        log.warning("No campaign events at %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    log.info("Campaign events: %d (%d positive)", len(df), int((df.label == 1).sum()))
    return df


def comparator_key(tic, sector, event_no):
    """Reproduce the key ``compare_event`` builds for dict inputs.

    MEASURED TRAP, 2026-08-04: for a dict input the comparator IGNORES any
    'filename' the caller supplies and constructs its own from three other keys::

        tic = event_input.get('tic', 'unknown')
        sector = event_input.get('sector', 'unknown')
        event_no = event_input.get('event_no', 0)
        filename = f"TIC_{tic}_{sector}_{event_no}"

    Passing 'filename' and not those three gave every one of 300 events the
    same key, "TIC_unknown_unknown_0", so the outer join matched nothing and the
    entire 17-column block came back zero and invalid. The guard caught it; a
    positional join would have silently mis-assigned all 300 rows.

    ``event_no`` is the GLOBAL record index, so the key is unique even when one
    (tic, sector) contributes many events.
    """
    return f"TIC_{tic}_{sector}_{event_no}"


def _compare_chunk(chunk):
    comp = ModelComparator()
    return comp.compare_events(chunk, output_file="chunk.csv", output_dir="/tmp/cbpvet_cmp")


def run_comparator(event_payloads, out_dir, workers=1):
    """The incumbent, on the same events we export.

    ``compare_event`` is stateless, so above the 2 h projection the recipe calls
    for sharding across a pool rather than running inline.
    """
    t0 = _time.time()
    if workers > 1 and len(event_payloads) > workers:
        os.makedirs("/tmp/cbpvet_cmp", exist_ok=True)
        chunks = [event_payloads[i::workers] for i in range(workers)]
        with Pool(workers) as pool:
            parts = pool.map(_compare_chunk, chunks)
        rows = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    else:
        comp = ModelComparator()
        rows = comp.compare_events(event_payloads, output_file="model_comparison.csv",
                                   output_dir=out_dir)
    elapsed = _time.time() - t0
    per_event = elapsed / max(len(event_payloads), 1)
    log.info("Comparator: %d rows from %d submitted, %.1f s (%.4f s/event, %d workers)",
             len(rows) if rows is not None else 0, len(event_payloads),
             elapsed, per_event, workers)
    if rows is not None and len(rows) < len(event_payloads):
        log.warning("Comparator DROPPED %d events silently; the outer join keeps "
                    "their rows with incumbent_valid = 0",
                    len(event_payloads) - len(rows))
    if rows is not None and len(rows):
        rows.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    return rows, per_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--campaign-dir", default=os.path.join(REPO, "data", "pilot_out"),
                    help="primary bank campaign directory")
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="additional injection directories (pair re-injection, ELC batch)")
    ap.add_argument("--rebalance", action="store_true",
                    help="flag negatives to drop so the class inversion rates match "
                         "(probe 1's pre-named remedy)")
    ap.add_argument("--no-comparator", action="store_true")
    ap.add_argument("--e2-passed", action="store_true",
                    help="admit the four gated pair scalars as valid features")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--comparator-workers", type=int, default=12)
    ap.add_argument("--freeze-split", action="store_true",
                    help="assign the frozen train/val/test split by TIC and verify it")
    args = ap.parse_args()

    out_dir = os.path.join(OUT, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    cat = load_catalogue(CAT_PATH, TEBC=True)
    raw = pd.read_csv(CAT_PATH)
    noise = pd.read_csv(NOISE)
    exp = EventExporter(STAGED, cat, raw, noise_screen=noise, e2_passed=args.e2_passed)

    records, locals_, recurs, keys, payloads = [], [], [], [], []
    t_start = _time.time()

    # ---- real-search negatives ------------------------------------------
    real = load_real_events(args.limit)
    # Real events of one TIC were all flagged by the same search run.
    exp.set_system_events(real, key=("tic",))
    for i, ev in real.iterrows():
        built = exp.build_record(ev, "real_search")   # local view from cached file
        if built is None:
            continue
        rec, loc, rc = built
        rec["pair_model_version"] = 0    # not an injection
        rec["source_dir"] = "real_search"
        event_no = len(records)          # GLOBAL index, keeps the key unique
        rec["event_key"] = comparator_key(int(ev.tic), int(ev.sector), event_no)
        records.append(rec); locals_.append(loc); recurs.append(rc); keys.append(rec["event_key"])
        cached = exp._cached(ev.tic, ev.sector)
        half = schema.local_halfwidth(float(ev.duration))
        sel = np.abs(cached["time"] - float(ev.event_time)) <= half
        payloads.append({"time": cached["time"][sel], "flux": cached["flux"][sel],
                         "flux_err": cached["flux_err"][sel],
                         "event_time": float(ev.event_time),
                         "event_width": float(ev.duration),
                         "tic": int(ev.tic), "sector": int(ev.sector),
                         "event_no": event_no})

    # ---- campaign events -------------------------------------------------
    sources = [(args.campaign_dir, "bank_injection")]
    def dir_pair_model_version(d):
        """Read the pair-model version the run RECORDED about itself.

        Guessing from the path name mislabelled data twice (an "elc" basename
        collision, then a pairfix dir whose model had been upgraded under it).
        The campaign writes pins.pair_model_version into its own summary, so
        that is the source of truth; ELC dirs have no bank pair model at all
        and get -1, meaning "not applicable: pairs are real dynamics".
        """
        sj = os.path.join(d, "campaign_summary.json")
        if os.path.exists(sj):
            try:
                pins = json.load(open(sj)).get("pins", {})
                if "pair_model_version" in pins:
                    return int(pins["pair_model_version"])
            except Exception:
                pass
        if "elc" in os.path.abspath(d).lower():
            return -1
        return 1

    for d in args.extra_dirs:
        # Test the FULL path: data/elc_batch/pilot basenames to "pilot", so
        # matching on the basename silently labelled ELC events as bank ones.
        # Caught by the smoke test 2026-08-04 via an absent elc_injection count.
        src = "elc_injection" if "elc" in os.path.abspath(d).lower() else "bank_injection"
        sources.append((d, src))

    for camp_dir, label_source in sources:
      source_version = dir_pair_model_version(camp_dir)
      camp = load_campaign_events(camp_dir, args.limit)
      # One TEST is one injection: events must relate only within it.
      exp.set_system_events(camp, key=("tic", "sector", "model_idx"))
      snip_dir = os.path.join(camp_dir, "snippets")
      for i, ev in camp.iterrows():
        loc_src = None
        payload = None
        snip_name = ev.get("snippet")
        if isinstance(snip_name, str) and os.path.exists(os.path.join(snip_dir, snip_name)):
            d = np.load(os.path.join(snip_dir, snip_name))
            loc_src = (d["time"], d["flux"])      # injected flux lives ONLY here
            payload = {"time": d["time"], "flux": d["flux"], "flux_err": d["flux_err"],
                       "event_time": float(d["event_time"]),
                       "event_width": float(ev.duration)}
        built = exp.build_record(ev, label_source, local_flux_source=loc_src)
        if built is None:
            continue
        rec, loc, rc = built
        # v2 = the ELC-calibrated pair anchoring from the re-injection; v1 = the
        # anchoring the 60k campaign ran with. Kept per row so the composition
        # is auditable in the manifest rather than inferred.
        rec["pair_model_version"] = source_version
        # Distinctive label: two different sources can share a basename.
        rec["source_dir"] = os.path.relpath(camp_dir, REPO)
        event_no = len(records)
        rec["event_key"] = comparator_key(int(ev.tic), int(ev.sector), event_no)
        records.append(rec); locals_.append(loc); recurs.append(rc); keys.append(rec["event_key"])
        if payload is not None:
            payload.update({"tic": int(ev.tic), "sector": int(ev.sector),
                            "event_no": event_no})
            payloads.append(payload)

    build_seconds = _time.time() - t_start
    log.info("Built %d records in %.1f s (%.4f s/event)",
             len(records), build_seconds, build_seconds / max(len(records), 1))
    if not records:
        raise SystemExit("No records built")

    # ---- the incumbent block, BEFORE writing -----------------------------
    per_event = None
    if args.no_comparator:
        log.warning("Comparator SKIPPED; the incumbent block is zero and invalid")
        inc = build_block(None, keys)
    else:
        rows, per_event = run_comparator(payloads, out_dir, workers=args.comparator_workers)
        inc = build_block(rows, keys)

    for rec in records:
        rec.update({f"valid_{k}": v for k, v in exp.scalar_valid(rec).items()})

    # ---- probe 1's pre-named remedy, as a FLAG not a deletion --------------
    # rebalance_negatives subsamples the over-represented inversion arm within
    # the negative class so its inversion rate matches the positives'. Applying
    # it by DELETING rows would throw away genuine negatives that other analyses
    # still want (funnel counts, per-stratum diagnostics). So every row is kept
    # and marked: arms and probes filter on rebalance_keep, and the datasheet
    # carries both the before and after numbers. Strictly more informative than
    # deletion, and the same remedy.
    rebalance_report = None
    tmp = pd.DataFrame(records)
    tmp["_row"] = np.arange(len(tmp))
    kept, rebalance_report = rebalance_negatives(tmp, seed=20260804)
    keep_rows = set(kept["_row"].tolist())
    for j, rec in enumerate(records):
        rec["rebalance_keep"] = int(j in keep_rows)
    log.info("Rebalance flag: keeping %d of %d rows; probe-1 gap %.6f -> %.6f",
             len(keep_rows), len(records), rebalance_report["probe1_gap_before"],
             rebalance_report["probe1_gap_after"])

    # ---- the frozen split, assigned by SYSTEM ---------------------------
    split_report = None
    if args.freeze_split:
        ev_df = pd.DataFrame(records)
        excl = []
        excl_path = os.path.join(REPO, "data", "exclusion_tics.txt")
        if os.path.exists(excl_path):
            for line in open(excl_path):
                if not line.startswith("#") and line.strip():
                    excl.append(int(line.split(",")[0]))
        tic_tab = splitmod.build_tic_table(ev_df, raw)
        assignment, _ = splitmod.assign(tic_tab, excluded_tics=excl)
        _, split_report = splitmod.report(ev_df, assignment, excluded_tics=excl)
        for rec in records:
            rec["split"] = assignment.get(int(rec["tic"]), "train")
        log.info("Frozen split verified. TIC-map sha256 %s",
                 split_report["assignment_sha256"][:16])
        for sname, st in split_report["by_split"].items():
            log.info("  %-5s %4d TICs  %7d events (%.1f%%)  %6d positive",
                     sname, st["n_tics"], st["n_events"],
                     100 * st["event_fraction"], st["n_positive"])

    shard = os.path.join(out_dir, f"bench_{args.tag}.h5")
    df = write_shard(shard, records, locals_, recurs, inc)

    total = _time.time() - t_start
    summary = {
        "tag": args.tag,
        "n_events": int(len(df)),
        "n_positive": int((df.label == 1).sum()),
        "n_negative": int((df.label == 0).sum()),
        "by_source": {schema.LABEL_SOURCES[k]: int(v)
                      for k, v in df.label_source.value_counts().items()},
        "distinct_tics": int(df.tic.nunique()),
        "incumbent_valid_fraction": float(df.incumbent_valid.mean()) if "incumbent_valid" in df else 0.0,
        "seconds_total": round(total, 1),
        "seconds_per_event_build": round(build_seconds / max(len(records), 1), 4),
        "seconds_per_event_comparator": round(per_event, 4) if per_event else None,
        "n_scalars": len(schema.all_scalar_names(INCUMBENT_COLS)),
        "local_shape": [schema.LOCAL_CHANNELS, schema.LOCAL_BINS],
        "recurrence_shape": [schema.RECUR_CHANNELS, schema.RECUR_BINS],
        "e2_passed": args.e2_passed,
        "exporter_stats": exp.stats,
        "rebalance": rebalance_report,
        "frozen_split": split_report,
        "sources": [{"dir": os.path.relpath(d, REPO), "label_source": lab}
                    for d, lab in sources],
    }
    # Projection to the full export, which decides whether it runs overnight.
    n_full = 3672 + 150000
    summary["projected_full_export_hours"] = round(
        n_full * (total / max(len(records), 1)) / 3600, 2)
    with open(os.path.join(out_dir, "export_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    for k, v in summary.items():
        log.info("  %-34s %s", k, v)


if __name__ == "__main__":
    main()
