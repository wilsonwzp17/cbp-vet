"""Detectability vetter-pass leg + the radius-axis curves (H1/H2/H5/H8).

What this produces
------------------
The second half of the detectability measurement for the deployment slide.
28_detectability.py injected 7,050 tests and recorded what the SEARCH
recovered; this script measures what the frozen VETTER keeps of those
recoveries, then draws the curves that put both fractions on one radius axis.

1. Exports every label==1 event (the recovered injected events; harvested
   real re-finds are label 0 and stay out) through the FROZEN exporter -
   the same unmodified ``cbpvet/export`` path that built bench-v1 - with
   ``label_source="bank_injection"`` per H2: the schema's enum is CLOSED, so
   the deploy_native provenance string lives on injector rows and run
   metadata only; a new enum string would crash the frozen code, by design.
2. Runs mono-cbp's own comparator to fill the incumbent block (workers > 1
   is fine here; this is not sealed data), writes a scoring shard under
   data/deploy_run/detectability/score/, and scores it with the
   hash-asserted frozen arm-b checkpoint through the single gate
   (arms.score_matrix). Per-event vetter_pass = score >= the BANKED
   OP-B stratum tau from t0_addendum's model_freeze block - the same
   operating point the deployment shortlist used, not a new threshold.
3. Joins each scored event back to its test on ``model_idx`` (unique per
   test in this run: the plan drew 7,050 model indices without replacement)
   and reduces to a per-TEST verdict: a recovered injection passes the
   vetter if ANY of its lead events (|event_time - inj_time| <
   inj_duration/2, the injector's own recovery rule) scores at or above
   tau. A test can flag one injection as several events; requiring only one
   to pass mirrors how a human queue works - the candidate survives if any
   of its detections does.
4. Builds the curves under data/deploy_run/detectability/curves/:
   (a) HEADLINE aggregate on program != 'secondary_all63' (disjoint-only
   programs, clean as measured); (b) per-system curves for the golden 27
   (150 tests each - small-n, said on the figure); (c) the secondary_all63
   variant, overlap-included, with the PRE-NAMED rebalance applied first:
   its inversion mix measured 0.527 (gate FAIL), so the over-represented
   inverted tests are subsampled to a 0.50 mix with a pinned seed before
   any rate is quoted - the probe-1 remedy's mechanism, applied at the
   test level, n dropped recorded. Curve y1 = search-recovered fraction
   (from tests_all), y2 = vetter-pass fraction OF RECOVERED, both with
   binomial (Wilson, z=1) error bars.

Traps encoded here
------------------
- AXIS NAME: ``rp_over_rstar_model`` is sqrt(inj_depth), a limb-darkening
  blind proxy for the bank's true stored ratio (deviations up to ~13%), and
  the column itself was rewritten once after an all-NaN column-name slip
  (campaign_summary.json 'corrections'). The axis is therefore labeled
  'sqrt(model depth)', never 'radius ratio', and both input CSVs are
  hash-asserted against interim_hashes_2026-08-07.json before anything runs.
- CHECKPOINT HASH ASSERT FIRST, exactly as 25_deploy_score.py: the model
  must hash to the t0_addendum record or the run refuses to start.
- The comparator keys rows by tic/sector/event_no it reads from the payload
  (13_export's measured trap): payloads carry all three, and the module is
  imported under its real file-backed name "13_export" so spawn workers can
  re-import it.
- set_system_events receives the FULL events table (labels 0 and 1) keyed
  by (tic, sector, model_idx): the observed-pair features must see the
  same flagged siblings the benchmark export path saw, not a label-1
  subset - one test is one injection, and its negatives are real context.
- H8's rider (one convention per surface): the figures show fractions with
  binomial intervals; raw counts live in the captions and the JSON, never
  mixed on one axis.
- Figure labels never carry a golden TIC: per-system panels are indexed
  g01..g27 (mapping kept in curves_summary.json, which stays local under
  data/ like every other deployment artifact).
- curves_summary.json is dumped with allow_nan=False: an undefined rate
  (empty bin) is null, never NaN, and the dump itself enforces it.

Usage
-----
    python experiments/30_detectability_score.py                  # full run
    python experiments/30_detectability_score.py --limit 60       # smoke
    python experiments/30_detectability_score.py --comparator-workers 8
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import logging
import math
import sys
import time as _time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet.export import EventExporter, build_block, schema, write_shard
from cbpvet.models import arms

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
CAT_PATH = os.path.expanduser("~/mono-cbp/catalogues/TEBC_morph_05_P_7.csv")
NOISE = os.path.join(REPO, "data", "noise_screen.csv")
ADDENDUM = os.path.join(REPO, "data", "bench", "full", "t0_addendum_2026-08-06.json")
INTERIM = os.path.join(REPO, "data", "bench", "full", "interim_hashes_2026-08-07.json")
DET = os.path.join(REPO, "data", "deploy_run", "detectability")
STAGED = os.path.join(REPO, "data", "deploy_staged", "lc")
SCORE_DIR = os.path.join(DET, "score")
CURVES_DIR = os.path.join(DET, "curves")

REBALANCE_SEED = 20260807          # pinned: the dropped-test set is part of the record
N_BINS_HEADLINE = 9
N_BINS_PER_SYSTEM = 6
WILSON_Z = 1.0                     # 1-sigma-equivalent binomial interval

BLUE, RED, GRAY = "#4878a8", "#c44e52", "#777777"   # make_figures palette

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("30_detect_score")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def comparator_key(tic, sector, event_no):
    return f"TIC_{tic}_{sector}_{event_no}"


def load_frozen_model(arm="b"):
    add = json.load(open(ADDENDUM))
    entry = add["model_freeze"]["models"][arm]
    path = os.path.join(REPO, entry["path"])
    got = sha256(path)
    if got != entry["sha256"]:
        raise RuntimeError(
            f"checkpoint hash mismatch for arm {arm}: {got} != recorded "
            f"{entry['sha256']}. REFUSING to score with an unverified model.")
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(path)
    log.info("frozen arm-%s checkpoint verified (%s...) and loaded",
             arm, entry["sha256"][:16])
    return model, entry


def assert_input_hashes():
    """The two CSVs this leg consumes are interim-hashed; refuse drift."""
    rec = json.load(open(INTERIM))["hashes"]
    for rel in ("data/deploy_run/detectability/tests_all.csv",
                "data/deploy_run/detectability/events_all.csv"):
        got = sha256(os.path.join(REPO, rel))
        if got != rec[rel]:
            raise RuntimeError(f"{rel} hash {got} != interim record {rec[rel]}; "
                               "inputs drifted, refusing to build curves on them")
    log.info("input hashes verified against interim_hashes_2026-08-07.json")


def wilson(k, n, z=WILSON_Z):
    """Binomial Wilson interval; (None, None) when undefined (n == 0)."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def curve_table(tests, edges):
    """Per-bin counts and rates for one curve.

    ``tests`` must carry rp_over_rstar_model, recovered, and vetter_pass
    (test-level, defined only where recovered == 1). Rates in empty cells are
    None so the JSON stays NaN-free by construction.
    """
    x = tests["rp_over_rstar_model"].to_numpy(dtype=float)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = tests[idx == b]
        n = int(len(sel))
        rec = int(sel["recovered"].sum())
        npass = int(sel.loc[sel["recovered"] == 1, "vetter_pass"].sum())
        rec_lo, rec_hi = wilson(rec, n)
        p_lo, p_hi = wilson(npass, rec)
        rows.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
            "n_tests": n, "n_recovered": rec, "n_vetter_pass": npass,
            "recovered_frac": (rec / n) if n else None,
            "recovered_ci68": [rec_lo, rec_hi] if n else None,
            "vetter_pass_frac_of_recovered": (npass / rec) if rec else None,
            "vetter_pass_ci68": [p_lo, p_hi] if rec else None,
        })
    return rows


def plot_curve(ax, table, small_n_note=False):
    """One curve on one axis: y1 recovered, y2 vetter-pass of recovered."""
    for series, color, marker in (
            ("recovered", BLUE, "o"), ("vetter", RED, "s")):
        xs, ys, lo, hi = [], [], [], []
        for r in table:
            frac = (r["recovered_frac"] if series == "recovered"
                    else r["vetter_pass_frac_of_recovered"])
            ci = (r["recovered_ci68"] if series == "recovered"
                  else r["vetter_pass_ci68"])
            if frac is None:
                continue
            xs.append(r["bin_center"]); ys.append(frac)
            # Wilson endpoints bracket the point estimate mathematically, but
            # at p in {0, 1} they touch it and float error can leave a -1e-17
            # difference, which errorbar rejects; clamp at zero.
            lo.append(max(0.0, frac - ci[0])); hi.append(max(0.0, ci[1] - frac))
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marker + "-", color=color,
                        capsize=3, lw=1.4, ms=4 if small_n_note else 5)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, lw=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparator-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: export only the first N label-1 events")
    args = ap.parse_args()
    os.makedirs(SCORE_DIR, exist_ok=True)
    os.makedirs(CURVES_DIR, exist_ok=True)

    assert_input_hashes()
    model, entry = load_frozen_model("b")      # hash-assert BEFORE any work
    tau = float(entry["taus_banked"]["OPB_stratum"])
    log.info("banked OP-B stratum tau = %.10f", tau)

    events = pd.read_csv(os.path.join(DET, "events_all.csv"))
    tests = pd.read_csv(os.path.join(DET, "tests_all.csv"))
    pos = events[events.label == 1].reset_index(drop=True)
    n_label1 = len(pos)
    if args.limit:
        pos = pos.head(args.limit)
    log.info("events %d (label-1 %d, exporting %d); tests %d (recovered %d)",
             len(events), n_label1, len(pos), len(tests),
             int(tests.recovered.sum()))

    # ---- export through the FROZEN path, exactly as 13/25 do ---------------
    cat = load_catalogue(CAT_PATH, TEBC=True)
    raw = pd.read_csv(CAT_PATH)
    noise = pd.read_csv(NOISE)
    exp = EventExporter(STAGED, cat, raw, noise_screen=noise, e2_passed=True)
    # FULL events table: the pair features must see every flagged sibling of
    # the same test, including the label-0 ones. One test = one injection.
    exp.set_system_events(events, key=("tic", "sector", "model_idx"))

    snip_dir = os.path.join(DET, "snippets")
    records, locals_, recurs, keys, payloads, meta = [], [], [], [], [], []
    skipped = 0
    t0 = _time.time()
    for _, ev in pos.iterrows():
        snip = os.path.join(snip_dir, str(ev["snippet"]))
        if not os.path.exists(snip):
            skipped += 1
            continue
        d = np.load(snip)
        loc_src = (d["time"], d["flux"])       # injected flux lives ONLY here
        built = exp.build_record(ev, "bank_injection", local_flux_source=loc_src)
        if built is None:
            skipped += 1
            continue
        rec, loc, rc = built
        # Audit-only bookkeeping (schema.AUDIT_COLUMNS; never features).
        # pair_model_version 3 = the empirical Kostov-resample pair model this
        # run pinned (28_detectability docstring + campaign_summary pins).
        rec["pair_model_version"] = 3
        rec["source_dir"] = "deploy_run/detectability"
        rec["rebalance_keep"] = 1              # scoring shard: every row scored
        rec["split"] = "deploy"
        event_no = len(records)
        rec["event_key"] = comparator_key(int(ev.tic), int(ev.sector), event_no)
        records.append(rec); locals_.append(loc); recurs.append(rc)
        keys.append(rec["event_key"])
        payloads.append({"time": d["time"], "flux": d["flux"],
                         "flux_err": d["flux_err"],
                         "event_time": float(d["event_time"]),
                         "event_width": float(ev.duration),
                         "tic": int(ev.tic), "sector": int(ev.sector),
                         "event_no": event_no})
        meta.append({"model_idx": int(ev.model_idx), "tic": int(ev.tic),
                     "sector": int(ev.sector),
                     "event_time": float(ev.event_time),
                     "inj_time": float(ev.inj_time),
                     "inj_duration": float(ev.inj_duration),
                     "snippet": str(ev["snippet"])})
    if not records:
        raise SystemExit("no records built - nothing to score")
    log.info("built %d records (%d skipped) in %.1f s",
             len(records), skipped, _time.time() - t0)
    assert len(records) + skipped == len(pos), (len(records), skipped, len(pos))

    # ---- incumbent block via mono-cbp's comparator (13_export's own path) --
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import importlib
    exp13 = importlib.import_module("13_export")
    rows, per_event = exp13.run_comparator(payloads, SCORE_DIR,
                                           workers=args.comparator_workers)
    inc = build_block(rows, keys)

    for rec in records:
        rec.update({f"valid_{k}": v for k, v in exp.scalar_valid(rec).items()})

    shard = os.path.join(SCORE_DIR, "detectability_shard.h5")
    df = write_shard(shard, records, locals_, recurs, inc)
    assert len(df) == len(records)

    # ---- score through the single gate -------------------------------------
    X, feats = arms.score_matrix(df, "b")
    assert len(feats) == entry["n_features"], (len(feats), entry["n_features"])
    scores = model.predict_proba(X)[:, 1]
    assert np.all((scores >= 0.0) & (scores <= 1.0)), "scores outside [0,1]"
    scored = pd.DataFrame(meta)
    scored["arm_b_score"] = scores
    scored["vetter_pass"] = (scores >= tau).astype(int)
    # lead = the injector's own recovery rule, per event
    scored["is_lead"] = (np.abs(scored.event_time - scored.inj_time)
                         < scored.inj_duration / 2.0).astype(int)

    # ---- join back to tests on model_idx (unique per test, this run) -------
    assert tests.model_idx.is_unique, "model_idx not unique in tests_all"
    tcols = tests[["model_idx", "program", "rp_over_rstar_model",
                   "inverted_lc", "recovered"]]
    scored = scored.merge(tcols, on="model_idx", how="left",
                          validate="many_to_one")
    join_cov = float(scored.program.notna().mean())
    assert join_cov == 1.0, f"join coverage {join_cov} != 1.0"
    scored.to_csv(os.path.join(SCORE_DIR, "scored_events.csv"), index=False)

    # ---- per-test verdict: any lead event at/above tau ----------------------
    lead = scored[scored.is_lead == 1]
    per_test = lead.groupby("model_idx").agg(
        n_lead_events=("arm_b_score", "size"),
        max_lead_score=("arm_b_score", "max")).reset_index()
    per_test["vetter_pass"] = (per_test.max_lead_score >= tau).astype(int)
    per_test.to_csv(os.path.join(SCORE_DIR, "test_vetter_pass.csv"), index=False)

    tests = tests.merge(per_test[["model_idx", "vetter_pass"]],
                        on="model_idx", how="left")
    tests["vetter_pass"] = tests["vetter_pass"].fillna(0).astype(int)
    n_rec = int(tests.recovered.sum())
    n_rec_scored = int(tests.model_idx.isin(per_test.model_idx).sum())
    if args.limit is None and n_rec_scored != n_rec:
        # a recovered test whose lead events all failed to export would sit in
        # the denominator without a score; that must be a disclosed count
        log.warning("recovered tests with a scored lead event: %d of %d",
                    n_rec_scored, n_rec)
    n_pass_total = int(tests.loc[tests.recovered == 1, "vetter_pass"].sum())
    log.info("tests: %d recovered, %d with scored lead, %d vetter-pass "
             "(%.3f of recovered)", n_rec, n_rec_scored, n_pass_total,
             n_pass_total / max(n_rec, 1))

    # ---- curves -------------------------------------------------------------
    rp = tests.rp_over_rstar_model.to_numpy(dtype=float)
    assert np.isfinite(rp).all(), "NaN in rp_over_rstar_model"
    lo_x, hi_x = float(rp.min()), float(rp.max())
    edges9 = np.linspace(lo_x, hi_x, N_BINS_HEADLINE + 1)
    edges6 = np.linspace(lo_x, hi_x, N_BINS_PER_SYSTEM + 1)

    headline_tests = tests[tests.program != "secondary_all63"]
    headline = curve_table(headline_tests, edges9)
    assert sum(r["n_tests"] for r in headline) == len(headline_tests)
    assert sum(r["n_recovered"] for r in headline) == \
        int(headline_tests.recovered.sum())

    # golden 27, indexed g01..g27 by ascending TIC; TICs stay out of figures
    golden_progs = sorted(
        (p for p in tests.program.unique() if p.startswith("per_system_")),
        key=lambda p: int(p.split("_")[-1]))
    g_index = {p: f"g{i + 1:02d}" for i, p in enumerate(golden_progs)}
    per_system = {}
    for p in golden_progs:
        sub = tests[tests.program == p]
        per_system[g_index[p]] = {
            "program": p,                      # local file; TICs allowed here
            "n_tests": int(len(sub)),
            "n_distinct_files": int(sub[["tic", "sector"]]
                                    .drop_duplicates().shape[0]),
            "bins": curve_table(sub, edges6),
        }

    # secondary_all63, overlap-included, PRE-NAMED rebalance applied first:
    # subsample the over-represented inverted tests to a 0.50 mix, pinned seed
    sec = tests[tests.program == "secondary_all63"]
    inv = sec[sec.inverted_lc == 1]
    nat = sec[sec.inverted_lc == 0]
    rng = np.random.default_rng(REBALANCE_SEED)
    if len(inv) > len(nat):
        keep_idx = rng.choice(inv.index.to_numpy(), size=len(nat),
                              replace=False)
        sec_bal = pd.concat([nat, inv.loc[np.sort(keep_idx)]])
        n_dropped = int(len(inv) - len(nat))
    else:                                       # symmetric, for safety
        keep_idx = rng.choice(nat.index.to_numpy(), size=len(inv),
                              replace=False)
        sec_bal = pd.concat([inv, nat.loc[np.sort(keep_idx)]])
        n_dropped = int(len(nat) - len(inv))
    mix_after = float(sec_bal.inverted_lc.mean())
    assert abs(mix_after - 0.5) < 1e-12, mix_after
    secondary = curve_table(sec_bal, edges9)

    # ---- figures (make_figures style: font 12, honest axes, counts in
    # captions; H8: fractions on axes, counts in text) ------------------------
    plt.rcParams.update({"font.size": 12})
    x_label = "sqrt(model depth)"              # the recorded axis caveat

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plot_curve(ax, headline)
    ax.set(xlabel=x_label, ylabel="fraction of injections",
           title="Detectability, headline (disjoint-only programs)")
    ax.legend(handles=[
        plt.Line2D([], [], color=BLUE, marker="o", label="search recovered"),
        plt.Line2D([], [], color=RED, marker="s",
                   label="vetter pass, of recovered")],
        loc="lower right", fontsize=10)
    n_h, r_h = len(headline_tests), int(headline_tests.recovered.sum())
    p_h = int(headline_tests.loc[headline_tests.recovered == 1,
                                 "vetter_pass"].sum())
    fig.text(0.5, -0.04,
             f"{n_h:,} injections (per-system + aggregate programs, disjoint "
             f"files only); {r_h:,} recovered by the search; {p_h:,} pass the "
             f"frozen vetter at the banked OP-B stratum tau. Error bars: "
             f"binomial 68% (Wilson). x = sqrt(injected model depth), a "
             f"limb-darkening-blind proxy for r_p/R_*.",
             ha="center", fontsize=9, wrap=True)
    fig.tight_layout()
    headline_png = os.path.join(CURVES_DIR, "headline_curve.png")
    fig.savefig(headline_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    ncols, nrows = 5, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15.5),
                             sharex=True, sharey=True)
    for k, p in enumerate(golden_progs):
        ax = axes[k // ncols][k % ncols]
        entry_g = per_system[g_index[p]]
        plot_curve(ax, entry_g["bins"], small_n_note=True)
        ax.set_title(f"{g_index[p]}  (n={entry_g['n_tests']}, "
                     f"files={entry_g['n_distinct_files']})", fontsize=10)
        ax.tick_params(labelsize=8)
    for k in range(len(golden_progs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
        # sharex hides tick labels everywhere but the grid's bottom row; with
        # this slot dark, the lowest VISIBLE panel of the column needs them back
        axes[k // ncols - 1][k % ncols].tick_params(labelbottom=True)
    fig.suptitle("Per-system detectability, golden 27 (g-indexed; "
                 "150 injections each - small-n bins)", fontsize=13)
    fig.text(0.5, 0.055,
             f"x = {x_label}; blue = search recovered fraction, red = vetter "
             f"pass fraction of recovered; binomial 68% (Wilson) bars. "
             f"{N_BINS_PER_SYSTEM} bins over the common range "
             f"[{lo_x:.3f}, {hi_x:.3f}]: ~25 tests per bin, so single-bin "
             f"wiggles are noise. Systems with files=1 ran all 150 injections "
             f"into one disjoint file (disclosed per H1). Masking regime per "
             f"H5: disclosed in campaign_summary.json, identical to the "
             f"benchmark campaign's.",
             ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    per_system_png = os.path.join(CURVES_DIR, "per_system_golden27.png")
    fig.savefig(per_system_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plot_curve(ax, secondary)
    ax.set(xlabel=x_label, ylabel="fraction of injections",
           title="SECONDARY, overlap-included (all-63 files; "
                 "inversion-rebalanced)")
    ax.legend(handles=[
        plt.Line2D([], [], color=BLUE, marker="o", label="search recovered"),
        plt.Line2D([], [], color=RED, marker="s",
                   label="vetter pass, of recovered")],
        loc="lower right", fontsize=10)
    r_s = int(sec_bal.recovered.sum())
    p_s = int(sec_bal.loc[sec_bal.recovered == 1, "vetter_pass"].sum())
    fig.text(0.5, -0.04,
             f"Secondary program: includes frozen-overlap files, NOT the "
             f"headline. Inversion mix measured 0.527 (gate FAIL): "
             f"{n_dropped} over-represented inverted tests subsampled out "
             f"(seed {REBALANCE_SEED}) to a 0.500 mix before quoting rates. "
             f"{len(sec_bal):,} of {len(sec):,} tests kept; {r_s:,} "
             f"recovered; {p_s:,} vetter-pass. Binomial 68% (Wilson) bars.",
             ha="center", fontsize=9, wrap=True)
    fig.tight_layout()
    secondary_png = os.path.join(CURVES_DIR, "secondary_rebalanced_curve.png")
    fig.savefig(secondary_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- summaries, hash-recorded ------------------------------------------
    score_summary = {
        "built": "2026-08-08",
        "checkpoint_sha256": entry["sha256"],
        "tau_used": tau,
        "tau_source": "t0_addendum_2026-08-06.json model_freeze b "
                      "taus_banked.OPB_stratum",
        "label_source": "bank_injection (H2: closed enum; deploy_native is "
                        "injector-row provenance only)",
        "n_events_label1": int(n_label1),
        "n_exported": int(len(records)),
        "n_skipped": int(skipped),
        "n_lead_events": int(len(lead)),
        "n_tests_recovered": n_rec,
        "n_recovered_with_scored_lead": n_rec_scored,
        "n_tests_vetter_pass": n_pass_total,
        "comparator_workers": args.comparator_workers,
        "comparator_s_per_event": per_event,
        "incumbent_valid_fraction": float(df.incumbent_valid.mean())
            if "incumbent_valid" in df else 0.0,
        "score_quantiles": {q: float(np.quantile(scores, float(q)))
                            for q in ("0.1", "0.5", "0.9")},
        "shard_sha256": sha256(shard),
        "limit": args.limit,
    }
    with open(os.path.join(SCORE_DIR, "score_summary.json"), "w") as fh:
        json.dump(score_summary, fh, indent=2, allow_nan=False)

    outputs = {
        "score/detectability_shard.h5": shard,
        "score/scored_events.csv": os.path.join(SCORE_DIR, "scored_events.csv"),
        "score/test_vetter_pass.csv": os.path.join(SCORE_DIR,
                                                   "test_vetter_pass.csv"),
        "score/score_summary.json": os.path.join(SCORE_DIR,
                                                 "score_summary.json"),
        "curves/headline_curve.png": headline_png,
        "curves/per_system_golden27.png": per_system_png,
        "curves/secondary_rebalanced_curve.png": secondary_png,
    }
    curves_summary = {
        "built": "2026-08-08",
        "tau_used": tau,
        "checkpoint_sha256": entry["sha256"],
        "axis": {"column": "rp_over_rstar_model",
                 "label": x_label,
                 "caveat": "sqrt(inj_depth): limb-darkening-blind proxy for "
                           "the bank's true stored ratio, deviations up to "
                           "~13% (campaign_summary.json corrections)",
                 "range": [lo_x, hi_x]},
        "error_bars": f"binomial Wilson interval, z={WILSON_Z} (~68%)",
        "test_pass_rule": "recovered test passes if ANY lead event "
                          "(|event_time - inj_time| < inj_duration/2) scores "
                          ">= tau",
        "headline": {
            "programs": "all except secondary_all63 (disjoint-only)",
            "n_tests": n_h, "n_recovered": r_h, "n_vetter_pass": p_h,
            "bin_edges": [float(e) for e in edges9],
            "bins": headline,
        },
        "per_system_golden27": {
            "note": "150 tests/system, 6 bins over the common range: "
                    "small-n, ~25 tests per bin",
            "bin_edges": [float(e) for e in edges6],
            "g_index_to_program_LOCAL_ONLY": {g_index[p]: p
                                              for p in golden_progs},
            "systems": per_system,
        },
        "secondary_all63_rebalanced": {
            "note": "overlap-included, NOT headline; pre-named rebalance "
                    "applied before any rate quoted",
            "mix_before": float(sec.inverted_lc.mean()),
            "mix_after": mix_after,
            "n_tests_before": int(len(sec)),
            "n_tests_after": int(len(sec_bal)),
            "n_dropped_inverted": n_dropped,
            "rebalance_seed": REBALANCE_SEED,
            "n_recovered": r_s, "n_vetter_pass": p_s,
            "bin_edges": [float(e) for e in edges9],
            "bins": secondary,
        },
        "verification": {
            "input_hashes_asserted": True,
            "n_events_label1": int(n_label1),
            "n_exported": int(len(records)),
            "n_skipped": int(skipped),
            "shard_rows_equal_exported": bool(len(df) == len(records)),
            "scores_in_unit_interval": True,
            "join_coverage": join_cov,
            "headline_recovered_matches_tests_all": bool(
                sum(r["n_recovered"] for r in headline) ==
                int(headline_tests.recovered.sum())),
            "recovered_tests_with_scored_lead": n_rec_scored,
        },
        "output_sha256": {k: sha256(v) for k, v in outputs.items()},
    }
    curves_path = os.path.join(CURVES_DIR, "curves_summary.json")
    with open(curves_path, "w") as fh:
        json.dump(curves_summary, fh, indent=2, allow_nan=False)
    log.info("curves_summary.json written (%s...)", sha256(curves_path)[:16])
    log.info("DONE: headline %d/%d recovered, %d vetter-pass; secondary "
             "rebalanced %d tests (%d dropped)", r_h, n_h, p_h,
             len(sec_bal), n_dropped)


if __name__ == "__main__":
    main()
