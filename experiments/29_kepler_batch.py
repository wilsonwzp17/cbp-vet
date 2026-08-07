"""Kepler STRETCH extraction: Kepler-35 + Kepler-1647 into a fresh data/kepler_batch/.

What this script produces
-------------------------
data/kepler_batch/
    lc/                        staged quarter files TIC_<KIC>_<QQ>.txt (SAP,
                               crowding-corrected, normalized), eclipse-masked
                               IN PLACE by EclipseMasker (these are our own
                               staged copies; nothing outside lc/ is rewritten)
    out/detected_events.txt    the search's flagged events - written by the
                               finder, NOT read row-by-row by this script
    out/event_snippets/        per-event snippets for the Aug-11 one-shot
    kepler_catalog.csv         the catalogue rows the masker and finder consumed
    host_kic.json              host name -> KIC as resolved by MAST download
    b7_resolution.json         B7 second half: TESS TICs of all 10 Kepler hosts
                               + the exclusion-file merge + intersection checks
    kepler_batch_report.json   event COUNTS, QA verdicts, runtimes, sources

Why this exists (H4, Freeze-Decision-Registry_2026-08-06 section H)
-------------------------------------------------------------------
H4 adopts the times-complete REDUCED Kepler tier: floor = Kepler-16 + Kepler-34
(already extracted by 19_kepler_smoke.py, smoke PASS 5/5), capped stretch =
Kepler-35 + Kepler-1647, extracted tonight into a FRESH data/kepler_batch/.
data/kepler_smoke/ is cited by the frozen manifest and is NEVER rewritten;
this script asserts it touches nothing under that path. Whatever is extracted
and QA'd by the pre-shot freeze is the declared tier (H4's hard cap); a host
that fails here is a recorded reduction, not a hidden one.

SEALED DISCIPLINE (H4 rider 1, ABSOLUTE)
----------------------------------------
The flagged events are REAL Kepler-planet candidates and the Aug-11 one-shot is
the single permitted look. This script therefore records ONLY: event counts per
host/quarter, the no-TCE-inside-eclipse-mask QA check, snippet-file existence
and loadability, and runtimes. It does NOT match events to known transit times
(the smoke's criterion iv is deliberately ABSENT), does not read or print
individual event times or properties, and does not run the exporter or
comparator on any of it. data/kepler_transit_times.csv is not even loaded.

Where every number comes from (no memory allowed)
-------------------------------------------------
- Binary periods: VERIFIED-FACTS_2026-07-29.md section 7 (Kostov 2020b Table 1
  transcription), re-parsed at runtime from CBPVET_DOCS_DIR, which is REQUIRED
  here (the smoke merely warned; tonight's brief pins the file's exact values).
  Kepler-35 b 20.734 d, Kepler-1647 b 11.259 d.
- bjd0, eclipse positions, widths, depths: the Villanova Kepler EB catalog
  (http://keplerebs.villanova.edu/overview/?k=9837578 and ?k=5473556,
  retrieved 2026-08-07), transcribed into VILLANOVA below. The catalog's bjd0
  is BJD - 2,400,000; BKJD = BJD - 2,454,833; hence
      bjd0_bkjd = bjd0_villanova - 54,833.0.
  Both converted epochs (132.85, 123.48 BKJD) land at the start of Kepler
  science data, as a primary-eclipse epoch must.
- The catalog values are not trusted blind: the downloaded data are folded on
  (P_verified, bjd0_bkjd) and the deepest bin MUST sit at phase ~1.0 (the
  finder's verified wraparound branch) or the host aborts. Eclipse widths are
  additionally MEASURED from that fold at 10 percent of peak depth (the
  experiment-17 method that built the TIC-172900988 catalogue) and the mask
  uses max(catalog width, measured width) per eclipse - never narrower than
  either source; both values and which one won are recorded in the report.

Traps honoured (from 19_kepler_smoke.py, the proven template)
-------------------------------------------------------------
- The catalogue row is MANDATORY: Kepler's 29.4-min cadence is under the 30-min
  threshold, so the finder calls bin_to_long_cadence and recomputes phase and
  eclipse mask from the row.
- sector_times=None: Kepler quarters would misindex the Skye metric.
- Filenames are built by THIS writer (Kepler metadata has QUARTER not SECTOR);
  the KIC integer takes the TIC field's place, consistently in files and rows.
- bjd0 is a primary-eclipse mid-time in BKJD so prim_pos = 1.0 exactly (the
  wraparound branch is the verified one).
- Files are staged in our own lc/; the masker's in-place rewrite touches only
  these staged copies.
- Per-quarter runtime logged.

Deviations from the smoke recipe, each with its reason
------------------------------------------------------
1. Ephemerides (bjd0, sec_pos, widths) come from the Villanova catalog instead
   of being self-measured - tonight's brief directs the catalog/paper route;
   the smoke's self-calibrating fold is retained as the sanity ASSERT and as
   the width floor (max rule above).
2. Criterion iv (known-transit matching) is OMITTED - sealed discipline,
   H4 rider 1. The QA here is (i) ran, (ii) no TCE inside mask,
   (iii) snippets loadable, (v) runtime.
3. The VERIFIED-FACTS period cross-check is REQUIRED, not best-effort.
4. Staged files live in lc/ (the brief's mandated layout), not the batch root.
5. The downloaded KEPLERID is asserted equal to the Villanova KIC - the smoke
   had no independent KIC pin; tonight we do, so name-resolution errors abort.

B7 second half (H4 rider 2), completed in this same batch
---------------------------------------------------------
Resolve the TESS TIC of all 10 Kepler CBP host systems via astroquery MAST
(nearest TIC row to the resolved name; the row's own KIC field is cross-checked
against the KICs this project measured from downloaded data where available),
then MERGE them into data/exclusion_tics.txt (the file is RE-READ immediately
before writing and never blind-overwritten - concurrent sessions write it),
then check intersections against the 591-TIC TEBC catalogue and the
deployment list data/goodsn_in_tebc.txt (expected EMPTY - a non-empty
intersection is reported loudly, never fixed silently).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime
import json
import logging
import re
import sys
import time as _time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.eclipse_masking import EclipseMasker
from mono_cbp.utils import get_eclipse_mask, time_to_phase

from cbpvet.search import TransitFinderExt

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
DATA = os.path.join(REPO, "data")
BATCH = os.path.join(DATA, "kepler_batch")
LC = os.path.join(BATCH, "lc")
OUT = os.path.join(BATCH, "out")

# Frozen-manifest artifacts this script must never touch.
FORBIDDEN = ("kepler_smoke", "lc_cache", os.path.join("bench", "full"),
             os.path.join("m1lite", "sealed"))
for _p in (BATCH, LC, OUT):
    for _f in FORBIDDEN:
        assert _f not in _p, f"write path {_p} intersects protected {_f}"

# Binary periods: pinned from VERIFIED-FACTS section 7 and re-verified at
# runtime against the file itself (REQUIRED; see verify_periods).
HOSTS = {
    "Kepler-35": {"p_bin": 20.734},
    "Kepler-1647": {"p_bin": 11.259},
}

# Villanova Kepler EB catalog rows, transcribed 2026-08-07 from
# http://keplerebs.villanova.edu/overview/?k=<kic>. bjd0 is BJD-2,400,000.
VILLANOVA = {
    "Kepler-35": {
        "kic": 9837578, "period": 20.7337490, "bjd0_bjd_m2400000": 54965.845830,
        "pwidth": 0.0115, "swidth": 0.0133, "pdepth": 0.3582, "sdepth": 0.2482,
        "sep": 0.5057, "morph": 0.08,
        "source": "http://keplerebs.villanova.edu/overview/?k=9837578 (retrieved 2026-08-07)",
    },
    "Kepler-1647": {
        "kic": 5473556, "period": 11.2588244, "bjd0_bjd_m2400000": 54956.478952,
        "pwidth": 0.0291, "swidth": 0.0227, "pdepth": 0.1789, "sdepth": 0.1606,
        "sep": 0.5526, "morph": 0.21,
        "source": "http://keplerebs.villanova.edu/overview/?k=5473556 (retrieved 2026-08-07)",
    },
}
BKJD_OFFSET = 54833.0  # BKJD = BJD - 2,454,833 = (BJD - 2,400,000) - 54,833

# B7: the 10 Kepler CBP host systems (Execution-Readiness B7).
B7_HOSTS = ["Kepler-16", "Kepler-34", "Kepler-35", "Kepler-38", "Kepler-47",
            "Kepler-64", "Kepler-413", "Kepler-453", "Kepler-1647", "Kepler-1661"]

TEBC_CSV = os.path.expanduser("~/mono-cbp/catalogues/TEBC_morph_05_P_7.csv")
EXCLUSION = os.path.join(DATA, "exclusion_tics.txt")
GOODSN_IN_TEBC = os.path.join(DATA, "goodsn_in_tebc.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("29_kepler_batch")


def verify_periods():
    """Re-parse VERIFIED-FACTS section 7 at runtime; REQUIRED for this run."""
    docs = os.environ.get("CBPVET_DOCS_DIR")
    if not docs:
        raise SystemExit("CBPVET_DOCS_DIR is required: tonight's brief pins the "
                         "periods to VERIFIED-FACTS section 7's exact values")
    vf_path = os.path.join(docs, "VERIFIED-FACTS_2026-07-29.md")
    if not os.path.exists(vf_path):
        raise SystemExit(f"VERIFIED-FACTS not found at {vf_path}")
    text = open(vf_path).read()
    for name, spec in HOSTS.items():
        m = re.search(re.escape(name) + r" b \| ([0-9.]+) \|", text)
        if not m:
            raise SystemExit(f"{name}: P_bin not found in VERIFIED-FACTS section 7")
        p_vf = float(m.group(1))
        if abs(p_vf - spec["p_bin"]) > 0.01:
            raise SystemExit(f"{name}: period mismatch, VF {p_vf} vs script {spec['p_bin']}")
        spec["p_bin"] = p_vf
        # Cross-check the Villanova period agrees with the verified one.
        p_cat = VILLANOVA[name]["period"]
        if abs(p_cat - p_vf) > 0.01:
            raise SystemExit(f"{name}: Villanova period {p_cat} disagrees with "
                             f"VERIFIED-FACTS {p_vf} by more than 0.01 d")
        log.info("%s: period %.5f d verified (VF) | Villanova %.7f d agrees "
                 "(delta %.5f d)", name, p_vf, p_cat, abs(p_cat - p_vf))


def download(name, spec):
    """All long-cadence quarters, SAP with crowding correction, by pinned KIC.

    Copied from 19_kepler_smoke.py (the proven path) with one CORRECTION found
    at run time: the smoke queried MAST by NAME and took whatever KEPLERID came
    back. For Kepler-35 the name-resolved cone search serves a NEIGHBOR star
    first (KEPLERID 9837586, not 9837578), so a by-name query would have
    extracted the wrong star. The query therefore targets "KIC <pin>" (the KIC
    is the Villanova catalog's, itself confirmed by the fold assert downstream)
    and every downloaded quarter's KEPLERID is still asserted against the pin.
    """
    import lightkurve as lk
    kic_pin = VILLANOVA[name]["kic"]
    res = lk.search_lightcurve(f"KIC {kic_pin}", mission="Kepler", cadence="long")
    log.info("%s (KIC %d): MAST offers %d long-cadence products", name, kic_pin, len(res))
    files, kic = [], None
    for row in res:
        try:
            lc = row.download(flux_column="sap_flux", quality_bitmask="hard")
        except Exception as exc:
            log.warning("  a quarter failed to download: %s", type(exc).__name__)
            continue
        if lc is None:
            continue
        kic = int(lc.meta.get("KEPLERID", 0))
        if kic != kic_pin:
            log.warning("  skipping a product whose KEPLERID %d != pinned KIC %d "
                        "(cone-search neighbor)", kic, kic_pin)
            continue
        quarter = int(lc.meta.get("QUARTER", -1))
        t = np.asarray(lc.time.value, float)          # BKJD
        f = np.asarray(lc.flux.value, float)
        fe = (np.asarray(lc.flux_err.value, float)
              if lc.flux_err is not None else np.full_like(f, np.nan))
        crowd, flfr = lc.meta.get("CROWDSAP"), lc.meta.get("FLFRCSAP")
        if crowd is not None and flfr is not None:
            med = np.nanmedian(f)
            f = (f - med * (1 - crowd)) / flfr
            fe = fe / flfr
        med = np.nanmedian(f)
        f, fe = f / med, fe / med
        keep = ~np.isnan(t * f)
        if keep.sum() < 100:
            continue
        files.append({"kic": kic, "quarter": quarter,
                      "time": t[keep], "flux": f[keep], "flux_err": fe[keep]})
    log.info("%s: downloaded %d usable quarters (KIC %d)", name, len(files), kic_pin)
    return kic_pin, files


def fold_qa_and_widths(name, files, p_bin, vill):
    """Fold on (P_verified, catalog bjd0) and (a) ASSERT the primary sits at
    phase ~1.0 (the smoke's own check, on the finder's verified wraparound
    branch), (b) measure both eclipse widths at 10 percent of peak depth
    (experiment-17 method, wraparound walk), (c) record where the secondary
    actually falls vs the catalog sep.
    """
    bjd0 = vill["bjd0_bjd_m2400000"] - BKJD_OFFSET   # BKJD
    ph = np.concatenate([time_to_phase(f["time"], p_bin, bjd0) for f in files])
    fl = np.concatenate([f["flux"] for f in files])
    nb = 800
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nb - 1)
    prof = np.array([np.median(fl[idx == b]) if (idx == b).any() else np.nan
                     for b in range(nb)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    depth = np.nanmedian(prof) - prof

    i1 = int(np.nanargmax(depth))
    d_wrap = np.abs(((centres - centres[i1] + 0.5) % 1.0) - 0.5)
    i2 = int(np.nanargmax(np.where(d_wrap > 0.1, depth, -np.inf)))

    def width_at(i, frac=0.10):
        thr = depth[i] * frac
        nl = nr = 0
        while nl < nb // 2 and np.isfinite(depth[(i - nl - 1) % nb]) and depth[(i - nl - 1) % nb] > thr:
            nl += 1
        while nr < nb // 2 and np.isfinite(depth[(i + nr + 1) % nb]) and depth[(i + nr + 1) % nb] > thr:
            nr += 1
        return (nl + nr + 1) / nb

    # (a) primary must fold to ~1.0 (equivalently ~0.0).
    prim_dist = float(min(centres[i1], 1.0 - centres[i1]))
    tol = max(vill["pwidth"] / 2.0, 0.005)
    log.info("%s: fold QA - deepest bin at phase %.4f (wrap distance %.4f, "
             "tolerance %.4f)", name, centres[i1], prim_dist, tol)
    if prim_dist > tol:
        raise SystemExit(f"{name}: FOLD QA FAILED - primary at phase "
                         f"{centres[i1]:.4f}, not ~1.0; catalog ephemeris "
                         f"rejected, host aborts")

    # (b) measured widths; the mask uses max(catalog, measured).
    w_prim_meas = float(width_at(i1))
    w_sec_meas = float(width_at(i2))
    # (c) secondary position as the data show it, relative to the primary.
    sec_pos_meas = float((centres[i2] - centres[i1]) % 1.0)

    ecl = {
        "bjd0": bjd0,
        "prim_pos": 1.0,
        "prim_width": max(vill["pwidth"], w_prim_meas),
        "sec_pos": vill["sep"],
        "sec_width": max(vill["swidth"], w_sec_meas),
    }
    qa = {
        "bjd0_bkjd": bjd0,
        "bjd0_conversion": "BKJD = (Villanova BJD-2400000 value) - 54833.0",
        "prim_fold_phase": float(centres[i1]),
        "prim_fold_wrap_distance": prim_dist,
        "prim_fold_tolerance": tol,
        "prim_fold_pass": True,
        "prim_depth_measured": float(depth[i1]),
        "sec_depth_measured": float(depth[i2]),
        "sec_pos_catalog": vill["sep"],
        "sec_pos_measured": sec_pos_meas,
        "sec_pos_agreement": abs(sec_pos_meas - vill["sep"]) < 0.02,
        "widths": {
            "prim": {"villanova": vill["pwidth"], "measured_10pct": w_prim_meas,
                     "used": ecl["prim_width"],
                     "source_used": "villanova" if vill["pwidth"] >= w_prim_meas else "measured"},
            "sec": {"villanova": vill["swidth"], "measured_10pct": w_sec_meas,
                    "used": ecl["sec_width"],
                    "source_used": "villanova" if vill["swidth"] >= w_sec_meas else "measured"},
        },
    }
    log.info("%s: widths prim %.4f (vill %.4f / meas %.4f), sec %.4f "
             "(vill %.4f / meas %.4f); sec_pos catalog %.4f vs measured %.4f",
             name, ecl["prim_width"], vill["pwidth"], w_prim_meas,
             ecl["sec_width"], vill["swidth"], w_sec_meas,
             vill["sep"], sec_pos_meas)
    return ecl, qa


def stage(name, kic, files, p_bin, ecl):
    """Write TIC_<KIC>_<QQ>.txt into lc/ - the cbpvet-side writer (Kepler meta
    has QUARTER not SECTOR, so the finder's filename parser is fed by us)."""
    os.makedirs(LC, exist_ok=True)
    written = []
    for f in files:
        ph = time_to_phase(f["time"], p_bin, ecl["bjd0"])
        arr = np.column_stack([f["time"], f["flux"], f["flux_err"], ph])
        path = os.path.join(LC, f"TIC_{kic}_{f['quarter']:02d}.txt")
        np.savetxt(path, arr, header="TIME FLUX FLUX_ERR PHASE ECL_MASK")
        written.append(path)
    row = {"tess_id": kic, "period": p_bin, "bjd0": ecl["bjd0"],
           "prim_pos": ecl["prim_pos"], "prim_width": ecl["prim_width"],
           "sec_pos": ecl["sec_pos"], "sec_width": ecl["sec_width"],
           "host": name}
    log.info("%s: staged %d quarter files as KIC %d", name, len(written), kic)
    return row


def run_extraction(args, report):
    """Download -> catalog row -> stage -> mask -> search -> sealed QA."""
    os.makedirs(BATCH, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    catalog_rows, host_kic, host_quarters = [], {}, {}
    if not args.skip_download:
        existing = [f for f in os.listdir(LC) if f.endswith(".txt")] if os.path.isdir(LC) else []
        if existing:
            raise SystemExit(f"lc/ already holds {len(existing)} staged files; "
                             "pass --skip-download to reuse them (never clobber)")
        for name, spec in HOSTS.items():
            t0 = _time.time()
            try:
                kic, files = download(name, spec)
                if not files:
                    raise SystemExit(f"{name}: no quarters downloaded")
                ecl, qa = fold_qa_and_widths(name, files, spec["p_bin"], VILLANOVA[name])
                catalog_rows.append(stage(name, kic, files, spec["p_bin"], ecl))
                host_kic[name] = kic
                host_quarters[name] = sorted(f["quarter"] for f in files)
                report["hosts"][name] = {
                    "kic": kic, "status": "extracted",
                    "n_quarters": len(files),
                    "quarters": host_quarters[name],
                    "download_seconds": round(_time.time() - t0, 1),
                    "ephemeris": {
                        "p_bin_verified_facts": spec["p_bin"],
                        "villanova": VILLANOVA[name],
                        **qa,
                    },
                }
            except SystemExit as exc:
                # H4's cap: a missing stretch host is a recorded reduction.
                log.error("%s: HOST FAILED - %s", name, exc)
                report["hosts"][name] = {"status": "FAILED", "reason": str(exc)}
        if not catalog_rows:
            raise SystemExit("both hosts failed; nothing to search")
        pd.DataFrame(catalog_rows).to_csv(os.path.join(BATCH, "kepler_catalog.csv"), index=False)
        json.dump(host_kic, open(os.path.join(BATCH, "host_kic.json"), "w"))
        json.dump(host_quarters, open(os.path.join(BATCH, "host_quarters.json"), "w"))
    else:
        catalog_rows = pd.read_csv(os.path.join(BATCH, "kepler_catalog.csv")).to_dict("records")
        host_kic = json.load(open(os.path.join(BATCH, "host_kic.json")))
        host_quarters = json.load(open(os.path.join(BATCH, "host_quarters.json")))
        for name in host_kic:
            report["hosts"].setdefault(name, {"kic": host_kic[name],
                                              "status": "extracted (reused staging)"})

    catalogue = pd.DataFrame(catalog_rows)

    # ---- mask (in place, on OUR staged copies only) -----------------------
    t0 = _time.time()
    EclipseMasker(catalogue=catalogue, data_dir=LC).mask_all()
    mask_s = _time.time() - t0
    log.info("Masking took %.1f s", mask_s)

    # ---- search: catalogue row mandatory, sector_times None ---------------
    finder = TransitFinderExt(catalogue=catalogue, sector_times=None)
    t0 = _time.time()
    df = finder.process_directory(LC, output_file="detected_events.txt",
                                  output_dir=OUT)
    search_s = _time.time() - t0
    n_files = len([f for f in os.listdir(LC) if f.endswith(".txt")])
    log.info("Search: %d events over %d quarter files in %.1f s (%.2f s/file)",
             len(df), n_files, search_s, search_s / max(n_files, 1))

    report["search"] = {
        "n_quarter_files": n_files,
        "n_events_total": int(len(df)),
        "mask_seconds": round(mask_s, 1),
        "search_seconds": round(search_s, 1),
        "seconds_per_quarter": round(search_s / max(n_files, 1), 2),
    }

    # ---- sealed QA: counts and aggregates ONLY ----------------------------
    # (i) ran without exception: reaching here is the test.
    report["qa"] = {"i_all_quarters_ran": True}

    # Event COUNTS per host/quarter (the only per-event fields touched are the
    # identifiers tic and sector; times/depths/durations are never read).
    if len(df):
        for name, kic in host_kic.items():
            sub = df[df["tic"].astype(int) == int(kic)]
            per_q = sub.groupby(sub["sector"].astype(int)).size()
            report["hosts"][name]["events_flagged_total"] = int(len(sub))
            report["hosts"][name]["events_per_quarter"] = {
                str(q): int(n) for q, n in per_q.items()}
    else:
        for name in host_kic:
            report["hosts"][name]["events_flagged_total"] = 0
            report["hosts"][name]["events_per_quarter"] = {}

    # (ii) no TCE centre inside the eclipse mask.
    inside = 0
    for _, ev in df.iterrows():
        row = catalogue[catalogue.tess_id == int(ev.tic)]
        if row.empty:
            continue
        r = row.iloc[0]
        ph = float(ev.phase)
        for pos, w in ((r.prim_pos, r.prim_width), (r.sec_pos, r.sec_width)):
            if bool(get_eclipse_mask(np.array([ph]), pos, w)[0]):
                inside += 1
                break
    report["qa"]["ii_no_tce_inside_mask"] = {"n_inside": int(inside),
                                             "pass": inside == 0}

    # (iii) snippets written and loadable.
    snip_dir = os.path.join(OUT, "event_snippets")
    snips = os.listdir(snip_dir) if os.path.isdir(snip_dir) else []
    loadable = 0
    for s in snips[:10]:
        d = np.load(os.path.join(snip_dir, s), allow_pickle=True)
        if "time" in d and "flux" in d and len(d["time"]) > 0:
            loadable += 1
    report["qa"]["iii_snippets"] = {"n_written": len(snips),
                                    "n_checked_loadable": loadable,
                                    "pass": len(snips) > 0 and loadable > 0}

    # (iv) DELIBERATELY ABSENT: known-transit matching is sealed until Aug 11
    # (H4 rider 1). Recorded so its absence is never mistaken for an oversight.
    report["qa"]["iv_known_transit_matching"] = "SEALED until the Aug-11 one-shot (H4 rider 1)"

    # (v) runtime logged above.
    report["qa"]["v_runtime_logged"] = True

    for name in HOSTS:
        h = report["hosts"].get(name, {})
        if h.get("status", "").startswith("extracted"):
            h["pass"] = bool(report["qa"]["ii_no_tce_inside_mask"]["pass"]
                             and report["qa"]["iii_snippets"]["pass"])
        else:
            h["pass"] = False
    return report


def resolve_b7(report):
    """B7 second half: TESS TICs for all 10 Kepler hosts, exclusion merge,
    intersection checks. Never blind-overwrites exclusion_tics.txt."""
    from astroquery.mast import Catalogs

    # KICs this project has measured from downloaded data (never from memory).
    known_kic = {}
    for path in (os.path.join(DATA, "kepler_smoke", "host_kic.json"),
                 os.path.join(BATCH, "host_kic.json")):
        if os.path.exists(path):
            known_kic.update({k: int(v) for k, v in json.load(open(path)).items()})

    rows = []
    for name in B7_HOSTS:
        try:
            res = Catalogs.query_object(name, catalog="TIC", radius=0.005)
        except Exception as exc:
            log.error("B7 %s: MAST query failed (%s)", name, type(exc).__name__)
            rows.append({"host": name, "tic": None, "error": str(exc)})
            continue
        if len(res) == 0:
            rows.append({"host": name, "tic": None, "error": "no TIC rows returned"})
            continue
        res.sort("dstArcSec")
        r = res[0]
        tic = int(r["ID"])
        kic_raw = r["KIC"] if "KIC" in res.colnames else None
        try:
            kic_tic = int(kic_raw)
        except (TypeError, ValueError):
            kic_tic = None
        entry = {"host": name, "tic": tic, "kic_from_tic_row": kic_tic,
                 "dst_arcsec": round(float(r["dstArcSec"]), 3),
                 "tmag": (round(float(r["Tmag"]), 3)
                          if np.isfinite(float(r["Tmag"])) else None)}
        if name in known_kic:
            entry["kic_project_measured"] = known_kic[name]
            entry["kic_verified"] = (kic_tic == known_kic[name])
            if not entry["kic_verified"]:
                log.error("B7 %s: TIC row KIC %s != project-measured KIC %d",
                          name, kic_tic, known_kic[name])
        if entry["dst_arcsec"] > 5.0:
            entry["warning"] = "nearest TIC is >5 arcsec from resolved position"
            log.warning("B7 %s: %s", name, entry["warning"])
        rows.append(entry)
        log.info("B7 %s -> TIC %d (KIC %s, %.2f arcsec)", name, tic, kic_tic,
                 entry["dst_arcsec"])

    resolved = {r["host"]: r["tic"] for r in rows if r.get("tic")}

    # ---- intersection checks ---------------------------------------------
    tebc = set(pd.read_csv(TEBC_CSV)["tess_id"].astype(int))
    goodsn = set(int(x) for x in open(GOODSN_IN_TEBC).read().split())
    new_tics = set(resolved.values())
    inter_tebc = sorted(new_tics & tebc)
    inter_deploy = sorted(new_tics & goodsn)

    # ---- merge into exclusion_tics.txt (RE-READ now; append-only) ---------
    lines = open(EXCLUSION).read().rstrip("\n").split("\n")
    existing_tics = set()
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        existing_tics.add(int(ln.split(",")[0]))

    def n_cached(tic):
        n = 0
        for cache in ("lc_cache", "lc_cache_qlp"):
            cdir = os.path.join(DATA, cache)
            if os.path.isdir(cdir):
                n += len([f for f in os.listdir(cdir) if f.startswith(f"TIC_{tic}_")])
        return n

    appended = []
    for r in rows:
        tic = r.get("tic")
        if tic is None or tic in existing_tics:
            continue
        reason = (f"known real transiting circumbinary planet host "
                  f"({r['host']} system; B7 batch 2026-08-07)")
        lines.append(f"{tic},{1 if tic in tebc else 0},{n_cached(tic)},{reason}")
        appended.append(tic)
        existing_tics.add(tic)
    if appended:
        with open(EXCLUSION, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        log.info("B7: appended %d TICs to %s (existing lines untouched)",
                 len(appended), EXCLUSION)
    else:
        log.info("B7: no new TICs to append (all already present)")

    full_inter_tebc = sorted(existing_tics & tebc)

    b7 = {
        "resolutions": rows,
        "n_resolved": len(resolved),
        "n_requested": len(B7_HOSTS),
        "exclusion_file_appended_tics": appended,
        "intersection_kepler_tics_x_tebc591": inter_tebc,
        "intersection_kepler_tics_x_deployment_list": inter_deploy,
        "deployment_list_size": len(goodsn),
        "tebc_size": len(tebc),
        "full_exclusion_x_tebc591": full_inter_tebc,
        "expected": {
            "kepler_x_deployment": "EMPTY",
            "full_exclusion_x_tebc": "[260128333, 319011894] only",
        },
    }
    if inter_deploy:
        log.error("B7 LOUD FINDING: Kepler-host TICs intersect the deployment "
                  "list: %s - NOT fixed silently, reported", inter_deploy)
    json.dump(b7, open(os.path.join(BATCH, "b7_resolution.json"), "w"), indent=2)
    report["b7"] = b7
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse already-staged lc/ files")
    ap.add_argument("--b7-only", action="store_true")
    ap.add_argument("--skip-b7", action="store_true")
    args = ap.parse_args()

    report = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "experiments/29_kepler_batch.py",
        "registry_basis": "H4, Freeze-Decision-Registry_2026-08-06 (riders 1-2)",
        "sealed_discipline": ("counts, mask QA, snippet loadability, runtimes "
                              "ONLY; no event times/properties recorded; no "
                              "exporter/comparator pass; single permitted look "
                              "is the Aug-11 one-shot"),
        "hosts": {},
    }

    if not args.b7_only:
        verify_periods()
        try:
            run_extraction(args, report)
        except SystemExit as exc:
            log.error("EXTRACTION FAILED: %s", exc)
            report["extraction_failure"] = str(exc)

    if not args.skip_b7:
        try:
            resolve_b7(report)
        except Exception as exc:
            log.error("B7 FAILED: %s", exc)
            report["b7_failure"] = f"{type(exc).__name__}: {exc}"

    os.makedirs(BATCH, exist_ok=True)
    out_path = os.path.join(BATCH, "kepler_batch_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report -> %s", out_path)
    for name in HOSTS:
        h = report["hosts"].get(name, {})
        log.info("  %s: %s (pass=%s, events=%s)", name, h.get("status"),
                 h.get("pass"), h.get("events_flagged_total"))


if __name__ == "__main__":
    main()
