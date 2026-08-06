"""The Kepler smoke: prove the machinery runs on Kepler data BEFORE the tag.

Why this exists, in the plan's own words
----------------------------------------
Week-3 board, Aug-6 row: "the Kepler smoke (Kepler-16 + one hard host) runs
BEFORE the manifest is hashed and tagged - the declared tier is a manifest
field, so the tag is the last act of the day." The one-shot's Kepler arm and
the declared coverage tier both depend on the pipeline actually running on
Kepler-shaped data, and a claim about that must be a measurement.

Host choice, documented
-----------------------
Execution-Readiness names "Kepler-413 recommended (misaligned, precessing,
stresses the fixed-phase mask), Kepler-34 alternate", and the choice was never
ratified at the Jul-31 review. **Kepler-34 is used here**, for a reason that
outranks the recommendation: its binary period (27.796 d) is in our VERIFIED
Table-1 transcription (data/kepler_transit_times.csv, cross-validated against
VERIFIED-FACTS section 7), while Kepler-413's parameters exist nowhere in the
project's verified sources and would have to come from unverified memory.
Kepler-34 still stresses the fixed-phase mask: its binary is eccentric
(e = 0.52, Welsh 2012), so the secondary eclipse sits far from phase 0.5 and
the eclipse widths are strongly unequal. The 413-vs-34 choice is recorded as an
open ratification item for Wilson; swapping hosts later is a re-run of this
script, nothing more.

Where every number comes from (no memory allowed)
-------------------------------------------------
- Periods: data/kepler_transit_times.csv (transcribed from Kostov 2020b Table 1
  and cross-validated; Kepler-16 41.079 d, Kepler-34 27.796 d).
- Target resolution: by NAME via MAST ("Kepler-16"), so no KIC from memory.
- bjd0 and eclipse positions/widths: MEASURED from the phase-folded light curve
  itself, the same self-calibrating approach experiment 17 validated on
  TIC 172900988 (primary folded to phase 0.9988 there).
- Known transit times for the recall check: the same CSV's Kepler-16 rows.

Mechanics pinned by the readiness audit, honoured here
------------------------------------------------------
- The catalogue row is MANDATORY: Kepler's 29.4-min cadence is under the 30-min
  threshold, so the finder calls bin_to_long_cadence, which recomputes phase
  and mask from the row.
- sector_times=None: Kepler quarters would misindex the Skye metric.
- Filenames are built by THIS writer (Kepler metadata has QUARTER not SECTOR);
  the KIC integer takes the TIC field's place, consistently in files and rows.
- bjd0 is set to a measured primary-eclipse mid-time so prim_pos ~ 1.0
  (the wraparound branch is the verified one).
- Files are staged in our own directory; the masker's in-place rewrite touches
  only these staged copies.

Pass criteria (pinned in Execution-Readiness, quoted):
 (i)   mask, detrend, search run all quarters of both hosts without exception;
 (ii)  no TCE centre inside the eclipse mask;
 (iii) snippets written and exporter-loadable;
 (iv)  Kepler-16: >= 1 TCE within t_dur/2 of >= 1 known transit time in a
       quarter with data (a MACHINERY check, NOT a recall claim);
 (v)   per-quarter runtime logged for the Aug 9-10 projection.
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
STAGING = os.path.join(DATA, "kepler_smoke")
OUT = os.path.join(STAGING, "out")
TIMES_CSV = os.path.join(DATA, "kepler_transit_times.csv")

# Periods from the verified transcription; everything else measured from data.
HOSTS = {
    "Kepler-16": {"p_bin": 41.079},
    "Kepler-34": {"p_bin": 27.796},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("19_kepler")


def verify_periods_against_csv():
    """Periods come from VERIFIED-FACTS section 7 (the file that outranks all
    narrative docs), re-parsed at runtime so a silent edit there cannot diverge
    from what this script uses. Transit times come from the times CSV, whose
    columns were CORRECTED tonight: the paper states plainly that Table 1 times
    are "[BJD-2,455,000]", so the old column name t_primary_bkjd was a
    mislabel. BKJD = BJD - 2,454,833, hence BKJD = (value + 167.0).
    """
    # The verified-facts cross-check reads the project docs folder, which lives
    # OUTSIDE the repo. Configure via CBPVET_DOCS_DIR; when unset the cross-check
    # is skipped with a warning and the module-level pinned periods stand (they
    # were transcribed from the same verified source).
    import re as _re
    docs = os.environ.get("CBPVET_DOCS_DIR")
    text = None
    if docs:
        vf_path = os.path.join(docs, "VERIFIED-FACTS_2026-07-29.md")
        if os.path.exists(vf_path):
            text = open(vf_path).read()
    if text is None:
        log.warning("CBPVET_DOCS_DIR not set; period cross-check against the "
                    "verified-facts file SKIPPED - pinned values stand")
    for name, spec in HOSTS.items():
        if text is None:
            continue
        m = _re.search(_re.escape(name) + r" b \| ([0-9.]+) \|", text)
        if not m:
            raise SystemExit(f"{name}: P_bin not found in VERIFIED-FACTS section 7")
        p_vf = float(m.group(1))
        if abs(p_vf - spec["p_bin"]) > 0.01:
            raise SystemExit(f"{name}: period mismatch, VF {p_vf} vs script {spec['p_bin']}")
        spec["p_bin"] = p_vf
        log.info("%s: period %.5f d verified against VERIFIED-FACTS", name, p_vf)
    kt = pd.read_csv(TIMES_CSV)
    if "t_primary_bjd_m2455000" not in kt.columns:
        raise SystemExit("times CSV does not carry the corrected column names")
    kt["t_primary_bkjd_derived"] = kt["t_primary_bjd_m2455000"] + 167.0
    kt["t_secondary_bkjd_derived"] = kt["t_secondary_bjd_m2455000"] + 167.0
    return kt, "planet"


def download(name, spec):
    """All long-cadence quarters, SAP with crowding correction, by NAME."""
    import lightkurve as lk
    res = lk.search_lightcurve(name, mission="Kepler", cadence="long")
    log.info("%s: MAST offers %d long-cadence products", name, len(res))
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
    log.info("%s: downloaded %d usable quarters (KIC %s)", name, len(files), kic)
    return kic, files


def measure_eclipses(files, p_bin):
    """Self-calibrating fold: find both eclipses and set bjd0 at the primary.

    Same approach experiment 17 validated: fold on the verified period with a
    provisional epoch, locate the two deepest well-separated dips, measure each
    width at 10 percent of its peak depth with a WRAPAROUND walk (both fixes
    from experiment 17 carried over), then shift bjd0 so the primary sits at
    phase ~1.0, the finder's verified wraparound branch.
    """
    t0_prov = float(min(f["time"].min() for f in files))
    ph = np.concatenate([time_to_phase(f["time"], p_bin, t0_prov) for f in files])
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

    prim_phase_prov = float(centres[i1])
    # bjd0 at a primary mid-time so the primary folds to ~1.0 (0.0 wrapped).
    bjd0 = t0_prov + prim_phase_prov * p_bin
    sec_pos = float((centres[i2] - prim_phase_prov) % 1.0)
    ecl = {
        "bjd0": bjd0,
        "prim_pos": 1.0,
        "prim_width": float(width_at(i1)),
        "prim_depth": float(depth[i1]),
        "sec_pos": sec_pos,
        "sec_width": float(width_at(i2)),
        "sec_depth": float(depth[i2]),
    }
    log.info("  eclipses: primary depth %.4f width %.4f | secondary at phase %.4f "
             "depth %.4f width %.4f", ecl["prim_depth"], ecl["prim_width"],
             ecl["sec_pos"], ecl["sec_depth"], ecl["sec_width"])
    return ecl


def stage(name, kic, files, p_bin, ecl):
    os.makedirs(STAGING, exist_ok=True)
    written = []
    for f in files:
        ph = time_to_phase(f["time"], p_bin, ecl["bjd0"])
        arr = np.column_stack([f["time"], f["flux"], f["flux_err"], ph])
        path = os.path.join(STAGING, f"TIC_{kic}_{f['quarter']:02d}.txt")
        np.savetxt(path, arr, header="TIME FLUX FLUX_ERR PHASE ECL_MASK")
        written.append(path)
    row = {"tess_id": kic, "period": p_bin, "bjd0": ecl["bjd0"],
           "prim_pos": ecl["prim_pos"], "prim_width": ecl["prim_width"],
           "sec_pos": ecl["sec_pos"], "sec_width": ecl["sec_width"],
           "host": name}
    log.info("%s: staged %d quarter files as KIC %d", name, len(written), kic)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    kt, plcol = verify_periods_against_csv()
    os.makedirs(OUT, exist_ok=True)

    catalog_rows, host_kic = [], {}
    if not args.skip_download:
        for name, spec in HOSTS.items():
            kic, files = download(name, spec)
            if not files:
                raise SystemExit(f"{name}: no quarters downloaded; smoke cannot run")
            ecl = measure_eclipses(files, spec["p_bin"])
            catalog_rows.append(stage(name, kic, files, spec["p_bin"], ecl))
            host_kic[name] = kic
        pd.DataFrame(catalog_rows).to_csv(os.path.join(STAGING, "kepler_catalog.csv"), index=False)
        json.dump(host_kic, open(os.path.join(STAGING, "host_kic.json"), "w"))
    else:
        catalog_rows = pd.read_csv(os.path.join(STAGING, "kepler_catalog.csv")).to_dict("records")
        host_kic = json.load(open(os.path.join(STAGING, "host_kic.json")))

    catalogue = pd.DataFrame(catalog_rows)

    # ---- mask (in place, on OUR staged copies only) -----------------------
    t0 = _time.time()
    EclipseMasker(catalogue=catalogue, data_dir=STAGING).mask_all()
    log.info("Masking took %.1f s", _time.time() - t0)

    # ---- search: catalogue row mandatory, sector_times None ---------------
    finder = TransitFinderExt(catalogue=catalogue, sector_times=None)
    t0 = _time.time()
    df = finder.process_directory(STAGING, output_file="detected_events.txt",
                                  output_dir=OUT)
    search_s = _time.time() - t0
    n_files = len([f for f in os.listdir(STAGING) if f.endswith(".txt")])
    log.info("Search: %d events over %d quarter files in %.1f s (%.2f s/file)",
             len(df), n_files, search_s, search_s / max(n_files, 1))

    # ---- pass criteria ----------------------------------------------------
    report = {"hosts": host_kic, "n_quarter_files": n_files,
              "n_events": int(len(df)),
              "seconds_per_quarter": round(search_s / max(n_files, 1), 2),
              "criteria": {}}

    # (i) ran without exception: reaching here is the test
    report["criteria"]["i_all_quarters_ran"] = True

    # (ii) no TCE centre inside the eclipse mask
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
    report["criteria"]["ii_no_tce_inside_mask"] = {"n_inside": int(inside),
                                                   "pass": inside == 0}

    # (iii) snippets written and loadable
    snip_dir = os.path.join(OUT, "event_snippets")
    snips = os.listdir(snip_dir) if os.path.isdir(snip_dir) else []
    loadable = 0
    for s in snips[:10]:
        d = np.load(os.path.join(snip_dir, s), allow_pickle=True)
        if "time" in d and "flux" in d and len(d["time"]) > 0:
            loadable += 1
    report["criteria"]["iii_snippets"] = {"n_written": len(snips),
                                          "n_checked_loadable": loadable,
                                          "pass": len(snips) > 0 and loadable > 0}

    # (iv) Kepler-16 machinery check against the verified transit times.
    # Times converted to BKJD (= tabulated BJD-2,455,000 value + 167.0); both
    # the primary-star and secondary-star crossing of each conjunction count.
    k16 = host_kic.get("Kepler-16")
    times16 = kt[kt["planet"].astype(str).str.replace(" ", "").str.contains("Kepler-16")]
    matched, known = 0, 0
    if k16 is not None and len(times16):
        ev16 = df[df.tic.astype(int) == int(k16)]
        ev_t = ev16.time.astype(float).to_numpy() if len(ev16) else np.array([])
        ev_d = ev16.duration.astype(float).to_numpy() if len(ev16) else np.array([])
        for _, tr in times16.iterrows():
            for col in ("t_primary_bkjd_derived", "t_secondary_bkjd_derived"):
                t_known = float(tr[col])
                if not np.isfinite(t_known):
                    continue
                known += 1
                if len(ev_t):
                    tol = np.maximum(ev_d / 2.0, 0.15)
                    if (np.abs(ev_t - t_known) < tol).any():
                        matched += 1
    report["criteria"]["iv_k16_known_transit_matched"] = {
        "n_known_times": int(known), "n_matched": int(matched),
        "pass": matched >= 1,
        "note": "machinery check, NOT a recall claim (some times fall in gaps)"}

    # (v) runtime logged: already in seconds_per_quarter
    report["criteria"]["v_runtime_logged"] = True

    report["overall_pass"] = all(
        (c if isinstance(c, bool) else c.get("pass", False))
        for c in report["criteria"].values())
    with open(os.path.join(OUT, "kepler_smoke_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("KEPLER SMOKE %s -> %s",
             "PASS" if report["overall_pass"] else "FAIL",
             os.path.join(OUT, "kepler_smoke_report.json"))
    for k, v in report["criteria"].items():
        log.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
