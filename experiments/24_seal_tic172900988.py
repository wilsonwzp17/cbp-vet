"""Complete W3.11's sealed pass for TIC 172900988 (the B10 leftover).

The gap this closes (found by the 2026-08-06 final sweep)
---------------------------------------------------------
W3.11's sealed pass scored the real-planet hosts' flagged events and sealed
them before any tuning against real planets. TIC 172900988 contributed ZERO
events at the time (no cached data; recorded in the ledger as the pre-named
reduced scope). The B10 pull then landed on Aug 4 but the scratch
masker+finder step never ran, so the one-shot's TESS D1 pool silently stayed
at the degraded 2-host level. The extraction ran 2026-08-06
(experiments/23_deploy_run.py over scratch COPIES of the pull; 5 files ->
21 events, event times deliberately not examined). This script scores those
events with the deterministically-refit M1-lite model and seals them.

Recorded deviations from the original recipe, with reasons (never silent):
1. SEPARATE FILE, not concatenation. The recipe said "concatenate to
   sealed_scores.csv", but that file is hash-committed in the ledger and
   never reopened; appending would require opening it and would invalidate
   the recorded hash. A sibling sealed file with its own hash preserves both
   the discipline and the recipe's intent.
2. HOST-COLUMN IMPUTATION. This TIC is not in the TEBC catalogue, so
   morph_coeff does not exist for it (it is a TEBC classifier product);
   it is imputed with the M1-lite training median and flagged. Tmag = 9.6319
   was fetched live from the TIC catalog (MAST) on 2026-08-06, not imputed.
3. The refit is asserted against m1lite_summary.json (selected model and
   val PR-AUC must reproduce) before anything is scored, same discipline as
   experiments/21_freeze_models.py.

M1-lite remains PROVISIONAL FOREVER and is never the headline; the one-shot's
real evaluation uses the frozen bench-v1 arm with its own hash-asserted
checkpoint (data/bench/full/models/).
"""

import hashlib
import importlib.util
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EVENTS_NEW = os.path.join(REPO, "data", "tic172900988_scratch", "out",
                          "detected_events.txt")
SUPP_CAT = os.path.join(REPO, "data", "catalogue_tic172900988.csv")
OUT_DIR = os.path.join(REPO, "data", "m1lite")
TMAG_TIC172900988 = 9.6319          # TIC catalog via MAST, fetched 2026-08-06

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("24_seal")


def load_m1lite():
    spec = importlib.util.spec_from_file_location(
        "m1lite", os.path.join(REPO, "experiments", "11_m1lite.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    m = load_m1lite()
    summary = json.load(open(os.path.join(OUT_DIR, "m1lite_summary.json")))

    import pandas as _pd
    from mono_cbp.utils import load_catalogue
    cat = load_catalogue(m.CAT_PATH, TEBC=True).drop_duplicates(
        "tess_id").set_index("tess_id")
    raw = _pd.read_csv(m.CAT_PATH).drop_duplicates("tess_id").set_index("tess_id")
    campaign_dir = os.path.join(REPO, "data", "campaign_out")
    df = m.build_dataset(campaign_dir, cat, raw)
    best, results, _split = m.fit_and_select(df)
    model = results[best]["model"]
    assert best == summary["selected_model"], (best, summary["selected_model"])
    got = results[best]["pr_auc"]
    want = summary["val_scores"][summary["selected_model"]]["pr_auc"]
    assert abs(got - want) < 1e-9, (got, want)
    log.info("M1-lite refit reproduces the recorded model (%s, val PR-AUC "
             "%.6f) - safe to score", best, got)

    ev = pd.read_csv(EVENTS_NEW, sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    ev = ev.rename(columns={"time": "event_time"})
    assert (ev.tic.astype(int) == 172900988).all() and len(ev) > 0

    supp = pd.read_csv(SUPP_CAT)
    morph_median = float(np.nanmedian(df.morph_coeff))
    sub = ev.copy()
    sub["log_p_bin"] = np.log10(float(supp.period.iloc[0]))
    sub["morph_coeff"] = morph_median          # imputed: not a TEBC system
    sub["tmag"] = TMAG_TIC172900988            # real, fetched from MAST
    sub["sin_phase"] = np.sin(2 * np.pi * sub.phase.astype(float))
    sub["cos_phase"] = np.cos(2 * np.pi * sub.phase.astype(float))
    sub = sub.dropna(subset=m.FEATURES)

    scores = model.predict_proba(sub[m.FEATURES].to_numpy(dtype=float))[:, 1]
    sealed = sub[["tic", "sector", "event_time"]].copy()
    sealed["m1lite_score"] = scores

    sealed_dir = os.path.join(OUT_DIR, "sealed")
    path = os.path.join(sealed_dir, "sealed_tic172900988.csv")
    assert not os.path.exists(path), "already sealed; never overwrite"
    sealed.to_csv(path, index=False)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()

    meta = {
        "sealed": "2026-08-06",
        "n_events": int(len(sealed)),
        "source_events": os.path.relpath(EVENTS_NEW, REPO),
        "sha256": h,
        "deviations": [
            "separate sealed file (original sealed_scores.csv is "
            "hash-committed and never reopened)",
            f"morph_coeff imputed with the training median {morph_median:.4f} "
            "(TIC not in TEBC; the column does not exist for it)",
            f"Tmag {TMAG_TIC172900988} fetched from the TIC catalog (MAST), "
            "not imputed",
        ],
        "note": "Hash-committed before any tuning against real planets. "
                "Contents deliberately not examined. M1-lite is provisional "
                "forever; the one-shot uses the frozen bench-v1 checkpoint.",
    }
    with open(os.path.join(sealed_dir, "sealed_tic172900988_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("SEALED %d events for TIC 172900988. sha256 %s", len(sealed), h)
    log.info("The file is NOT inspected. Record the hash in the ledger.")


if __name__ == "__main__":
    main()
