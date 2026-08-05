"""B7 (the exclusion list) and W3.3c (the E2 outlook recount).

Two small but load-bearing pieces.

B7: the exclusion list
----------------------
Some TICs must never appear in training data, because they carry the known real
planets the benchmark is graded on. Injecting into them, or letting them into a
train split, would contaminate the test set with the very systems the one-shot
is supposed to evaluate blind.

No machine-readable exclusion list existed anywhere in the project. The rule
lived only in prose, which means every consumer had to remember it. Now it is a
file, and the campaign and the exporter read it rather than each carrying their
own literal.

W3.3c: the E2 outlook recount
-----------------------------
E2 is the gate that asks whether enough distinct systems will yield a detectable
1-2 punch to make the pair analysis worth reporting. The floor is 30 to 40
distinct TICs.

The chain, per host:

    T_obs      total observed baseline, summed over the host's cached sectors
    P_crit     innermost stable circumbinary period (Holman-Wiegert, mu = 0.3)
    P_p        planet period drawn above P_crit
    f2(sigma)  probability of two transits per conjunction, from Chen and
               Kipping 2022 Table 1 Scenario A
    r          search recovery at the drawn depth, from mono-cbp Table 3
    lambda     expected number of detected pairs
             = (T_obs / P_p) * f2(sigma) * 0.8 * r_lead * r_partner
    p_host     1 - exp(-lambda), times 0.85 for QA and windowing attrition

The 0.8 factor is the duty-cycle allowance: not every conjunction lands inside
observed data. Summing p_host over hosts gives the expected number of distinct
TICs contributing at least one detected pair.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet import physics

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
CACHE = os.path.join(REPO, "data", "lc_cache")
NOISE = os.path.join(REPO, "data", "noise_screen.csv")
DATA = os.path.join(REPO, "data")

# The three TICs that carry known real planets in our TESS sample.
TESS_PLANET_TICS = [260128333, 319011894, 172900988]

# Chen and Kipping 2022 Table 1, Scenario A: P(two transits per conjunction).
CK_SIGMA = np.array([1e-4, 1e-3, 3e-3, 1e-2, 3e-2])
CK_F2 = np.array([0.445, 0.367, 0.326, 0.261, 0.181])

DUTY_CYCLE = 0.8          # conjunctions that fall inside observed data
ATTRITION = 0.85          # QA plus windowing survival
MEAN_RECOVERY = 0.41      # mono-cbp Table 3, depth-weighted mean
E2_FLOOR = 30             # distinct TICs; the gate is 30 to 40

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("15_excl_e2")


def build_exclusion_list():
    """Write data/exclusion_tics.txt and check it against the frozen catalogue."""
    cat = pd.read_csv(CAT_PATH).drop_duplicates("tess_id")
    catalog_tics = set(cat.tess_id.astype(int))

    rows = []
    for tic in TESS_PLANET_TICS:
        in_cat = tic in catalog_tics
        n_files = len([f for f in os.listdir(CACHE) if f.startswith(f"TIC_{tic}_")])
        rows.append({"tic": tic, "in_catalogue": int(in_cat), "n_cached_files": n_files,
                     "reason": "known real transiting circumbinary planet host"})
        log.info("  TIC %d: in catalogue %s, %d cached files", tic, in_cat, n_files)

    # The recipe expects exactly two of the three to be in the frozen catalogue.
    # CHECK rather than assume, per B7.
    present = {r["tic"] for r in rows if r["in_catalogue"]}
    expected = {260128333, 319011894}
    if present != expected:
        log.warning("Catalogue intersection is %s, expected %s. Not fatal, but the "
                    "frozen denominator assumption must be revisited.", present, expected)
    else:
        log.info("Catalogue intersection is exactly {260128333, 319011894}, as expected")

    df = pd.DataFrame(rows)
    path = os.path.join(DATA, "exclusion_tics.txt")
    with open(path, "w") as fh:
        fh.write("# TICs excluded from ALL training data.\n")
        fh.write("# They carry known real transiting circumbinary planets and are the\n")
        fh.write("# benchmark's test set; injecting into them or admitting them to a\n")
        fh.write("# train split would contaminate the one-shot evaluation.\n")
        fh.write("# tic,in_catalogue,n_cached_files,reason\n")
        for r in rows:
            fh.write(f"{r['tic']},{r['in_catalogue']},{r['n_cached_files']},{r['reason']}\n")
    log.info("Wrote %s (%d TICs)", path, len(rows))
    return df


def e2_recount(top_n=150, n_models=24, seed=20260804):
    """Expected number of distinct TICs yielding at least one DETECTED pair."""
    rng = np.random.default_rng(seed)
    noise = pd.read_csv(NOISE)
    noise = noise[np.isfinite(noise.depth_ratio) & np.isfinite(noise.rms_cadence)]
    noise["partner_depth"] = 0.005 * noise.depth_ratio
    noise["partner_snr"] = noise.partner_depth / (noise.rms_cadence / np.sqrt(12))
    eligible = noise[(noise.depth_ratio >= 0.33) & (noise.partner_snr >= 5)]
    eligible = eligible.sort_values("partner_snr", ascending=False)
    log.info("N_physical (three-part screen): %d", len(eligible))

    cat = load_catalogue(CAT_PATH, TEBC=True).drop_duplicates("tess_id").set_index("tess_id")
    raw = pd.read_csv(CAT_PATH).drop_duplicates("tess_id").set_index("tess_id")

    per_host, used = [], 0
    for _, r in eligible.iterrows():
        if used >= top_n:
            break
        tic = int(r.tic)
        if tic not in cat.index or tic not in raw.index:
            continue
        files = [f for f in os.listdir(CACHE) if f.startswith(f"TIC_{tic}_")]
        if not files:
            continue
        t_obs = 0.0
        for f in files:
            arr = np.loadtxt(os.path.join(CACHE, f), skiprows=1, usecols=0)
            if arr.size >= 2:
                t_obs += float(np.nanmax(arr) - np.nanmin(arr))
        if t_obs <= 0:
            continue

        rr = raw.loc[tic]
        _, _, ecc = physics.eclipse_eccentricity(
            rr.get("prim_pos_2g"), rr.get("sec_pos_2g"),
            rr.get("prim_width_2g"), rr.get("sec_width_2g"))
        ecc = float(ecc) if np.isfinite(ecc) else 0.0
        p_bin = float(cat.loc[tic, "period"])

        # One draw per model, matching the ELC allocation: 60 percent tight.
        sigmas = np.where(rng.random(n_models) < 0.60,
                          10 ** rng.uniform(-4, -3, n_models),
                          10 ** rng.uniform(-3, -2, n_models))
        f2 = np.interp(np.log10(sigmas), np.log10(CK_SIGMA), CK_F2)
        p_p = physics.draw_planet_period(rng, p_bin, ecc, size=n_models)

        # Recovery of each member of the pair, drawn about the Table 3 mean.
        r_lead = np.clip(rng.normal(MEAN_RECOVERY, 0.12, n_models), 0.05, 0.95)
        r_partner = np.clip(rng.normal(MEAN_RECOVERY, 0.12, n_models), 0.05, 0.95)

        lam = (t_obs / p_p) * f2 * DUTY_CYCLE * r_lead * r_partner
        p_m = (1.0 - np.exp(-lam)) * ATTRITION
        p_host = 1.0 - np.prod(1.0 - p_m)
        per_host.append({"tic": tic, "t_obs": t_obs, "p_bin": p_bin, "ecc": ecc,
                         "p_crit": physics.p_crit(p_bin, ecc),
                         "partner_snr": float(r.partner_snr),
                         "median_lambda": float(np.median(lam)),
                         "p_host": float(p_host)})
        used += 1

    df = pd.DataFrame(per_host)
    expected = float(df.p_host.sum())
    verdict = "POSITIVE" if expected >= E2_FLOOR else "NEGATIVE"
    log.info("E2 recount over %d hosts: expected distinct paired TICs = %.1f "
             "(floor %d) -> %s", len(df), expected, E2_FLOOR, verdict)
    log.info("  hosts with p_host > 0.5: %d;  median p_host %.3f",
             int((df.p_host > 0.5).sum()), float(df.p_host.median()))

    df.to_csv(os.path.join(DATA, "e2_recount_per_host.csv"), index=False)
    out = {
        "n_physical": int(len(eligible)), "n_hosts_evaluated": int(len(df)),
        "expected_paired_tics": expected, "floor": E2_FLOOR, "verdict": verdict,
        "margin_over_floor": expected - E2_FLOOR,
        "hosts_above_half": int((df.p_host > 0.5).sum()),
        "pins": {"duty_cycle": DUTY_CYCLE, "attrition": ATTRITION,
                 "mean_recovery": MEAN_RECOVERY, "mu": physics.MU_PINNED,
                 "ck_sigma": CK_SIGMA.tolist(), "ck_f2": CK_F2.tolist(),
                 "n_models_per_host": n_models, "seed": seed},
        "note": "OUTLOOK ONLY. The binding E2 decision is made on the measured "
                "ELC batch and campaign harvest, not on this projection.",
    }
    with open(os.path.join(DATA, "e2_recount.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=150)
    args = ap.parse_args()
    print("\n--- B7: exclusion list ---")
    build_exclusion_list()
    print("\n--- W3.3c: E2 outlook recount ---")
    res = e2_recount(top_n=args.top_n)
    print(json.dumps({k: v for k, v in res.items() if k != "pins"}, indent=2))
