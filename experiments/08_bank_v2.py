"""B1: build the v2 transit-profile bank, 16,384 Sobol-sampled shapes.

Why a new bank
--------------
mono-cbp's stock bank is a 7 x 7 grid of depth by duration, all at impact
parameter 0 and one limb-darkening law: 49 template shapes, repeated. A
classifier trained on injections from that bank can memorise 49 shapes and
score well without learning transit morphology at all. Our own v1 bank
(experiments/02) fixed the variety problem but held only 48 profiles drawn from
plain numpy rng, which is too few to fill a six-dimensional shape space and
leaves visible clumping.

v2 fixes both: 16,384 profiles, and Sobol rather than pseudo-random sampling.

Why Sobol
---------
Sobol is a low-discrepancy sequence. Pseudo-random draws in six dimensions clump
and leave holes; you need a great many of them before every corner of the space
is populated. A Sobol sequence fills space evenly by construction, so the same
number of profiles covers the shape space far better. ``random_base2(m=14)``
gives 2^14 = 16,384 points, and the power-of-two count is required because
Sobol's balance properties only hold on full powers of two.

The six axes
------------
0. depth, log-uniform 0.10 to 1.5 percent. Log because detectability scales
   multiplicatively. The 0.10 percent floor is the depth at which mono-cbp's own
   Table 3 recovery has fallen to 0.13, so below it we would be manufacturing
   almost-all-negative labels.
1. duration, uniform 0.05 to 1.0 days. The ceiling matches
   ``max_duration_days = 1.0``: the search discards anything longer, so a longer
   injection could never be recovered and would poison the positive class with
   guaranteed misses.
2. impact parameter b, 0 to 0.9. Controls V-shaped versus U-shaped profiles and
   admits near-grazing cases.
3. and 4. limb-darkening coefficients.
5. asymmetry, see below.

The asymmetry axis, and why it is here
--------------------------------------
batman transits are symmetric. A real circumbinary transit is not, because the
star being crossed is itself moving during the crossing, so ingress and egress
happen at different relative speeds. If every injected positive is perfectly
symmetric, "is this profile symmetric" becomes a clean shortcut to the label.

So 30 percent of profiles get a piecewise-linear time warp about the transit
centre: the pre-centre side scaled by (1 + alpha), the post-centre side by
(1 - alpha), then re-interpolated onto the uniform grid. The warp is chosen so
that **total duration is preserved exactly**, since the stretched and compressed
halves cancel. Only the ingress-to-egress balance changes, which is the physical
effect we want, and the injector's recovery test against ``duration_model``
stays valid.

Pair mechanics are NOT applied here. Which profiles get a partner, how far
apart, and at what depth ratio all depend on the host system, so they are
assigned at campaign launch. The constants that govern them are pinned in
``cbpvet/injection/pair_model.py`` and copied into this bank's config file so
the bank and the campaign cannot drift apart.

Usage
-----
    python experiments/08_bank_v2.py               # full 16,384
    python experiments/08_bank_v2.py --m 8         # 256, for a quick check
"""

import argparse
import json
import logging
import os
import sys
import time as _time

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils.transit_models import (
    create_transit_models,
    load_transit_models,
    save_transit_models,
)

from cbpvet.injection import pair_model

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
OUT_DIR = os.path.join(REPO, "data", "profile_bank_v2")

# ---- pinned sampling constants -------------------------------------------
SOBOL_SEED = 20260804          # pinned; changing it changes the bank identity
SOBOL_M = 14                   # 2^14 = 16,384 profiles
DEPTH_RANGE = (1.0e-3, 1.5e-2)  # 0.10% to 1.5%, log-uniform
DURATION_RANGE = (0.05, 1.0)    # days; ceiling matches max_duration_days
B_RANGE = (0.0, 0.9)
U1_RANGE = (0.2, 0.5)
U2_RANGE = (0.1, 0.3)
ASYM_FRACTION = 0.30            # fraction of profiles that get warped
ASYM_ALPHA_RANGE = (0.1, 0.3)
CADENCE_MINUTES = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("08_bank_v2")


def lerp(u, lo, hi):
    return lo + u * (hi - lo)


def log_lerp(u, lo, hi):
    return 10.0 ** lerp(u, np.log10(lo), np.log10(hi))


def apply_asymmetry(time, flux, alpha):
    """Piecewise-linear time warp about t=0, preserving total duration.

    The transit is centred at t=0 by construction in create_transit_models, so
    warping about zero warps about the transit centre. Scaling the two sides by
    (1 + alpha) and (1 - alpha) leaves the total in-transit span unchanged while
    making ingress and egress take different amounts of time.
    """
    warped_t = np.where(time < 0.0, time * (1.0 + alpha), time * (1.0 - alpha))
    return np.interp(time, warped_t, flux)


def build(m=SOBOL_M, seed=SOBOL_SEED):
    n = 2 ** m
    sob = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(m=m)
    log.info("Sobol sample: %d points in 6 dimensions (seed %d)", len(sob), seed)

    models, rows, base_time = [], [], None
    t0 = _time.time()
    for i in range(n):
        u = sob[i]
        depth = log_lerp(u[0], *DEPTH_RANGE)
        duration = lerp(u[1], *DURATION_RANGE)
        b = lerp(u[2], *B_RANGE)
        u1 = lerp(u[3], *U1_RANGE)
        u2 = lerp(u[4], *U2_RANGE)

        one = create_transit_models(
            depth_range=(depth, depth), duration_range=(duration, duration),
            num_depths=1, num_durations=1, cadence_minutes=CADENCE_MINUTES,
            impact_parameter=b, limb_dark_coeffs=(u1, u2),
        )
        base_time = one["time"]
        model = one["models"][0]

        # Asymmetry on the lowest 30 percent of the sixth Sobol coordinate.
        pre_warp_depth = float(-model["flux"].min())
        if u[5] < ASYM_FRACTION:
            alpha = ASYM_ALPHA_RANGE[0] + (ASYM_ALPHA_RANGE[1] - ASYM_ALPHA_RANGE[0]) * (
                u[5] / ASYM_FRACTION
            )
            model["flux"] = apply_asymmetry(base_time, model["flux"], alpha)
        else:
            alpha = 0.0

        models.append(model)
        rows.append({
            "model_idx": i,
            "depth": depth,
            "duration": duration,
            "impact_parameter": b,
            "u1": u1,
            "u2": u2,
            "asym_alpha": alpha,
            "ror": model["ror"],
            "pre_warp_depth": pre_warp_depth,
            "realized_depth": float(-model["flux"].min()),
        })

        if (i + 1) % 2000 == 0:
            log.info("  %d/%d profiles (%.1f s)", i + 1, n, _time.time() - t0)

    bank = {
        "time": base_time,
        "models": models,
        # load_transit_models reconstructs the count as num_depths * num_durations,
        # so the whole bank must be declared as one "row" of depths.
        "num_depths": n,
        "num_durations": 1,
        "depth_range": DEPTH_RANGE,
        "duration_range": DURATION_RANGE,
        "cadence_minutes": CADENCE_MINUTES,
    }
    log.info("Built %d profiles in %.1f s", n, _time.time() - t0)
    return bank, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=SOBOL_M, help="Sobol exponent; 2^m profiles")
    ap.add_argument("--seed", type=int, default=SOBOL_SEED)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    bank, rows = build(m=args.m, seed=args.seed)

    bank_path = os.path.join(OUT_DIR, "transit_models_v2.npz")
    save_transit_models(bank, bank_path)
    log.info("Saved bank to %s", bank_path)

    # save_transit_models keeps only flux/depth/duration/b/ror, so the sampled
    # limb darkening and asymmetry would be lost. The sidecar manifest carries
    # every drawn parameter, one row per model, keyed by model_idx.
    import csv
    manifest_path = os.path.join(OUT_DIR, "bank_v2_manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved per-model manifest to %s", manifest_path)

    config = {
        "sobol_seed": args.seed,
        "sobol_m": args.m,
        "n_profiles": 2 ** args.m,
        "depth_range": list(DEPTH_RANGE),
        "duration_range": list(DURATION_RANGE),
        "impact_parameter_range": list(B_RANGE),
        "u1_range": list(U1_RANGE),
        "u2_range": list(U2_RANGE),
        "asymmetry_fraction": ASYM_FRACTION,
        "asymmetry_alpha_range": list(ASYM_ALPHA_RANGE),
        "cadence_minutes": CADENCE_MINUTES,
        "pair_mechanics": pair_model.config_pins(),
        "note": "Limb-darkening naming: Execution-Readiness B1 calls these Kipping "
                "q1/q2, but the ranges it gives (0.2-0.5, 0.1-0.3) are the direct "
                "u1/u2 ranges proven in experiments/02, and they are used as u1/u2 "
                "here. Recorded rather than silently reinterpreted.",
    }
    with open(os.path.join(OUT_DIR, "bank_v2_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    # ---- verification -----------------------------------------------------
    lb = load_transit_models(bank_path)
    assert len(lb["models"]) == 2 ** args.m, "round-trip lost models"
    d = np.array([m["depth"] for m in lb["models"]])
    du = np.array([m["duration"] for m in lb["models"]])
    bb = np.array([m["impact_parameter"] for m in lb["models"]])
    alpha = np.array([r["asym_alpha"] for r in rows])
    realized = np.array([r["realized_depth"] for r in rows])

    log.info("Round-trip OK: %d profiles", len(lb["models"]))
    log.info("  depth    %.4f%% .. %.4f%%  (%d distinct)",
             d.min() * 100, d.max() * 100, len(np.unique(np.round(d, 9))))
    log.info("  duration %.3f .. %.3f d    (%d distinct)",
             du.min(), du.max(), len(np.unique(np.round(du, 6))))
    log.info("  impact b %.3f .. %.3f", bb.min(), bb.max())
    log.info("  asymmetric: %d of %d (%.1f%%), alpha %.3f .. %.3f",
             int((alpha > 0).sum()), len(alpha), 100 * (alpha > 0).mean(),
             alpha[alpha > 0].min() if (alpha > 0).any() else 0.0, alpha.max())
    # (a) The warp itself must not change how deep the transit is. This is the
    # assertion that actually tests our code: same profile, before vs after warp.
    pre = np.array([r["pre_warp_depth"] for r in rows])
    warped = alpha > 0
    warp_rel = np.abs(realized[warped] - pre[warped]) / pre[warped]
    log.info("  WARP depth preservation: max relative change %.2e over %d warped profiles",
             warp_rel.max() if warped.any() else 0.0, int(warped.sum()))
    assert not warped.any() or warp_rel.max() < 1e-3, "asymmetry warp altered transit depth"

    # (b) Requested depth vs the depth batman actually produces. This gap is
    # inherited from mono-cbp's approximate depth-to-radius conversion
    # (ror = sqrt(depth * f0 / f) in create_transit_models) and is present in the
    # stock bank too: it grows with duration and is NOT caused by the warp
    # (measured 2026-08-04: worst case was an UNWARPED profile). Downstream
    # consumers should treat realized_depth as the injected truth.
    # Measured 2026-08-04 over the full 16,384: median 6.1e-4, p99 1.5e-2, max
    # 9.2e-2, and every profile above 5 percent sits at b in [0.885, 0.900].
    # That is the near-grazing corner, where a planet only clips the stellar
    # limb and the area-ratio-with-limb-darkening approximation degrades.
    # Correlation is +0.42 with b and +0.36 with depth, and exactly 0.00 with
    # duration, which is the signature of that mechanism and not of our warp.
    # The assertions below therefore test the MECHANISM rather than imposing an
    # arbitrary ceiling: bulk accuracy must stay tight, and any large error must
    # be near-grazing. If a big error ever shows up at low b, something new is
    # wrong and the run should stop.
    rel = np.abs(realized - d) / d
    tail = rel > 0.05
    log.info("  requested vs realized depth: median %.2e, p99 %.2e, max %.2e",
             np.median(rel), np.percentile(rel, 99), rel.max())
    log.info("  above 5%%: %d of %d (%.3f%%), all at b in [%.3f, %.3f]",
             int(tail.sum()), len(rel), 100 * tail.mean(),
             bb[tail].min() if tail.any() else 0.0, bb[tail].max() if tail.any() else 0.0)
    assert np.median(rel) < 2e-3, "bulk depth conversion drifted"
    assert np.percentile(rel, 99) < 3e-2, "depth conversion p99 drifted"
    assert not tail.any() or bb[tail].min() > 0.85, (
        "a large depth-conversion error appeared away from the near-grazing corner; "
        "the inherited-approximation explanation no longer covers it"
    )
    depth_fidelity = {
        "median_rel_error": float(np.median(rel)),
        "p99_rel_error": float(np.percentile(rel, 99)),
        "max_rel_error": float(rel.max()),
        "n_above_5pct": int(tail.sum()),
        "min_b_above_5pct": float(bb[tail].min()) if tail.any() else None,
        "cause": "inherited mono-cbp ror = sqrt(depth * f0 / f) approximation, "
                 "degrades for near-grazing deep transits; not caused by the warp",
        "consequence": "downstream must use realized_depth as the injected truth, "
                       "not the requested depth column",
    }
    assert du.max() <= DURATION_RANGE[1] + 1e-9, "duration exceeds max_duration_days"
    assert d.min() >= DEPTH_RANGE[0] - 1e-12, "depth below the pinned floor"
    # Re-write the config now that the measured fidelity is known, so the freeze
    # manifest carries the number rather than a promise about it.
    config["depth_fidelity"] = depth_fidelity
    with open(os.path.join(OUT_DIR, "bank_v2_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    log.info("All assertions passed.")


if __name__ == "__main__":
    main()
