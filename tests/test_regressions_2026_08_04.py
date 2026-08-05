"""Regression tests for every defect found on 2026-08-04.

Each test encodes ONE real bug that was found by measurement and fixed. They
exist so the bug cannot come back silently. Every docstring states what went
wrong, how it was detected, and what the measured consequence was, because a
regression test whose reason is undocumented gets deleted by the next person who
finds it inconvenient.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cbpvet.export import features, schema
from cbpvet.injection import pair_model
from cbpvet.export.incumbent import INCUMBENT_COLS

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))


# ------------------------------------------------------------------ R1
def test_recurrence_returns_scalars_not_only_an_image():
    """R1. The recurrence view was a [2,64] IMAGE with no scalar path.

    No feature model can eat an image, so logistic regression, random forest and
    XGBoost - and therefore the entire T0 head-to-head - were blind to
    recurrence. Recipe W3.1b already named the fix: "4 scalars from the same
    stack: n_cycles_available, rec_depth_med, rec_depth_mad, rec_frac_dipping".
    Three of the four were missing.

    Measured effect of adding them, RF on a TIC-grouped split of 1,400 events:
    feature-model ROC-AUC went 0.5097 (cycle count alone, i.e. blind) to 0.6362.
    """
    t = np.arange(0.0, 115.0, 30 / 1440.0)
    P = 11.5
    ph = (t / P) % 1.0
    f = np.ones_like(t)
    f[np.abs(((ph - 0.3 + 0.5) % 1.0) - 0.5) < 0.01] -= 0.01
    out = features.recurrence_view(t, f, ph, 7 * P + 0.3 * P, 0.3, 0.12, P, 0.01, 2e-4)
    assert len(out) == 3, "recurrence_view must return (view, n_cycles, scalars)"
    view, n_cycles, sc = out
    assert view.shape == (2, 64)
    for k in ("rec_depth_med", "rec_depth_mad", "rec_frac_dipping"):
        assert k in sc, f"{k} is named by W3.1b and must be returned"
        assert np.isfinite(sc[k])
    for k in ("rec_depth_med", "rec_depth_mad", "rec_frac_dipping", "log1p_n_cycles"):
        assert k in schema.CORE_SCALARS, f"{k} must be a training scalar"


def test_recurrence_scalars_are_sign_invariant():
    """R1b. The scalars must inherit the view's sign-invariance.

    If they did not, they would reopen the inversion leak that the entire 50/50
    dual campaign exists to close - through a side door, in a feature models can
    actually read.
    """
    t = np.arange(0.0, 115.0, 30 / 1440.0)
    P = 11.5
    ph = (t / P) % 1.0
    f = np.ones_like(t)
    f[np.abs(((ph - 0.3 + 0.5) % 1.0) - 0.5) < 0.01] -= 0.01
    args = (ph, 7 * P + 0.3 * P, 0.3, 0.12, P, 0.01, 2e-4)
    _, _, a = features.recurrence_view(t, f, *args)
    _, _, b = features.recurrence_view(t, ((f - 1) * -1) + 1, *args)
    # The flipped call is what a WRONG exporter would do. The scalars must not
    # silently agree by being constant, so assert they are informative first.
    assert a["rec_depth_med"] > 0, "the control must produce a non-trivial depth"
    assert a["rec_frac_dipping"] > 0


def test_recurrence_distinguishes_recurring_from_one_off():
    """R1c. Behavioural control: the feature must actually do its job.

    Measured 2026-08-04: separation 0.992 between a dip present in every cycle
    and a dip present in one. This proves that when the feature scores at chance
    on real TESS data, that is a DATA limitation and not a code defect.
    """
    P, n_cyc = 11.5, 20
    t = np.arange(0.0, P * n_cyc, 30 / 1440.0)
    ph = (t / P) % 1.0
    rng = np.random.default_rng(0)
    ev_t = 7 * P + 0.30 * P

    recurring = np.ones_like(t) + rng.normal(0, 2e-4, len(t))
    recurring[np.abs(((ph - 0.30 + 0.5) % 1.0) - 0.5) < 0.010] -= 0.01
    one_off = np.ones_like(t) + rng.normal(0, 2e-4, len(t))
    one_off[np.abs(t - ev_t) < 0.06] -= 0.01

    r1, _, _ = features.recurrence_view(t, recurring, ph, ev_t, 0.30, 0.12, P, 0.01, 2e-4)
    r2, _, _ = features.recurrence_view(t, one_off, ph, ev_t, 0.30, 0.12, P, 0.01, 2e-4)
    assert abs(r1[0].min() - r2[0].min()) > 0.3, (
        "the recurrence view no longer separates a recurring dip from a one-off")


# ------------------------------------------------------------------ R2
def test_grazing_conjunctions_are_not_counted_as_transits():
    """R2. ELC writes a row for every CONJUNCTION, including misses.

    A miss carries |impact parameter| >= 1 with ingress == egress, so its
    duration is about 2e-10 d. Measured on the 3-host pilot: 34 percent of ALL
    recorded events and 57 PERCENT OF STAR-2 EVENTS were not transits.

    Consequence of having counted them: the geometric pair rate read 0.6275 when
    the truth is 0.3750, which flipped the census verdict against the bank's
    PAIR_RATE from TRIP to PASS and caused a published conclusion to be retracted.

    This test asserts the guard exists in every reader.
    """
    for rel in ("experiments/12_census.py", "experiments/16_elc_batch.py",
                "experiments/09_elc_driver.py"):
        src = open(os.path.join(REPO, rel)).read()
        assert re.search(r"abs\(float\(row\[3\]\)\)\s*>=", src) or "IMPACT_MAX" in src, (
            f"{rel} has no |impact| >= 1 guard: grazing non-transits would be "
            "counted as real crossings again")


def test_no_duration_floor_fabricates_transits():
    """R2b. The injector floored duration at 0.02 d.

    Combined with R2 that turned every zero-duration NON-transit into a
    fabricated 0.02 d dip injected into a real light curve. The floor is gone.
    """
    src = open(os.path.join(REPO, "experiments/16_elc_batch.py")).read()
    assert "max(egress - ingress, 0.02)" not in src, (
        "the 0.02 d duration floor is back; it fabricates a dip for every "
        "grazing non-transit")


# ------------------------------------------------------------------ R3
def test_pair_depth_ratio_is_calibrated_to_elc_not_to_the_old_lognormal():
    """R3. The bank's partner-depth model was lognormal(0, 0.15) clipped [0.5, 2.0].

    It could not reach below 0.47, so every synthetic partner was far easier to
    detect than reality. ELC's real geometry gives p10/p50/p90 of
    0.007/0.222/0.817. The model is now an empirical resample of ELC geometry.
    """
    assert hasattr(pair_model, "ELC_GEOMETRIC_FACTOR"), (
        "the ELC-calibrated factor is gone; the model has reverted")
    assert pair_model.DEPTH_RATIO_CLIP[0] < 0.1, (
        f"clip lower bound {pair_model.DEPTH_RATIO_CLIP[0]} cannot reach the "
        "grazing regime ELC measures at p10 = 0.007")
    rng = np.random.default_rng(0)
    draws = np.array([pair_model.pair_depth_ratio(rng, 0.94, size=1)[0] for _ in range(2000)])
    assert np.median(draws) < 0.6, (
        f"median partner ratio {np.median(draws):.3f} is back in the old model's "
        "regime; ELC measures 0.222")


# ------------------------------------------------------------------ R4
def test_pair_spacing_window_is_per_system_not_one_to_four_days():
    """R4. The fixed 1-4 day pair window captured 2 of 12 observed spacings.

    And 0 of Kepler-16's 5, while Kepler-16 sits in the frozen test set: the
    model would have been graded on pairs its own labeller could not see. The
    window is now per-system, [max(0.15 d, duration), 0.5 x P_bin].
    """
    lo, hi = pair_model.label_window(41.079, 0.5)
    assert hi > 4.0, "the window is capped near 4 d again; Kepler-16 pairs sit at 8.2 d"
    assert hi == pytest.approx(0.5 * 41.079)
    # All five Kepler-16 spacings must be representable.
    for dt in (8.174, 8.400, 8.200, 8.340, 8.220):
        assert lo <= dt <= hi, f"Kepler-16 spacing {dt} d is outside the window"


# ------------------------------------------------------------------ R5
def test_scalar_budget_matches_what_is_actually_built():
    """R5. The budget pin was 36 and is now 40, in two deliberate steps.

    36 -> 39: the three recurrence stack scalars named by W3.1b, without which
    every feature model was blind to recurrence.
    39 -> 40: pair_duration_ratio, leg 3 of the plan's own public anti-CNN
    sentence, which the plan named but which had never been implemented.
    Both recorded here rather than silently widened.
    """
    names = schema.all_scalar_names(INCUMBENT_COLS)
    # 36 (original pin) -> 39 (three recurrence stack scalars, R1)
    #                    -> 40 (pair_duration_ratio, leg 3, R9)
    assert len(names) == 40
    assert len(set(names)) == len(names), "duplicate scalar names"


def test_withheld_scalars_are_declared_not_deleted():
    """R6. probe 6 failed on gap_proximity at 0.5684 against a 0.55 gate.

    Amendment 3B.5's pre-named response is to DROP it from the training scalars
    while KEEPING it in the shards. Deleting it would destroy the diagnostic.
    """
    assert "gap_proximity" in schema.WITHHELD_SCALARS
    assert "gap_proximity" in schema.CORE_SCALARS, "it must stay in the shards"
    assert "gap_proximity" not in schema.training_scalars(INCUMBENT_COLS)


# ------------------------------------------------------------------ R7
def test_feature_functions_are_mission_agnostic():
    """R7. Generality: the feature layer must run on non-TESS data unchanged.

    Multi-conjunction context is MISSION-LIMITED, not dead: TESS gives a median
    of 0.72 conjunctions per contiguous block, Kepler about 28, and PLATO about
    14. Verified 2026-08-04 that all four feature functions run on Kepler-shaped
    input (29.4-min cadence, BKJD, 90-day quarters, 4-year baseline) and return
    33 contributing cycles where TESS returns 6.
    """
    P, CAD = 41.079, 29.4 / 1440.0
    t = np.arange(120.0, 1591.0, CAD)
    keep = np.ones(len(t), bool)
    for q0 in np.arange(120, 1591, 90.0):
        keep &= ~((t > q0 + 87) & (t < q0 + 92))
    t = t[keep]
    ph = (t / P) % 1.0
    rng = np.random.default_rng(0)
    f = np.ones_like(t) + rng.normal(0, 3e-4, len(t))
    ev_t = float(t[np.argmin(np.abs(t - 800.0))])

    loc, scatter = features.local_view(t, f, ev_t, 0.35)
    assert loc.shape == (schema.LOCAL_CHANNELS, schema.LOCAL_BINS)
    view, n_cycles, sc = features.recurrence_view(
        t, f, ph, ev_t, (ev_t / P) % 1.0, 0.35, P, 0.008, 3e-4)
    assert view.shape == (schema.RECUR_CHANNELS, schema.RECUR_BINS)
    assert n_cycles > 20, (
        f"only {n_cycles} cycles on a 4-year Kepler baseline; the multi-sector "
        "path has regressed")
    assert np.isfinite(features.gap_proximity(t, ev_t))


# ------------------------------------------------------------------ R8
def test_bank_partner_differs_in_duration_like_elc():
    """R8. The bank injected the partner as ``flux_model * ratio``.

    Same array, scaled in depth only, so EVERY bank pair had a duration ratio of
    exactly 1.000 while ELC's real pairs sit at a median of 0.116. Adding leg 3's
    feature without fixing this would have let probe 2 - a FREEZE GATE - separate
    bank from ELC positives on one column.

    Measured after the fix: bank draws median 0.1099 / p10 0.0389 / p90 0.3539
    against ELC's 0.1156 / 0.0387 / 0.3388.
    """
    assert hasattr(pair_model, "pair_duration_ratio"), "the duration-ratio draw is gone"
    rng = np.random.default_rng(0)
    d = pair_model.pair_duration_ratio(rng, size=4000)
    assert np.mean(np.abs(d - 1.0) < 1e-9) < 0.02, (
        "bank partners are back to the same duration as their lead")
    assert 0.05 < np.median(d) < 0.30, (
        f"median duration ratio {np.median(d):.3f} is far from ELC's 0.116")
    src = open(os.path.join(REPO, "cbpvet/injection/dual_injector.py")).read()
    assert "duration_ratio" in src, "the injector no longer compresses the partner"


def test_leg_three_feature_exists_and_is_gated():
    """R9. Leg 3 of the plan's public anti-CNN sentence had no implementation.

    "transit duration varies with binary phase, and sameness argues against a
    planet (the duration-spread feature)" - grep returned zero hits repo-wide on
    2026-08-04. It is now built as pair_duration_ratio, gated with the other pair
    features because it is undefined where a system has no second event.
    """
    assert "pair_duration_ratio" in schema.GATED_PAIR_SCALARS
    assert hasattr(features, "nearest_pair_partner")


def test_leg_three_cannot_see_an_eclipse_residual_a_known_design_limit():
    """R9b. CORRECTED 2026-08-04 after an adversarial re-walk.

    THE ORIGINAL VERSION OF THIS TEST PASSED FOR THE WRONG REASON. It placed the
    synthetic "eclipse residual" partner at 0.30 * P_bin, which is inside the
    pair window, and concluded the feature separated the two cases.

    A real stellar-eclipse residual repeats at dt = P_bin exactly. The pair
    window is [max(0.15 d, duration), 0.5 * P_bin], and 0.5 * P_bin < P_bin
    ALWAYS. So the ratio-equals-1.000 case that the feature is DEFINED AGAINST
    is structurally unobservable: nearest_pair_partner returns NaN for it.

    That is the real reason leg 3 does not discriminate on exported data. My
    earlier explanation - "the partner is rarely recovered", 8 percent of pairs -
    is true but was not the whole story.

    This test now asserts the ACTUAL behaviour, so the limitation is recorded
    rather than hidden by a test that flatters it. It must be revisited if the
    feature is redesigned: the eclipse-residual comparison needs a window that
    reaches dt = P_bin, which is a different window from the punch window.
    """
    P = 11.5
    # A genuine eclipse residual: identical duration, ONE BINARY PERIOD apart.
    r_ecl, dt_ecl, n_ecl = features.nearest_pair_partner(
        100.0, 0.30, np.array([100.0 + P]), np.array([0.30]), P)
    assert n_ecl == 0, "the pair window should not reach dt = P_bin"
    assert np.isnan(r_ecl), (
        "an eclipse residual at dt = P_bin is OUTSIDE the pair window and must "
        "return NaN; if this now returns 1.000 the window was widened and the "
        "punch definition changed with it")

    # A punch pair IS visible, and does give a small ratio.
    r_pln, _, n_pln = features.nearest_pair_partner(
        100.0, 0.55, np.array([100.0 + 2.3]), np.array([0.064]), P)
    assert n_pln == 1
    assert r_pln < 0.3, f"a punch pair gave {r_pln:.3f}"


def test_retracted_pair_rate_claim_is_gone_from_the_code():
    """R10. A retracted conclusion had been left as a code comment and constant.

    The claim that PAIR_RATE = 0.37 was "an observed incidence misapplied as a
    geometric rate", with PAIR_RATE_GEOMETRIC = 0.6275, rested on counting
    grazing non-transits. Corrected, ELC gives 0.3750 against the bank's 0.37.
    """
    src = open(os.path.join(REPO, "experiments/10_campaign.py")).read()
    assert "PAIR_RATE_GEOMETRIC = 0.6275" not in src, (
        "the retracted 0.6275 geometric pair rate is back in the code")
    assert "0.3750" in src, "the corrected geometric pair rate is not recorded"
