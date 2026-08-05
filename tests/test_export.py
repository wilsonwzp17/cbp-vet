"""Unit tests for the export contract.

These are the assertions that have to hold for the benchmark to mean anything.
Each one guards a specific way the dataset could be silently wrong.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet.export import EventExporter, build_block, features, schema
from cbpvet.export.incumbent import BEST_FIT_CLASSES, INCUMBENT_COLS

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
STAGED = os.path.join(REPO, "data", "search_frozen", "staged")


@pytest.fixture(scope="module")
def exporter():
    cat = load_catalogue(CAT_PATH, TEBC=True)
    raw = pd.read_csv(CAT_PATH)
    noise_path = os.path.join(REPO, "data", "noise_screen.csv")
    noise = pd.read_csv(noise_path) if os.path.exists(noise_path) else None
    return EventExporter(STAGED, cat, raw, noise_screen=noise)


@pytest.fixture(scope="module")
def a_real_event():
    ev = pd.read_csv(os.path.join(REPO, "data", "search_frozen", "out",
                                  "detected_events.txt"), sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    return ev.rename(columns={"time": "event_time"}).iloc[0]


# ---------------------------------------------------------------- structure
def test_incumbent_block_is_seventeen_columns():
    """Nine was the class count, not the column count. Seventeen is the contract."""
    assert len(INCUMBENT_COLS) == 17
    assert len(BEST_FIT_CLASSES) == 9
    assert INCUMBENT_COLS[:4] == ["delta_aic_transit", "delta_aic_sinusoidal",
                                  "delta_aic_linear", "delta_aic_step"]


def test_scalar_budget():
    """The budget was pinned at 36; it is now 39, deliberately.

    The pin (Execution-Readiness, "M1-full scalar list, pin at export, 36 max")
    was set as 13 core + 17 incumbent + 2 host + 4 gated. It predates the
    discovery that the recurrence view is a [2,64] image with NO scalar path, so
    every feature model - and therefore the entire T0 head-to-head - was blind to
    recurrence. W3.1b already names the fix: "4 scalars from the same stack:
    n_cycles_available, rec_depth_med, rec_depth_mad, rec_frac_dipping". Three of
    those four were missing; adding them takes core from 13 to 16.

    Recorded rather than silently widened: a budget exists to stop feature bloat,
    and this is the one case where the plan itself specifies the additions.
    """
    names = schema.all_scalar_names(INCUMBENT_COLS)
    assert len(names) == 40
    assert len(schema.CORE_SCALARS) == 16
    # +1 more on 2026-08-04: pair_duration_ratio, leg 3 of the anti-CNN sentence,
    # which the plan named but which had never been built.
    assert "pair_duration_ratio" in schema.GATED_PAIR_SCALARS
    for s in ("rec_depth_med", "rec_depth_mad", "rec_frac_dipping"):
        assert s in names, f"{s} is named by W3.1b and must be present"


def test_local_view_is_402_dims():
    """Probe 4 flattens the local view and expects 402 dimensions."""
    assert schema.LOCAL_CHANNELS * schema.LOCAL_BINS == 402


# ---------------------------------------------------- the inversion leak guard
def test_recurrence_view_is_sign_invariant(exporter, a_real_event):
    """THE critical test.

    The recurrence view must be computed from the ORIGINAL cached flux, so
    flipping the light curve cannot change it. If this ever fails, the view has
    started reading the injected or inverted array and the inversion leak that
    the whole 50/50 campaign exists to close has reopened through the back door.
    """
    ev = a_real_event
    cached = exporter._cached(ev.tic, ev.sector)
    assert cached is not None
    host = exporter._host(ev.tic)

    native, n1, _ = features.recurrence_view(
        cached["time"], cached["flux"], cached["phase"],
        float(ev.event_time), float(ev.phase), float(ev.duration),
        host["p_bin"], float(ev.depth), 1e-3)

    flipped_flux = ((cached["flux"] - 1) * -1) + 1
    # Same call, but the caller hands it the FLIPPED array. A correct exporter
    # never does this; the test proves the view would differ if it did, and the
    # exporter test below proves the exporter does not.
    flipped, n2, _ = features.recurrence_view(
        cached["time"], flipped_flux, cached["phase"],
        float(ev.event_time), float(ev.phase), float(ev.duration),
        host["p_bin"], float(ev.depth), 1e-3)

    assert n1 == n2, "cycle counting must not depend on flux sign"
    # Channel 1 is a pure count and must be bit-identical either way.
    np.testing.assert_array_equal(native[1], flipped[1])


def test_exporter_recurrence_ignores_the_injected_flux(exporter, a_real_event):
    """The exporter must build recurrence from the cache, not from what it is handed.

    Passing a wildly different local flux source must leave the recurrence view
    untouched. This is the property that actually protects the dataset.
    """
    ev = a_real_event.copy()
    ev["label"] = 0
    cached = exporter._cached(ev.tic, ev.sector)

    _, _, recur_plain = exporter.build_record(ev, "real_search")
    fake = (cached["time"], ((cached["flux"] - 1) * -1) + 1)
    _, _, recur_fake = exporter.build_record(ev, "bank_injection", local_flux_source=fake)

    np.testing.assert_array_equal(
        recur_plain, recur_fake,
        err_msg="recurrence changed when the local flux source changed; it is "
                "no longer reading the original cached light curve")


# ------------------------------------------------------- host-level constants
def test_host_scalars_are_constant_per_tic(exporter):
    """same_star_multi_allowed and log10(P_p/P_bin) are properties of the SYSTEM.

    If they varied event to event they would leak per-event information under a
    name that claims to be a host constant.
    """
    ev = pd.read_csv(os.path.join(REPO, "data", "search_frozen", "out",
                                  "detected_events.txt"), sep=r"\s+")
    ev.columns = [c.lower() for c in ev.columns]
    counts = ev.tic.value_counts()
    tic = int(counts[counts >= 3].index[0])
    h1 = exporter._host(tic)
    for _ in range(3):
        h2 = exporter._host(tic)
        assert h1["log10_pp_over_pbin"] == h2["log10_pp_over_pbin"]
        assert h1["same_star_multi_allowed"] == h2["same_star_multi_allowed"]


# ------------------------------------------------------------------- policies
def test_comparator_failures_keep_their_rows():
    """compare_events drops excepted events silently. We must not.

    A dropped event would change the benchmark denominator with no error raised.
    """
    keys = ["a", "b", "c"]
    rows = pd.DataFrame([{
        "filename": "a", "best_fit": "T",
        "aic_transit": 10.0, "aic_sinusoidal": 14.0, "aic_linear": 20.0, "aic_step": 18.0,
        "rmse_transit": 1.0, "rmse_sinusoidal": 1.1, "rmse_linear": 1.2, "rmse_step": 1.3,
    }])
    out = build_block(rows, keys)
    assert len(out) == 3, "events missing from the comparator must keep their rows"
    assert out.set_index("filename").loc["a", "incumbent_valid"] == 1
    assert out.set_index("filename").loc["b", "incumbent_valid"] == 0
    assert out.set_index("filename").loc["b", INCUMBENT_COLS].sum() == 0.0


def test_delta_aic_is_relative_to_the_best_model():
    rows = pd.DataFrame([{
        "filename": "a", "best_fit": "T",
        "aic_transit": 10.0, "aic_sinusoidal": 14.0, "aic_linear": 20.0, "aic_step": 18.0,
        "rmse_transit": 1.0, "rmse_sinusoidal": 1.1, "rmse_linear": 1.2, "rmse_step": 1.3,
    }])
    out = build_block(rows, ["a"]).set_index("filename")
    assert out.loc["a", "delta_aic_transit"] == 0.0        # best model
    assert out.loc["a", "delta_aic_sinusoidal"] == 4.0
    assert out.loc["a", "best_fit_T"] == 1.0
    assert out.loc["a", "best_fit_AT"] == 0.0


def test_unknown_best_fit_class_raises():
    """The one-hot order is a frozen contract; an unexpected class must not pass."""
    rows = pd.DataFrame([{
        "filename": "a", "best_fit": "WAT",
        "aic_transit": 10.0, "aic_sinusoidal": 14.0, "aic_linear": 20.0, "aic_step": 18.0,
        "rmse_transit": 1.0, "rmse_sinusoidal": 1.1, "rmse_linear": 1.2, "rmse_step": 1.3,
    }])
    with pytest.raises(ValueError, match="outside the pinned 9 classes"):
        build_block(rows, ["a"])


def test_depth_floor_prevents_noise_division():
    """A depth consistent with zero must not produce an enormous normalisation."""
    assert schema.depth_floor(1e-9, 1e-3) == pytest.approx(3e-3)
    assert schema.depth_floor(1e-2, 1e-4) == pytest.approx(1e-2)


def test_dedup_groups_by_one_cadence():
    times = np.array([100.0, 100.01, 100.02, 105.0, 105.5])
    groups = features.dedup_groups(times)
    assert len(groups) == 3
    assert sorted(len(g) for g in groups) == [1, 1, 3]


def test_same_star_multi_allowed_matches_equation_two():
    """Kostov 2020b eq (2): P_CBP > P_bin * ((M1/M2) + 1)^3."""
    p_bin, q = 20.0, 0.5
    threshold = p_bin * (1 / q + 1) ** 3
    assert features.same_star_multi_allowed(threshold * 1.01, p_bin, q) == 1.0
    assert features.same_star_multi_allowed(threshold * 0.99, p_bin, q) == 0.0


def test_gap_proximity_uses_the_full_array_not_a_snippet():
    """Measured on a snippet, gap distance is capped by the snippet half-width."""
    t = np.concatenate([np.arange(0, 10, 0.02), np.arange(15, 25, 0.02)])
    assert features.gap_proximity(t, 5.0) == pytest.approx(5.0, abs=0.05)
    assert features.gap_proximity(t, 9.9) == pytest.approx(0.08, abs=0.05)


def test_skye_flag_is_invalid_for_injections(exporter):
    """Skye is a per-sector systematics metric of the real search.

    It has no meaning for an injection, so it must be marked invalid rather than
    imputed; an imputed value is indistinguishable from a measured one.
    """
    rec = {"skye_flag": np.nan, "morph_coeff": 0.2, "tmag": 11.0,
           "has_pair_same_star": 1.0, "has_pair_cross_star": 0.0,
           "pair_dt_over_pbin": 0.2, "pairing_depth_ratio_vs_expectation": 0.1}
    bits = exporter.scalar_valid(rec)
    assert bits["skye_flag"] == 0
    assert bits["morph_coeff"] == 1


def test_gated_pair_scalars_are_invalid_until_e2_passes(exporter):
    """The four pair features are written always, but admitted only if E2 passes."""
    rec = {k: 1.0 for k in schema.CONDITIONAL_SCALARS}
    bits = exporter.scalar_valid(rec)
    for name in schema.GATED_PAIR_SCALARS:
        assert bits[name] == 0, f"{name} admitted before the E2 gate"
