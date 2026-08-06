"""The frozen record schema: what one event looks like to a model.

Every event, whether a real-search negative, a bank injection, or an ELC
injection, becomes exactly one record with the same fields computed by the same
code. That sameness is the whole point. Anywhere the two classes are built
differently, the difference itself becomes learnable and the benchmark stops
measuring what it claims to measure.

A record has three parts.

1. The local view, [2, 201]
   The event itself. Channel 0 is normalised flux on a fixed grid centred on the
   event; channel 1 is a validity mask marking which grid points had real data.
   The mask channel matters because gaps are informative but must be declared,
   not smuggled in as zeros that the model reads as flux.

2. The recurrence view, [2, 64]
   The question the incumbent's biggest failure mode cannot answer: **does
   anything happen at this same orbital phase in OTHER binary cycles?**
   1,084 of mono-cbp's 1,647 manual-queue events are unmasked stellar eclipses,
   and a stellar eclipse recurs every single cycle. A planet transit does not.
   So the event's own cycle is excluded and the remaining cycles are stacked at
   the same phase. Channel 0 is the median stacked depth, channel 1 the log
   count of contributing cycles, because a deep stack from 40 cycles means
   something a stack from 2 does not.

   **This view is computed from the ORIGINAL cached light curve, never from the
   injected or inverted array.** That is what makes it sign-invariant: flipping
   a light curve cannot change it, so it cannot carry the inversion leak. The
   unit test asserts exactly this.

3. The scalars, at most 36
   13 core measurements, the 17-column incumbent block, 2 host-level constants,
   and 4 pair features that are only admitted if the E2 gate passes.

scalar_valid
------------
Some scalars are undefined for some events rather than merely missing. The Skye
systematics flag is computed per sector by the search and has no meaning for an
injection. Rather than impute a value and let the model treat it as real, every
event carries a validity bit vector, and each arm handles invalidity explicitly:
XGBoost routes NaN natively, logistic regression and random forests get a median
impute plus the validity bit as its own feature.
"""

import numpy as np

# ---- local view -----------------------------------------------------------
LOCAL_BINS = 201            # odd, so the event centre lands on a bin centre
LOCAL_CHANNELS = 2          # flux, valid-mask
LOCAL_HALFWIDTH_DUR = 3.0   # matches the snippet contract
LOCAL_HALFWIDTH_MIN = 0.75  # days

# ---- recurrence view ------------------------------------------------------
RECUR_BINS = 64
RECUR_CHANNELS = 2          # median stacked depth, log1p(n cycles)
RECUR_WINDOW_DUR = 1.5      # keep |dphi| * P <= 1.5 * t_dur
SIDEBAND_LO, SIDEBAND_HI = 1.5, 3.0   # continuum band, in units of t_dur
SIDEBAND_MIN_POINTS = 5
N_CYCLES_CAP = 60           # log1p cap; re-pinned from TESS-train p99 at freeze

# ---- scalars --------------------------------------------------------------
CORE_SCALARS = [
    "snr",
    "depth",
    "t_dur",
    "log_p_bin",
    "morph_coeff",
    "tmag",
    "sin_phase",
    "cos_phase",
    "detrend_fraction",     # n_detrend_detections / 21
    "skye_flag",            # valid only for label_source == real_search
    "gap_proximity",
    "local_scatter",
    "log1p_n_cycles",
    # Scalar reduction of the recurrence stack, per W3.1b. Without these the
    # [2,64] recurrence view is readable ONLY by a CNN, so every feature model
    # - and therefore the whole T0 head-to-head - is blind to recurrence.
    "rec_depth_med",
    "rec_depth_mad",
    "rec_frac_dipping",
]

HOST_SCALARS = [
    "same_star_multi_allowed",   # Kostov 2020b eq (2), constant per TIC
    "log10_pp_over_pbin",        # from P_crit, constant per TIC
]

# Admitted only if the E2 gate passes; otherwise present but masked invalid.
#
# CORRECTION OF RECORD, 2026-08-06 (pre-freeze stress test, critical finding).
# The previous set included has_pair_same_star / has_pair_cross_star derived
# from injection bookkeeping. Measured on the full shard: has_pair_same_star
# EQUALLED the label (P(label=1 | x=1) = 1.000000 over 66,320 rows; single-
# column TIC-grouped val ROC-AUC 1.0000), because pair_role was stamped on
# every recovered injection while real events carried none. No probe covered
# the pair scalars, so nothing would have caught it at the freeze.
#
# The doctrine this enforces: STAR IDENTITY IS UNOBSERVABLE FOR A REAL EVENT
# (that is ELC's entire point), so no observed feature may claim it. Same-star
# versus cross-star classification exists only in ELC audit columns. Every
# gated feature below is computed from OBSERVED flagged events, identically for
# both classes, by exporter._observed_partner -> features.nearest_pair_partner.
GATED_PAIR_SCALARS = [
    "has_observed_pair",
    "pair_dt_over_pbin",
    "pairing_depth_ratio_vs_expectation",
    # Leg 3 of the plan's anti-CNN sentence: "transit duration varies with binary
    # phase, and sameness argues against a planet". An eclipse residual repeats
    # with the SAME duration and sits at exactly 1.000; ELC's real punch pairs
    # sit at a median of 0.116. Gated with the other pair features because it is
    # only defined where a system has a second event inside the pair window.
    "pair_duration_ratio",
]

# Scalars that can be undefined per event; the bit vector covers these.
CONDITIONAL_SCALARS = ["skye_flag", "morph_coeff", "tmag"] + GATED_PAIR_SCALARS

# Injection-truth and bookkeeping columns. Stored in every shard for audit and
# diagnostics; FORBIDDEN as features. The regression suite asserts none of
# these ever appears in training_scalars, and the arm contract test (to be
# written with the arms) must assert each arm's fitted feature list equals
# training_scalars exactly.
AUDIT_COLUMNS = [
    "truth_pair_role", "truth_partner_dt", "truth_partner_ratio",
    "pair_model_version", "source_dir", "rebalance_keep", "split",
    "event_key", "label", "label_source", "inverted_lc", "incumbent_valid",
    "n_cycles_raw",
]

LABEL_SOURCES = ["real_search", "bank_injection", "elc_injection"]

# ---- scalars WITHHELD from every arm's feature matrix ---------------------
# These stay in the shards, so they remain available for diagnostics, the
# datasheet, and any later re-gating. They are simply not offered to a model.
#
# gap_proximity: probe 6 measured val ROC-AUC 0.5684 at full scale against a
# 0.55 gate, after the joint importance-sampling construction fix had already
# brought it down from 0.7042 at pilot scale. Amendment 3B.5 pre-names exactly
# this response: "if probe 6 still fails: drop gap_proximity from the training
# scalars (kept in shards), re-gate; if still failing, near-gap strata become
# reported-not-gated with the restriction in the manifest."
#
# The residual leak is a SELECTION effect, not a sampling failure: the epoch
# sampler placed injections on target 99.9 percent of the time, but an injection
# near a gap is less likely to be RECOVERED, so surviving positives are pulled
# away from gaps. That cannot be sampled away, which is why the response is to
# withhold the feature rather than to re-inject.
WITHHELD_SCALARS = {
    "gap_proximity": "probe 6 FAIL at 0.5684 (gate 0.55); Amendment 3B.5 "
                     "pre-named response applied 2026-08-04",
    # PROVISIONAL 2026-08-06, ratify at the tag. skye_flag is finite for
    # exactly the real_search rows and NaN for every injection, so
    # finite-vs-NaN identifies the class with probability 1 (measured:
    # P(label=1 | finite) = 0 over 3,672 finite rows). An XGBoost arm routes
    # NaN natively and would learn finite => negative; at deployment EVERY
    # event is real-search and skye-valid, so the model would push everything
    # toward negative. No imputation fixes a train/deploy validity mismatch,
    # so it is withheld from features (kept in shards for diagnostics and the
    # M0 funnel, where it belongs). Follows the 3B.5 withholding pattern; the
    # promotion of this from provisional to pinned is a tag-time item.
    "skye_flag": "validity-pattern leak: finite <=> real_search, measured "
                 "2026-08-06; withheld provisionally pending tag ratification",
}


def training_scalars(incumbent_cols):
    """The scalars a model may actually see: everything minus the withheld."""
    return [c for c in all_scalar_names(incumbent_cols) if c not in WITHHELD_SCALARS]


def all_scalar_names(incumbent_cols):
    return CORE_SCALARS + list(incumbent_cols) + HOST_SCALARS + GATED_PAIR_SCALARS


def local_halfwidth(t_dur):
    return max(LOCAL_HALFWIDTH_DUR * float(t_dur), LOCAL_HALFWIDTH_MIN)


def depth_floor(depth, local_scatter):
    """Normalising depth, floored so shallow noisy events do not explode.

    Dividing by a measured depth that is itself consistent with zero produces
    enormous values driven entirely by noise. The floor of three times the local
    scatter is the smallest depth that is meaningfully distinguishable, so below
    it the normalisation stops being informative and is held fixed.
    """
    return max(float(depth), 3.0 * float(local_scatter))
