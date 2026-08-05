"""Pair mechanics for circumbinary transit injections.

The "1-2 punch"
---------------
When a circumbinary planet passes in front of its binary, the two stars are
themselves moving. Within a single conjunction the planet can therefore cross
star 1 and then, hours to days later, cross star 2. Two transits, one
conjunction, different depths, close together. Kostov et al. 2020b call this
signature the punch and use it to pin the planet's period from well under one
orbit. It is the single most distinctive thing a circumbinary planet does that
an eclipse or an artefact does not, so the training data has to contain it.

This module pins the two numbers that define an injected pair: how far apart in
time the two transits fall, and how their depths relate.

The spacing, and why the old window was wrong
---------------------------------------------
The plan previously used a fixed 1 to 4 day window. Checked against the 12
observed spacings in Kostov 2020b Table 1, that window captures **2 of 12, and
0 of Kepler-16's 5**, while Kepler-16 sits in our frozen real-planet test set.
The model would have been graded on pairs its own labeller could not see.

The fix is to work in the dimensionless ratio dt / P_bin rather than in days.
Across the four systems that ratio runs 0.093 to 0.383 with median 0.200, and
Kepler-16's five conjunctions sit at 0.199 to 0.205, essentially identical,
because the geometry repeats per system. A window of [0.05, 0.5] x P_bin
captures **12 of 12**.

The one outlier, Kepler-1647 at 0.383, is the system with by far the slowest
planet, P_p / P_bin = 98 against 5.6 to 10.4 for the others. That is correct
physics rather than noise: the slower the planet crosses, the further the stars
travel while it does. So the ratio is sampled **conditioned on P_p / P_bin**,
which is what ``sample_dt_over_pbin`` does with a Gaussian kernel in
log10(P_p / P_bin).

Formula of record: Amendment-1 section 3B.6 and Week-3 Jul-30, "per-host
dt_pred distribution = P_bin x (dt/P_bin from Kostov 2020b Table 1),
conditioned on P_p/P_bin per VERIFIED-FACTS section 7". This supersedes the
earlier analytic form in Execution-Readiness B1, which used a cube-root
expression with a calibrated constant C; both were fitted to the same 12
spacings, and the empirical resample is used because it needs no functional
assumption.

For ELC-generated positives none of this is used: ``iwriteeclipse`` writes the
exact transit time and the identity of the star crossed. The sampler is needed
only for the synthetic bank, and the window only for labelling REAL events.
"""

import numpy as np

# Kostov et al. 2020b Table 1: the 14 observed conjunctions, 12 with both
# transit times. Transcribed in VERIFIED-FACTS section 7.
# (system, P_bin days, P_planet days, spacing days)
KOSTOV_TABLE1 = [
    ("Kepler-16",   41.079, 229.0, 8.174),
    ("Kepler-16",   41.079, 229.0, 8.400),
    ("Kepler-16",   41.079, 229.0, 8.200),
    ("Kepler-16",   41.079, 229.0, 8.340),
    ("Kepler-16",   41.079, 229.0, 8.220),
    ("Kepler-34",   27.796, 289.0, 5.558),
    ("Kepler-34",   27.796, 289.0, 7.377),
    ("Kepler-34",   27.796, 289.0, 3.270),
    ("Kepler-34",   27.796, 289.0, 4.659),
    ("Kepler-35",   20.734, 131.0, 4.440),
    ("Kepler-35",   20.734, 131.0, 1.920),
    ("Kepler-1647", 11.259, 1108.0, 4.312),
]

# Derived once: the dimensionless spacing and the period ratio it is conditioned on.
DT_OVER_PBIN = np.array([sp / pb for _, pb, _, sp in KOSTOV_TABLE1])
LOG_PERIOD_RATIO = np.array([np.log10(pp / pb) for _, pb, pp, _ in KOSTOV_TABLE1])

# Kernel bandwidth in dex for the conditional resample. 0.3 dex keeps the
# Kepler-16/34/35 cluster (log ratio 0.75 to 1.02) together and lets the
# Kepler-1647 point (log ratio 1.99) dominate only for genuinely slow planets.
KERNEL_BANDWIDTH_DEX = 0.3
# Multiplicative jitter on the resampled ratio, so 16k draws are not 12 values.
RATIO_JITTER_SIGMA = 0.05

# Label window, per Amendment 3B.6: [max(0.15 d, one transit duration), 0.5 P_bin].
MIN_WINDOW_DAYS = 0.15
MAX_WINDOW_PBIN_FRACTION = 0.5

# Depth-ratio model. The geometric factor g = observed_ratio / host_depth_ratio,
# REFITTED 2026-08-04 evening on the FULL batch: 1,399 real pairs across 127
# hosts, 47x the sample of the first fit. Quantiles of g:
#   p5 0.0240  p10 0.0915  p25 0.3843  p50 0.6387  p75 0.7949  p90 0.8981
#
# Two guards were needed before the fit was trustworthy, both found by looking
# at the raw numbers rather than at the plan:
#   (a) grazing NON-transits (|impact| >= 1, ingress == egress) excluded;
#   (b) MAX_PLANET_DEPTH: 63 of 1,725 pairs had a partner "depth" of 0.13-0.14
#       while the lead was 0.0003-0.0007. Those are STELLAR ECLIPSES caught by
#       min(flux) inside the transit window, not planet transits. Injected
#       planet depths cannot exceed ~2e-2.
#
# The first fit, on 30 pairs from 3 hosts, OVERSHOT: the census then measured
# ELC 0.3526 against the bank's 0.1830, 92.7 percent off. Superseded here.
#
# The top quantile is large (a grazing LEAD gives a small d1, so d2/d1 is big).
# That is real, and DEPTH_RATIO_CLIP bounds the output, so the clip binds for a
# small measured fraction reported in the fit JSON rather than being hidden.
# The old model, lognormal(0, 0.15) clipped [0.5, 2.0], could not reach below
# 0.47 and so made every synthetic partner far easier to detect than reality.
ELC_GEOMETRIC_FACTOR = np.array([
    0.00400, 0.01315, 0.04099, 0.07419, 0.10334, 0.16359, 0.20757, 0.27405,
    0.31708, 0.36809, 0.39552, 0.42518, 0.46267, 0.49589, 0.51653, 0.54358,
    0.56443, 0.59403, 0.61384, 0.63110, 0.64975, 0.66376, 0.68154, 0.69921,
    0.71971, 0.73787, 0.74996, 0.76214, 0.77678, 0.78662, 0.80130, 0.81813,
    0.83766, 0.85812, 0.87676, 0.89130, 0.90416, 0.94344, 1.32256, 11.23515,
])
DEPTH_RATIO_JITTER_SIGMA = 0.20
DEPTH_RATIO_CLIP = (1e-3, 2.0)
# Superseded, kept so the change is auditable:
DEPTH_RATIO_LOGNORM_SIGMA_SUPERSEDED = 0.15
DEPTH_RATIO_CLIP_SUPERSEDED = (0.5, 2.0)


# Partner/lead DURATION ratio, measured on 33 real ELC punch pairs (grazing
# non-transits excluded), 2026-08-04. 25 quantiles of the empirical distribution.
#
# WHY THIS EXISTS. The bank previously injected the partner as
# ``flux_model * partner_ratio``: the SAME array, scaled in depth only, so every
# bank pair had a duration ratio of exactly 1.000. ELC's two crossings of one
# conjunction differ by a median factor of 8.6, because the planet crosses two
# stars of different size at different impact parameters.
#
# Left unfixed, adding a pair-duration feature would let probe 2 separate bank
# from ELC positives on a single column. Probe 2 is a FREEZE GATE, so that is
# not cosmetic: it would block the freeze.
#
# The signal is real geometry, not a template artifact. Re-running ELC with
# EQUAL-SIZED stars (ratrad = 1.0) still gives 0.349, not 1.0, because the two
# crossings have different impact parameters (b1 0.264 against b2 0.359). The
# template's radius ratio amplifies the effect from 0.349 to 0.116; it does not
# create it. An eclipse residual, by contrast, repeats with the SAME duration
# and sits at exactly 1.000, which is what makes the feature discriminating.
#
# PROVISIONAL: 33 pairs from three hosts. Refit on the full 150-host batch.
ELC_DURATION_RATIO_Q = np.array([
    0.01798, 0.02474, 0.03573, 0.05006, 0.05469, 0.06354, 0.06819, 0.07885,
    0.08959, 0.09110, 0.09910, 0.10976, 0.11563, 0.13493, 0.15785, 0.17445,
    0.20756, 0.22383, 0.25460, 0.26722, 0.30855, 0.32602, 0.36188, 0.43093,
    0.83925,
])
DURATION_RATIO_JITTER_SIGMA = 0.15
DURATION_RATIO_CLIP = (0.02, 1.0)


def pair_duration_ratio(rng, size=1):
    """Partner/lead transit-duration ratio, resampled from ELC geometry.

    Capped at 1.0: the partner crossing of the smaller, fainter star is the
    shorter one in every ELC pair measured, and letting the bank draw above 1
    would invent a configuration the physics does not produce here.
    """
    q = ELC_DURATION_RATIO_Q
    idx = rng.integers(0, len(q), size=size)
    vals = q[idx] * np.exp(rng.normal(0.0, DURATION_RATIO_JITTER_SIGMA, size=size))
    return np.clip(vals, *DURATION_RATIO_CLIP)


def label_window(p_bin, duration_days):
    """The per-system window in which two events may be called a pair.

    Args:
        p_bin: binary orbital period in days.
        duration_days: transit duration in days.

    Returns:
        (lo, hi) in days. ``hi`` can fall below ``lo`` for very short-period
        binaries, in which case no pair is representable for that system and
        callers must skip it rather than clip into a degenerate window.
    """
    lo = max(MIN_WINDOW_DAYS, float(duration_days))
    hi = MAX_WINDOW_PBIN_FRACTION * float(p_bin)
    return lo, hi


def sample_dt_over_pbin(rng, log_period_ratio, size=1):
    """Resample dt / P_bin from Table 1, conditioned on log10(P_p / P_bin).

    Each of the 12 observed ratios is weighted by a Gaussian kernel in
    log10(P_p / P_bin) centred on the requested value, so a system whose planet
    is as slow as Kepler-1647's draws mostly from Kepler-1647's spacing, and a
    typical system draws from the tight 0.09 to 0.27 cluster. A small
    multiplicative jitter keeps 16,384 draws from collapsing onto 12 values.
    """
    w = np.exp(-0.5 * ((LOG_PERIOD_RATIO - log_period_ratio) / KERNEL_BANDWIDTH_DEX) ** 2)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        w = np.ones_like(w)
        total = w.sum()
    idx = rng.choice(len(DT_OVER_PBIN), size=size, p=w / total)
    ratios = DT_OVER_PBIN[idx] * np.exp(rng.normal(0.0, RATIO_JITTER_SIGMA, size=size))
    return ratios


def pair_spacing(rng, p_bin, p_planet, duration_days, size=1):
    """Signed spacing in days between the two transits of one conjunction.

    Returns an array of signed dt. The magnitude is P_bin times a conditionally
    resampled dt/P_bin, clipped into the per-system label window so that every
    injected pair is representable by the labeller that will have to find it.
    Sign is plus or minus with equal probability: either star may be crossed
    first, and the classifier must not learn an ordering convention.

    Returns None if the system admits no valid window at this duration.
    """
    lo, hi = label_window(p_bin, duration_days)
    if hi <= lo:
        return None
    ratios = sample_dt_over_pbin(rng, np.log10(p_planet / p_bin), size=size)
    dt = np.clip(ratios * p_bin, lo, hi)
    sign = rng.choice([-1.0, 1.0], size=size)
    return dt * sign


def pair_depth_ratio(rng, host_depth_ratio, size=1):
    """Depth of the partner transit relative to the lead transit.

    The anchor is the host's own eclipse depth ratio, because for a detached
    eclipsing binary that ratio measures the two stars' SURFACE BRIGHTNESS ratio
    (both eclipses block the same area, the smaller star's disc). That is the
    physical part and it is right.

    What was wrong was the scatter. The first version used
    ``lognormal(0, 0.15)`` clipped to [0.5, 2.0], giving p10/p50/p90 of
    0.744 / 0.952 / 1.096. Measured against ELC's real dynamics on 30 pairs,
    the truth is 0.007 / 0.222 / 0.817: real partner depths span three orders
    of magnitude, because the planet's chord across the SMALLER star is often
    grazing while the chord across the larger one is not. The old model could
    not even reach below 0.47, so every synthetic partner was easy to detect.

    Correction of record, 2026-08-04, per Amendment 3B.6's pre-named response to
    a calibration-census disagreement on rate/spacing/RATIO.

    The correction resamples the purely geometric factor
    ``g = observed_ratio / host_depth_ratio`` empirically from the ELC
    measurements, with mild log-space jitter, rather than imposing a parametric
    shape. A lognormal fit to the same data was tried and rejected: it
    reproduced the median but overshot the upper tail badly (p90 of 1.93 against
    a measured 0.82), because the real distribution is bounded while a lognormal
    is not. Empirical resampling is also what the spacing model does, and that
    one validated against ELC to within 14.8 percent.

    PROVISIONAL: fitted on 30 pairs from three hosts. Refit when the full ELC
    batch runs; the sample is the limitation, not the method.
    """
    g = ELC_GEOMETRIC_FACTOR
    idx = rng.integers(0, len(g), size=size)
    jitter = np.exp(rng.normal(0.0, DEPTH_RATIO_JITTER_SIGMA, size=size))
    ratio = float(host_depth_ratio) * g[idx] * jitter
    return np.clip(ratio, *DEPTH_RATIO_CLIP)


def config_pins():
    """The pinned constants, for the campaign config and the freeze manifest."""
    return {
        "dt_model": "empirical resample of Kostov 2020b Table 1 dt/P_bin, "
                    "conditioned on log10(P_p/P_bin) with a Gaussian kernel",
        "kernel_bandwidth_dex": KERNEL_BANDWIDTH_DEX,
        "ratio_jitter_sigma": RATIO_JITTER_SIGMA,
        "n_table1_spacings": len(DT_OVER_PBIN),
        "dt_over_pbin_range": [float(DT_OVER_PBIN.min()), float(DT_OVER_PBIN.max())],
        "dt_over_pbin_median": float(np.median(DT_OVER_PBIN)),
        "label_window": f"[max({MIN_WINDOW_DAYS} d, duration), {MAX_WINDOW_PBIN_FRACTION} x P_bin]",
        "depth_ratio_model": "empirical resample of the ELC-measured geometric "
                             "factor g = ratio / host_depth_ratio, 30 pairs",
        "depth_ratio_jitter_sigma": DEPTH_RATIO_JITTER_SIGMA,
        "depth_ratio_clip": list(DEPTH_RATIO_CLIP),
        "depth_ratio_superseded": {
            "model": "lognormal(0, 0.15) clipped [0.5, 2.0]",
            "why": "measured p10/p50/p90 0.744/0.952/1.096 against ELC's "
                   "0.007/0.222/0.817; could not reach below 0.47",
            "authority": "Amendment 3B.6 pre-named response to a census "
                         "disagreement on rate/spacing/ratio",
        },
        "supersedes": "Execution-Readiness B1 analytic dt form with calibrated C",
    }
