"""Feature construction: one event in, one model-ready record out.

Every function here is used identically for real-search negatives and for
injected positives. Where a quantity cannot be computed for one class, it is
marked invalid rather than filled with a plausible number, because a plausible
number is indistinguishable from a real one and becomes a provenance tell.
"""

import logging

import numpy as np

from . import schema

logger = logging.getLogger("cbpvet.export.features")


def local_view(time, flux, event_time, t_dur):
    """[2, 201] view of the event: normalised flux plus a validity mask.

    The grid is fixed in units of the event's own duration, so a 0.1 d event and
    a 0.9 d event occupy the same number of bins and the FLUX SHAPE is
    duration-normalised. HONESTY CORRECTION 2026-08-06: the claim that duration
    is therefore unreadable was measured FALSE for the mask channel - at fixed
    cadence, occupied-bin density is a monotone function of t_dur (channel-1
    valid-fraction separates the classes at AUC 0.6515 on the frozen full
    shard, 0.6451 on rebalance-kept rows; measured 2026-08-06 sweep).
    Since t_dur is itself a sanctioned core scalar this adds no new information
    class, but a CNN arm receives it ungated; the disposition (disclose as
    recovery physics vs normalise occupancy) is a freeze-registry item A4.

    Channel 1 marks which bins had real data. Gaps are genuinely informative,
    since real false positives cluster near them, but they must be declared
    explicitly. Filling a gap with zero flux would read as a 100 percent transit.
    """
    half = schema.local_halfwidth(t_dur)
    edges = np.linspace(-half, half, schema.LOCAL_BINS + 1)
    rel = np.asarray(time, dtype=float) - float(event_time)
    sel = (rel >= edges[0]) & (rel <= edges[-1])
    rel, f = rel[sel], np.asarray(flux, dtype=float)[sel]

    view = np.zeros((schema.LOCAL_CHANNELS, schema.LOCAL_BINS), dtype=np.float32)
    if len(rel) == 0:
        return view, 0.0

    idx = np.clip(np.digitize(rel, edges) - 1, 0, schema.LOCAL_BINS - 1)
    # Out-of-transit scatter, measured outside +/- one duration from centre.
    oot = f[np.abs(rel) > float(t_dur)]
    scatter = float(1.4826 * np.median(np.abs(oot - np.median(oot)))) if len(oot) > 5 else float(np.std(f))
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = 1e-6

    for b in range(schema.LOCAL_BINS):
        m = idx == b
        if m.any():
            view[0, b] = np.median(f[m]) - 1.0
            view[1, b] = 1.0
    return view, scatter


def recurrence_view(time_native, flux_native, phase, event_time, event_phase,
                    t_dur, period, depth, local_scatter):
    """[2, 64] stack of OTHER cycles at the same orbital phase.

    This is the view that separates an unmasked stellar eclipse, which repeats
    every cycle, from a planet transit, which does not. It is the single most
    valuable thing the incumbent's per-event model comparison cannot see, since
    that comparison only ever looks at one window at a time.

    Critically, ``flux_native`` must be the ORIGINAL cached flux, never the
    injected or inverted array. Two consequences follow, both intended:
    the view is exactly sign-invariant, so it cannot carry the inversion leak;
    and for an injected positive it correctly shows no recurrence, which is what
    a real planet also shows.

    Per-cycle continuum: each contributing cycle is divided by the median of its
    own sideband cadences before stacking. Without that, a cycle sitting on a
    residual detrending slope contributes its slope to the stack and washes out
    a genuine repeat.
    """
    view = np.zeros((schema.RECUR_CHANNELS, schema.RECUR_BINS), dtype=np.float32)
    t = np.asarray(time_native, dtype=float)
    f = np.asarray(flux_native, dtype=float)
    ph = np.asarray(phase, dtype=float)
    if len(t) == 0 or not np.isfinite(period) or period <= 0:
        return view, 0

    # Signed phase distance from the event, wrapped to [-0.5, 0.5).
    dphi = ((ph - float(event_phase) + 0.5) % 1.0) - 0.5
    window = schema.RECUR_WINDOW_DUR * float(t_dur) / float(period)
    if window <= 0:
        return view, 0

    cycle = np.floor((t - (float(event_time) - float(event_phase) * period)) / period)
    event_cycle = np.floor(0.0)  # by construction the event sits at cycle offset 0
    # Identify the event's own cycle from its time, then exclude it entirely.
    event_cycle = np.floor((float(event_time) - (float(event_time) - float(event_phase) * period)) / period)

    in_window = np.abs(dphi) <= window
    other_cycle = cycle != event_cycle
    sideband = (np.abs(dphi) * period > schema.SIDEBAND_LO * t_dur) & \
               (np.abs(dphi) * period <= schema.SIDEBAND_HI * t_dur)

    denom = schema.depth_floor(depth, local_scatter)
    edges = np.linspace(-window, window, schema.RECUR_BINS + 1)
    per_bin = [[] for _ in range(schema.RECUR_BINS)]
    contributing = set()

    for c in np.unique(cycle[other_cycle & (in_window | sideband)]):
        m_cycle = cycle == c
        sb = f[m_cycle & sideband]
        continuum = np.median(sb) if len(sb) >= schema.SIDEBAND_MIN_POINTS else np.nan
        m_win = m_cycle & in_window
        if not m_win.any():
            continue
        if not np.isfinite(continuum):
            continuum = np.median(f[m_win])
        if not np.isfinite(continuum) or continuum == 0:
            continue
        fc = f[m_win] / continuum
        idx = np.clip(np.digitize(dphi[m_win], edges) - 1, 0, schema.RECUR_BINS - 1)
        for b, val in zip(idx, fc):
            per_bin[b].append((c, val))
        contributing.add(float(c))

    for b in range(schema.RECUR_BINS):
        if not per_bin[b]:
            continue
        vals = np.array([v for _, v in per_bin[b]])
        cycles_here = len({c for c, _ in per_bin[b]})
        view[0, b] = np.median(vals - 1.0) / denom
        view[1, b] = np.log1p(min(cycles_here, schema.N_CYCLES_CAP))

    # ---- scalar reduction of the SAME stack ------------------------------
    # W3.1b names these as the E1 fallback: "4 scalars from the same stack:
    # n_cycles_available, rec_depth_med, rec_depth_mad (1.4826*MAD),
    # rec_frac_dipping". Only the first was implemented.
    #
    # Why this matters far beyond the E1 branch. The recurrence view is a
    # [2, 64] image, and only a CNN can eat an image. Every feature model -
    # logistic regression, random forest, XGBoost, and therefore the entire T0
    # head-to-head - is blind to it. If the CNN is deprioritised to fall, which
    # is exactly what open decision D-A proposes, the headline model would carry
    # NO recurrence information at all, and recurrence is the leg the "not a
    # cat-or-dog CNN" argument actually rests on.
    #
    # These scalars are computed from the native, un-inverted cached flux like
    # the view itself, so they inherit its sign-invariance.
    per_cycle_depth = {}
    for b in range(schema.RECUR_BINS):
        for c, v in per_bin[b]:
            d = (1.0 - v) / denom
            if c not in per_cycle_depth or d > per_cycle_depth[c]:
                per_cycle_depth[c] = d
    scalars = {"rec_depth_med": 0.0, "rec_depth_mad": 0.0, "rec_frac_dipping": 0.0}
    if per_cycle_depth:
        d = np.array(list(per_cycle_depth.values()), dtype=float)
        med = float(np.median(d))
        scalars["rec_depth_med"] = med
        scalars["rec_depth_mad"] = float(1.4826 * np.median(np.abs(d - med)))
        # A cycle "dips" if its stacked depth clears three times the local
        # scatter, expressed in the same normalised units as the view.
        thresh = 3.0 * float(local_scatter) / denom
        scalars["rec_frac_dipping"] = float(np.mean(d > thresh))

    return view, len(contributing), scalars


def gap_proximity(time_full, event_time, gap_threshold=0.5):
    """Days to the nearest sector edge or intra-sector gap.

    Measured on the FULL cached time array, never on the snippet: a snippet is
    by construction centred on its event, so gap distance measured inside it is
    capped by the snippet's own half-width and would be a different quantity for
    different durations.
    """
    t = np.sort(np.asarray(time_full, dtype=float))
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return np.nan
    edges = [t[0], t[-1]]
    dt = np.diff(t)
    for j in np.where(dt > gap_threshold)[0]:
        edges.extend([t[j], t[j + 1]])
    return float(np.min(np.abs(np.asarray(edges) - float(event_time))))


def same_star_multi_allowed(p_planet, p_bin, mass_ratio):
    """Kostov 2020b equation (2): can this system show multiple same-star transits?

    P_CBP > P_bin * ((M1/M2) + 1)^3. A host-level constant, so it is identical
    for every event of a given TIC, which the unit test asserts.
    """
    if not np.isfinite(mass_ratio) or mass_ratio <= 0:
        return 0.0
    return float(p_planet > p_bin * (1.0 / mass_ratio + 1.0) ** 3)


def pairing_depth_ratio_vs_expectation(observed_ratio, expected_ratio):
    """How far the observed pair depth ratio sits from the host's eclipse ratio.

    The two stars' eclipse depth ratio predicts how deep a crossing of each
    should look. A pair whose depths disagree wildly with that prediction is
    more likely to be two unrelated events than one conjunction.
    """
    if not np.isfinite(observed_ratio) or not np.isfinite(expected_ratio) or expected_ratio <= 0:
        return np.nan
    return float(np.log10(max(observed_ratio, 1e-6) / expected_ratio))


def nearest_pair_partner(event_time, event_duration, other_times, other_durations,
                         p_bin, other_depths=None):
    """Leg 3: the duration ratio against the nearest event inside the pair window.

    THE PHYSICS. A planet crossing a binary crosses two DIFFERENT stars, on
    different chords, while those stars move. The two transits of one conjunction
    therefore have different durations. An unmasked stellar eclipse repeats with
    the SAME duration every time, so its ratio sits at exactly 1.000. That
    contrast is the plan's third leg: "transit duration varies with binary phase,
    and sameness argues against a planet".

    MEASURED on 33 real ELC punch pairs, 2026-08-04:
        partner/lead duration ratio  median 0.116, p10 0.039, p90 0.339
        97 percent of pairs differ by more than a factor of 2
        0 percent are within 10 percent of each other

    AND IT IS NOT A TEMPLATE ARTIFACT. Re-running ELC with EQUAL-SIZED stars
    (ratrad = 1.0) still gives 0.349, not 1.0, because the two crossings have
    different impact parameters (0.264 against 0.359). The template's radius
    ratio amplifies the effect; it does not create it.

    Computed IDENTICALLY for both classes: for any event, find the nearest other
    flagged event of the same system inside the per-system pair window and take
    the duration ratio, smaller over larger, so the value is in (0, 1]. Real
    negatives get theirs from the other real TCEs of that system; injected
    positives get theirs from the other flagged events of the same test. Neither
    uses injection truth, so the feature is computable at deployment.

    Returns (ratio, dt_days, n_candidates, partner_depth); ratio is NaN when the
    system has no other event inside the window, which must be MASKED rather
    than imputed. partner_depth is the nearest in-window event's measured depth
    (NaN when absent or not supplied), so the OBSERVED pair depth ratio can be
    built from flagged events alone -- the injection-truth partner_ratio must
    never be a feature (2026-08-06: the truth-derived has_pair flags were found
    to equal the label verbatim, AUC 1.0000).
    """
    other_times = np.asarray(other_times, dtype=float)
    other_durations = np.asarray(other_durations, dtype=float)
    if other_times.size == 0 or not np.isfinite(event_duration) or event_duration <= 0:
        return np.nan, np.nan, 0, np.nan

    lo = max(MIN_PAIR_WINDOW_DAYS, float(event_duration))
    hi = 0.5 * float(p_bin)
    if hi <= lo:
        return np.nan, np.nan, 0, np.nan

    dt = np.abs(other_times - float(event_time))
    inside = (dt >= lo) & (dt <= hi) & np.isfinite(other_durations) & (other_durations > 0)
    if not inside.any():
        return np.nan, np.nan, 0, np.nan

    j = int(np.argmin(np.where(inside, dt, np.inf)))
    d_other = float(other_durations[j])
    d_self = float(event_duration)
    ratio = min(d_self, d_other) / max(d_self, d_other)
    dep = np.nan
    if other_depths is not None:
        od = np.asarray(other_depths, dtype=float)
        if od.size == other_times.size and np.isfinite(od[j]):
            dep = float(od[j])
    return float(ratio), float(dt[j]), int(inside.sum()), dep


MIN_PAIR_WINDOW_DAYS = 0.15


def dedup_groups(event_times, tol_days=0.0209):
    """Group events of one (tic, sector) that are the same physical event.

    The search reports peaks from 21 detrending windows and already picks the
    best per group, but events one cadence apart can still survive as separate
    rows. The canonical row is the highest-SNR member; multiplicity is kept as
    METADATA only and is unit-tested out of every arm's feature matrix, since a
    duplicated event is an artefact of the search rather than a property of the
    astrophysics.
    """
    order = np.argsort(event_times)
    groups, current = [], [order[0]] if len(order) else []
    for prev, nxt in zip(order[:-1], order[1:]):
        if abs(event_times[nxt] - event_times[prev]) <= tol_days:
            current.append(nxt)
        else:
            groups.append(current)
            current = [nxt]
    if len(current):
        groups.append(current)
    return groups
