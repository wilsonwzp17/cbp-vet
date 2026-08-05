"""Binary geometry helpers: eccentricity from eclipse timing, stability radius.

Two quantities are needed all over the data engine, and both are derived from
columns the TEBC catalogue already carries, so no external fit is required.

Eccentricity from the eclipse pattern
-------------------------------------
For an eclipsing binary the phase offset of the secondary eclipse from 0.5
measures ``e cos w``, and the ratio of eclipse durations measures ``e sin w``::

    e cos w = (pi / 2) * (phi_sec - 0.5)          Winn 2010
    e sin w = (w_sec - w_pri) / (w_sec + w_pri)

**Correction of record (VERIFIED-FACTS section 3):** the plan previously carried
``(pi/2)(2 phi_sec - 1)``, which is exactly twice too large.

A second trap, also from VERIFIED-FACTS section 3: use ``(sec_pos - prim_pos)
mod 1``, never the raw ``sec_pos``. At least one catalogue row has
``prim_pos = 1.0002`` against ``sec_pos = 5.3e-05``, which raw arithmetic reads
as two coincident eclipses.

Critical stability radius
-------------------------
A circumbinary planet cannot orbit arbitrarily close to its binary; inside a
critical radius the orbit is unstable. Holman and Wiegert 1999 give a fit for
that radius in terms of the binary's mass ratio and eccentricity. It matters
here because injected planet periods are drawn relative to it: nine of ten
Kepler circumbinary systems orbit within a factor of two of this boundary, so
drawing periods relative to it puts the synthetic population where the real one
actually sits.

``mu = M2 / (M1 + M2)`` is pinned at 0.3, recorded in the manifest. The
catalogue carries no masses, and the fit's dependence on mu is weak across the
plausible range.
"""

import numpy as np

MU_PINNED = 0.3   # pinned per Execution-Readiness B5; manifest field


def eclipse_eccentricity(prim_pos, sec_pos, prim_width, sec_width):
    """Return (e_cos_w, e_sin_w, e) from eclipse positions and widths."""
    if any(v is None or not np.isfinite(v) for v in (prim_pos, sec_pos)):
        return np.nan, np.nan, np.nan
    phi_sec = (float(sec_pos) - float(prim_pos)) % 1.0
    e_cos_w = (np.pi / 2.0) * (phi_sec - 0.5)
    if (prim_width is None or sec_width is None
            or not np.isfinite(prim_width) or not np.isfinite(sec_width)
            or (sec_width + prim_width) <= 0):
        e_sin_w = 0.0
    else:
        e_sin_w = (float(sec_width) - float(prim_width)) / (float(sec_width) + float(prim_width))
    e = float(np.hypot(e_cos_w, e_sin_w))
    return float(e_cos_w), float(e_sin_w), min(e, 0.95)


def a_crit_over_a_bin(e, mu=MU_PINNED):
    """Holman and Wiegert 1999 critical semi-major axis, in units of a_bin."""
    e = float(np.clip(e, 0.0, 0.8))
    return (1.60 + 5.10 * e - 2.22 * e ** 2 + 4.12 * mu
            - 4.27 * e * mu - 5.09 * mu ** 2 + 4.61 * e ** 2 * mu ** 2)


def p_crit(p_bin, e, mu=MU_PINNED):
    """Shortest stable circumbinary period, in days.

    Kepler's third law makes the mass scale cancel: the ratio of periods is the
    ratio of semi-major axes to the three-halves power, and both orbits share the
    same total mass.
    """
    return float(p_bin) * a_crit_over_a_bin(e, mu) ** 1.5


def draw_planet_period(rng, p_bin, e, mu=MU_PINNED, tail_fraction=0.20, size=1):
    """Draw a planet period above the stability boundary.

    Log-uniform over [P_crit, 3 P_crit], with a ``tail_fraction`` of draws
    extended to 10 P_crit. The concentration near the boundary reproduces the
    observed population: nine of ten Kepler circumbinary systems sit within a
    factor of two of it (Kostov 2020b).
    """
    pc = p_crit(p_bin, e, mu)
    u = rng.random(size)
    hi = np.where(rng.random(size) < tail_fraction, np.log10(10.0), np.log10(3.0))
    return pc * 10.0 ** (u * hi)
