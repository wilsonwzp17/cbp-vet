"""Probe-1 remedy: match the negative class's inversion rate to the positives'.

Why this is needed
------------------
The campaign flips each test's light curve with probability 0.5, so the POSITIVE
class is 50/50 inverted by construction. The NEGATIVE class is not, and cannot
be, because the two arms do not yield negatives at the same rate.

Measured on the 2k pilot, 2026-08-04:

    inverted arm   1.136 negatives per test
    native   arm   1.698 negatives per test      (1.50x more)

The mechanism is the inversion trick working exactly as designed. On the native
arm every real dip in the light curve is still dip-shaped, so the search flags
it and it is harvested as a negative. On the inverted arm those same real dips
became bumps and are never flagged. So negatives are structurally enriched in
native curves: 0.3942 inverted against the positives' 0.4956, a gap of 0.1014
at 6.5 standard errors, well past probe 1's 0.05 gate.

This is not a bug to be fixed in generation. Making the arms yield equally would
mean either discarding the real-dip negatives (which are the most valuable
negatives we have, being genuine in-the-wild false positives) or suppressing the
inversion trick (which is what protects the native arm's labels).

So the remedy is the pre-named escalation, "rebalance mix": subsample the
over-represented arm WITHIN the negative class until its inversion rate matches
the positive class. Applied at export, seeded, and reported in the datasheet
with both the before and after numbers, never silently.
"""

import numpy as np


def rebalance_negatives(events, seed=20260804, target=None, strata=None):
    """Subsample negatives so their inversion rate matches the positives'.

    Args:
        events: DataFrame with ``label`` and ``inverted_lc`` columns.
        seed: pinned; the dropped-row set is part of the frozen dataset.
        target: desired negative inversion rate. Defaults to the positive
            class's own rate, which is what probe 1 compares against.
        strata: optional list of columns to rebalance within, so the match holds
            inside each stratum and not merely on average.

    Returns:
        (rebalanced_events, report dict)
    """
    rng = np.random.default_rng(seed)
    pos = events[events.label == 1]
    neg = events[events.label == 0]
    if not len(pos) or not len(neg):
        return events, {"applied": False, "reason": "one class empty"}

    tgt = float(pos.inverted_lc.mean()) if target is None else float(target)

    def _one(block):
        inv = block[block.inverted_lc == 1]
        nat = block[block.inverted_lc == 0]
        if not len(inv) or not len(nat):
            return block
        # Keep whichever arm is scarce relative to the target, and thin the other.
        n_inv_max = int(np.floor(len(nat) * tgt / (1 - tgt))) if tgt < 1 else len(inv)
        n_nat_max = int(np.floor(len(inv) * (1 - tgt) / tgt)) if tgt > 0 else len(nat)
        if len(inv) > n_inv_max:
            inv = inv.iloc[rng.permutation(len(inv))[:n_inv_max]]
        elif len(nat) > n_nat_max:
            nat = nat.iloc[rng.permutation(len(nat))[:n_nat_max]]
        import pandas as pd
        return pd.concat([inv, nat])

    import pandas as pd
    if strata:
        kept = pd.concat([_one(g) for _, g in neg.groupby(strata, dropna=False)])
    else:
        kept = _one(neg)

    out = pd.concat([pos, kept]).sort_index()
    report = {
        "applied": True,
        "target_rate": tgt,
        "positives_inversion_rate": float(pos.inverted_lc.mean()),
        "negatives_before": int(len(neg)),
        "negatives_after": int(len(kept)),
        "negatives_dropped": int(len(neg) - len(kept)),
        "neg_inversion_before": float(neg.inverted_lc.mean()),
        "neg_inversion_after": float(kept.inverted_lc.mean()),
        "probe1_gap_before": float(abs(pos.inverted_lc.mean() - neg.inverted_lc.mean())),
        "probe1_gap_after": float(abs(pos.inverted_lc.mean() - kept.inverted_lc.mean())),
        "seed": seed,
        "strata": strata,
    }
    return out, report
