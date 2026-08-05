"""The frozen train/validation/test split, assigned by SYSTEM and never by event.

Why grouping by TIC is not optional
------------------------------------
Events from one eclipsing binary are not independent. They share the same star,
the same systematics, the same detrending behaviour, the same crowding. Split
rows at random and the same system lands on both sides, so a model can score
well by recognising the star rather than the transit. Every number the benchmark
reports would then be inflated for a reason that has nothing to do with vetting.

So the split is assigned at TIC level and every event inherits its system's
side. A pytest asserts the three TIC sets are pairwise disjoint and that no
event's split disagrees with its TIC's.

Stratification
--------------
A purely random TIC split would, by luck, put most of the well-observed systems
on one side. Three things are balanced:

* **coverage tier** (fewer than 3, 3 to 9, 10 or more observed binary cycles),
  because recurrence is the distinctive feature and it needs cycles to exist;
* **P_bin tercile**, because binary period sets the eclipse cadence and the pair
  window;
* **Tmag tercile**, because brightness sets the noise floor.

Assignment is greedy: TICs are sorted by event count descending, and each is
placed on whichever side currently minimises the weighted squared deviation from
the target proportions. Taking the biggest systems first matters, because one
system with a thousand events can unbalance a split that looks fine by TIC count.

The 10-plus tier is deliberately over-allocated to test, 60/15/25 instead of
70/15/15. That tier is where the recurrence view has the most to say and where
the headline contrast is measured, and its minimum detectable effect is set by
the number of distinct TICs on the test side. Buying test power there is worth
more than the training rows it costs.

The excluded TICs, which carry the known real planets, are pre-assigned to test
before anything else runs.
"""

import hashlib
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cbpvet.bench.split")

TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGETS_HIGH_COVERAGE = {"train": 0.60, "val": 0.15, "test": 0.25}
COVERAGE_EDGES = (3, 10)          # <3, 3-9, 10+
SPLIT_SEED = 20260804


def coverage_tier(n_cycles):
    if n_cycles < COVERAGE_EDGES[0]:
        return "lt3"
    if n_cycles < COVERAGE_EDGES[1]:
        return "3to9"
    return "10plus"


def build_tic_table(events, catalogue_raw):
    """One row per TIC: its stratum key and how many events it contributes."""
    g = events.groupby(events.tic.astype(int))
    tab = pd.DataFrame({
        "n_events": g.size(),
        "n_positive": g.label.sum(),
        "n_cycles": g.n_cycles_raw.max() if "n_cycles_raw" in events.columns else 0,
    })
    raw = catalogue_raw.drop_duplicates("tess_id").set_index("tess_id")
    tab["p_bin"] = raw.reindex(tab.index).period
    tab["tmag"] = raw.reindex(tab.index).Tmag
    tab["coverage"] = tab.n_cycles.apply(coverage_tier)
    # Terciles computed over the TICs actually present, so the strata are
    # populated rather than nominal.
    for col in ("p_bin", "tmag"):
        vals = tab[col]
        ok = np.isfinite(vals)
        tab[f"{col}_tercile"] = "unknown"
        if ok.sum() >= 3:
            tab.loc[ok, f"{col}_tercile"] = pd.qcut(
                vals[ok], 3, labels=["lo", "mid", "hi"], duplicates="drop").astype(str)
    tab["stratum"] = (tab.coverage + "|" + tab.p_bin_tercile + "|" + tab.tmag_tercile)
    return tab


def assign(tic_table, excluded_tics=(), seed=SPLIT_SEED):
    """Greedy stratified assignment, biggest systems first."""
    rng = np.random.default_rng(seed)
    assignment = {}

    for tic in excluded_tics:
        if tic in tic_table.index:
            assignment[int(tic)] = "test"     # pre-assigned, never negotiable

    # Track event mass per (stratum, split) so balance is by EVENTS, not TICs.
    mass = {}
    for stratum, block in tic_table.groupby("stratum"):
        block = block.drop(index=[t for t in assignment if t in block.index], errors="ignore")
        block = block.sort_values("n_events", ascending=False)
        targets = TARGETS_HIGH_COVERAGE if stratum.startswith("10plus") else TARGETS
        totals = {k: 0.0 for k in targets}
        for tic, row in block.iterrows():
            n = float(row.n_events)
            total_after = sum(totals.values()) + n
            best, best_cost = None, None
            for k in targets:
                trial = dict(totals)
                trial[k] += n
                cost = sum((trial[s] / total_after - targets[s]) ** 2 for s in targets)
                # Tiny random tiebreak so ties do not always favour "train".
                cost += 1e-9 * rng.random()
                if best_cost is None or cost < best_cost:
                    best, best_cost = k, cost
            totals[best] += n
            assignment[int(tic)] = best
        mass[stratum] = totals
    return assignment, mass


def apply_split(events, assignment):
    ev = events.copy()
    ev["split"] = ev.tic.astype(int).map(assignment)
    return ev


def verify(events, assignment, excluded_tics=()):
    """The assertions that make the split trustworthy. Raises on any failure."""
    problems = []
    sets = {s: {t for t, v in assignment.items() if v == s} for s in TARGETS}
    for a in TARGETS:
        for b in TARGETS:
            if a < b and sets[a] & sets[b]:
                problems.append(f"TICs in both {a} and {b}: {sorted(sets[a] & sets[b])[:5]}")
    for t in excluded_tics:
        if t in assignment and assignment[t] != "test":
            problems.append(f"excluded TIC {t} assigned to {assignment[t]}, must be test")
    if "split" in events.columns:
        bad = events[events.split != events.tic.astype(int).map(assignment)]
        if len(bad):
            problems.append(f"{len(bad)} event rows disagree with their TIC's split")
        if events.split.isna().any():
            problems.append(f"{int(events.split.isna().sum())} events have no split")
    if problems:
        raise AssertionError("SPLIT VERIFICATION FAILED:\n  " + "\n  ".join(problems))
    return True


def report(events, assignment, excluded_tics=()):
    ev = apply_split(events, assignment)
    verify(ev, assignment, excluded_tics)
    out = {"seed": SPLIT_SEED, "targets": TARGETS,
           "targets_high_coverage": TARGETS_HIGH_COVERAGE,
           "n_tics": len(assignment), "by_split": {}}
    for s in TARGETS:
        sub = ev[ev.split == s]
        out["by_split"][s] = {
            "n_tics": int(sum(1 for v in assignment.values() if v == s)),
            "n_events": int(len(sub)),
            "event_fraction": float(len(sub) / max(len(ev), 1)),
            "n_positive": int((sub.label == 1).sum()),
        }
    # The TIC-to-split map is part of the frozen dataset's identity.
    payload = json.dumps({str(k): v for k, v in sorted(assignment.items())}, sort_keys=True)
    out["assignment_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return ev, out
