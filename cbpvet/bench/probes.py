"""The seven probes: adversarial tests that the dataset is not secretly cheatable.

Why probes exist
----------------
A benchmark built from injections can be excellent-looking and worthless. Every
choice made while manufacturing the positives is a chance to leave a signature
that separates the classes for a reason that has nothing to do with planets. If
that happens, a model scores well here and fails on real data, and nothing in a
normal train-validate-test workflow would reveal it, because the leak is present
in all three splits.

So before the dataset is allowed to freeze, we deliberately TRY to cheat it. Each
probe trains a small classifier on one suspect quantity and asks whether that
quantity alone predicts the label. If it does, the dataset is leaking.

**The pass condition is inverted from normal machine learning: a probe must
FAIL to predict.** Gate is validation ROC-AUC <= 0.55, essentially chance.

The seven
---------
1. Inversion balance. Not a classifier but a count: does the inverted fraction
   differ between the classes? This is the one leak the whole dual campaign
   exists to close. Gate |delta| <= 0.05.
2. Provenance. Can a model tell a bank injection from an ELC injection, using
   positives only? If yes, "which generator made this" is contaminating the
   label, and the two provenances are not interchangeable evidence.
3. Eclipse-phase distance. Real false positives cluster near eclipses; if
   injections do not, phase distance alone separates the classes.
4. Host identity. Can the local view alone identify WHICH star an event came
   from? If yes, the model can memorise systems instead of learning transits.
5. Gap proximity. The most dangerous single leak, because mono-cbp's injector
   deliberately places injections away from gaps while real false positives
   cluster at them.
6. Coverage. Do positives come from systems with more observed binary cycles
   than negatives? Sampling introduces this easily.
7. Depth. REPORTED ONLY, never gated. Positives genuinely are deeper than
   negatives (recovered-positive median 0.519 percent against the real planets'
   0.215 percent), because shallow injections are genuinely harder to recover.
   That is real astrophysics, not a leak, so it is measured and disclosed rather
   than suppressed.

Splitting
---------
Always grouped by TIC. Events from one system are not independent, so a random
row split would put the same star in train and validation and let a probe pass
by memorising it. At full scale the FROZEN split is used; at pilot scale a
provisional GroupShuffleSplit is used and every result is labelled provisional.
"""

import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger("cbpvet.bench.probes")

GATE_AUC = 0.55
GATE_PROBE1_DELTA = 0.05
SEED = 0

# Single-column probes. Each name is the scalar the probe is allowed to see.
#
# NUMBERING CORRECTED 2026-08-04. An earlier version had probe 5 = gap_proximity
# and probe 6 = coverage, with a note that probe 6's column was "inferred". That
# was wrong: Amendment 3B.5 names it outright, "Probe 6 (gap_proximity) gates,
# WITH a construction fix and a pre-named response", and specifies that response
# precisely. Getting the number wrong would have attached probe 6's pre-named
# response to the wrong quantity, so the numbering now follows the plan.
SINGLE_COLUMN_PROBES = {
    3: ("phase_distance", "eclipse-phase distance"),
    5: ("log1p_n_cycles", "observed binary-cycle coverage"),
    6: ("gap_proximity", "distance to the nearest gap or sector edge"),
    7: ("depth", "event depth [REPORTED ONLY, never gated]"),
}

# Amendment 3B.5, verbatim in effect: if probe 6 fails after the joint
# importance-sampling construction fix, DROP gap_proximity from the training
# scalars (it stays in the shards) and re-gate. If it still fails, near-gap
# strata become reported-not-gated with the restriction stated in the manifest.
PROBE6_RESPONSE = ("drop gap_proximity from the training scalars, keep it in the "
                   "shards, and re-gate; if it still fails, near-gap strata "
                   "become reported-not-gated with the restriction in the manifest")
REPORTED_ONLY = {7}


def phase_distance(phase, prim_pos, prim_width, sec_pos, sec_width):
    """Wraparound distance to the nearer eclipse edge, floored at zero."""
    out = []
    for ph, pp, pw, sp, sw in zip(phase, prim_pos, prim_width, sec_pos, sec_width):
        d = []
        for pos, w in ((pp, pw), (sp, sw)):
            if pos is None or not np.isfinite(pos) or not np.isfinite(w):
                continue
            d.append(max(0.0, abs(((ph - pos + 0.5) % 1.0) - 0.5) - w / 2.0))
        out.append(min(d) if d else 0.0)
    return np.asarray(out)


def _grouped_split(groups, test_size=0.25):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def _auc(model, X, y, groups):
    """Fit on a TIC-grouped train split and score on the held-out systems."""
    if len(np.unique(y)) < 2:
        return np.nan, 0
    tr, va = _grouped_split(groups)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        return np.nan, len(va)
    model.fit(X[tr], y[tr])
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X[va])[:, 1]
    else:
        s = model.decision_function(X[va])
    return float(roc_auc_score(y[va], s)), int(len(va))


def probe_1_inversion(labels, inverted):
    """A count, not a classifier: inverted fraction per class."""
    pos, neg = inverted[labels == 1], inverted[labels == 0]
    if not len(pos) or not len(neg):
        return {"probe": 1, "name": "inversion balance", "value": None,
                "n": 0, "pass": None, "note": "one class empty"}
    r1, r0 = float(pos.mean()), float(neg.mean())
    delta = abs(r1 - r0)
    se = float(np.sqrt(0.25 / len(pos) + 0.25 / len(neg)))
    return {
        "probe": 1, "name": "inversion balance", "metric": "abs class delta",
        "value": delta, "positive_rate": r1, "negative_rate": r0,
        "standard_error": se, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "gate": GATE_PROBE1_DELTA, "pass": bool(delta <= GATE_PROBE1_DELTA),
        "underpowered": bool(2 * se > GATE_PROBE1_DELTA),
    }


def probe_2_provenance(scalars, label_source, labels, groups):
    """Bank against ELC, positives only, on the whole scalar block."""
    m = (labels == 1)
    y = label_source[m]
    classes = np.unique(y)
    if len(classes) < 2:
        return {"probe": 2, "name": "provenance", "value": None, "pass": None,
                "note": f"only one positive provenance present ({classes.tolist()}); "
                        "probe 2 cannot run until ELC positives exist"}
    X = np.nan_to_num(scalars[m], nan=0.0, posinf=0.0, neginf=0.0)
    yy = (y == classes[1]).astype(int)
    model = HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                           random_state=SEED)
    auc, n = _auc(model, X, yy, groups[m])
    return {"probe": 2, "name": "provenance (bank vs ELC)", "metric": "val ROC-AUC",
            "value": auc, "n_val": n, "gate": GATE_AUC,
            "pass": bool(auc <= GATE_AUC) if np.isfinite(auc) else None}


def probe_single_column(idx, column_values, labels, groups, name, reported_only=False):
    """Probes 3, 5, 6, 7: can ONE scalar alone predict the label?"""
    x = np.asarray(column_values, dtype=float).reshape(-1, 1)
    ok = np.isfinite(x).ravel()
    if ok.sum() < 50 or len(np.unique(labels[ok])) < 2:
        return {"probe": idx, "name": name, "value": None, "pass": None,
                "note": f"insufficient finite values ({int(ok.sum())})"}
    X = RobustScaler().fit_transform(x[ok])
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    auc, n = _auc(model, X, labels[ok], groups[ok])
    return {
        "probe": idx, "name": name, "metric": "val ROC-AUC", "value": auc,
        "n_val": n, "gate": None if reported_only else GATE_AUC,
        "reported_only": reported_only,
        "pass": None if reported_only else (bool(auc <= GATE_AUC) if np.isfinite(auc) else None),
    }


def probe_4_host_identity(local_views, tics, top_n=100):
    """Can the flattened local view alone name the host system?

    If it can, the model is free to memorise systems rather than learn transit
    morphology, and every per-TIC quirk becomes a shortcut. The gate is chance
    top-1 accuracy plus two standard errors.
    """
    X = local_views.reshape(len(local_views), -1)
    uniq, counts = np.unique(tics, return_counts=True)
    keep_tics = uniq[np.argsort(-counts)[:top_n]]
    m = np.isin(tics, keep_tics)
    if m.sum() < 200 or len(keep_tics) < 5:
        return {"probe": 4, "name": "host identity", "value": None, "pass": None,
                "note": f"too few events ({int(m.sum())}) or systems ({len(keep_tics)})"}
    Xs, ys = X[m], tics[m]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xs))
    cut = int(0.75 * len(perm))
    tr, va = perm[:cut], perm[cut:]
    model = LogisticRegression(max_iter=400, random_state=SEED, n_jobs=1)
    model.fit(np.nan_to_num(Xs[tr]), ys[tr])
    acc = float((model.predict(np.nan_to_num(Xs[va])) == ys[va]).mean())
    k = len(keep_tics)
    chance = 1.0 / k
    gate = chance + 2 * np.sqrt(chance * (1 - chance) / max(len(va), 1))
    return {"probe": 4, "name": "host identity", "metric": "top-1 accuracy",
            "value": acc, "n_systems": int(k), "n_val": int(len(va)),
            "chance": chance, "gate": float(gate), "pass": bool(acc <= gate)}


def mde(pbar, n_units, z=2.80):
    """Minimum detectable effect on a proportion, in absolute terms.

    ``n_units`` is the number of independent units, which is DISTINCT TICs and
    not events: events from one system are correlated, so counting events would
    overstate the power by a large factor.

    A worked check from the plan: N = 20, pbar = 0.5 gives MDE = 0.44. That is
    far above any target worth stating, which is why the pre-named response is
    to widen the stratum rather than to argue about it at the freeze.
    """
    if not n_units:
        return np.nan
    return float(z * np.sqrt(2 * pbar * (1 - pbar) / n_units))
