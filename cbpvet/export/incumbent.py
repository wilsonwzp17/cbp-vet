"""The 17-column incumbent feature block, from mono-cbp's own model comparator.

What the incumbent actually is
------------------------------
mono-cbp's vetting stage fits four models to every flagged event (transit,
sinusoid, linear, step), ranks them by AIC, and labels the event with the
winner. Its published funnel narrows 7,176 events to 1,647 that a human then
inspects, and that 1,647 is exactly 715 "T" plus 932 "AT".

For the benchmark this matters twice over. It is the thing our model is measured
against, and it is also a feature block our model is allowed to consume: ablation
arm (b) asks whether the learned vetter adds anything ON TOP of what the
incumbent already computes. So the incumbent's own outputs have to be exported
faithfully.

Why seventeen and not nine
--------------------------
Every prior planning document said nine. Read from ``comparator.py`` directly,
the results dict is::

    filename, best_fit,
    aic_transit, aic_sinusoidal, aic_linear, aic_step,
    rmse_transit, rmse_sinusoidal, rmse_linear, rmse_step

That is eight numbers and one categorical label. There are no posteriors. The
"nine" everyone carried was the count of categorical CLASSES that
``_classify_event`` can emit, not a count of columns.

So the block is **4 delta-AIC + 4 RMSE + a 9-level one-hot = 17 columns.**
Writing nine floats would have silently dropped the four RMSEs, and the RMSEs
are precisely what separates T from AT, which is the 715-versus-932 split that
defines the incumbent's manual queue. After the freeze, ablation arm (b) would
have been unrepairable.

Deltas rather than raw AIC
--------------------------
Raw AIC values scale with the number of cadences in the window, so they encode
window length as much as model quality. The comparison is what carries meaning,
so each model's AIC is exported relative to the best of the four.

The silent-drop trap
--------------------
``compare_events`` wraps each event in ``try/except`` and, on failure, logs and
**does not append a row**. So its output can be shorter than its input with no
error raised. Joining on position, or assuming equal lengths, would silently
misalign every downstream feature. Here the join is an explicit outer join on
filename, and a failed event keeps its row with an all-zero block and
``incumbent_valid = 0``. An event is never dropped: dropping would change the
benchmark's denominator invisibly.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cbpvet.export.incumbent")

# Fixed order, the single source of truth for both the exporter and arm (b0).
# Changing this order after the freeze would silently permute the one-hot.
BEST_FIT_CLASSES = ["T", "AT", "Sin", "ASin", "L", "AL", "St", "ASt", "A"]

AIC_KEYS = ["aic_transit", "aic_sinusoidal", "aic_linear", "aic_step"]
RMSE_KEYS = ["rmse_transit", "rmse_sinusoidal", "rmse_linear", "rmse_step"]

DELTA_AIC_COLS = [f"delta_{k}" for k in AIC_KEYS]
ONEHOT_COLS = [f"best_fit_{c}" for c in BEST_FIT_CLASSES]
INCUMBENT_COLS = DELTA_AIC_COLS + RMSE_KEYS + ONEHOT_COLS   # 4 + 4 + 9 = 17

assert len(INCUMBENT_COLS) == 17, "the incumbent block must be exactly 17 columns"


def build_block(comparator_rows, event_keys):
    """Build the 17-column block, aligned to ``event_keys`` by outer join.

    Args:
        comparator_rows: DataFrame from ``ModelComparator.compare_events``,
            carrying at least ``filename`` plus the four AIC and four RMSE keys.
        event_keys: ordered sequence of the filenames of EVERY event that was
            submitted, including ones the comparator may have dropped.

    Returns:
        DataFrame indexed like ``event_keys``, with the 17 columns plus
        ``incumbent_valid``.
    """
    keys = pd.Index(list(event_keys), name="filename")
    out = pd.DataFrame(0.0, index=keys, columns=INCUMBENT_COLS)
    out["incumbent_valid"] = 0

    if comparator_rows is None or not len(comparator_rows):
        logger.warning("No comparator rows; the whole block is zero and invalid")
        return out.reset_index()

    df = comparator_rows.copy()
    if "filename" not in df.columns:
        raise KeyError("comparator output has no 'filename' column to join on")
    df = df.drop_duplicates("filename").set_index("filename")

    common = keys.intersection(df.index)
    n_missing = len(keys) - len(common)
    if n_missing:
        logger.warning(
            "%d of %d events have no comparator row (silently dropped by "
            "compare_events); their block is zero with incumbent_valid = 0",
            n_missing, len(keys),
        )
    sub = df.loc[common]

    aic = sub[AIC_KEYS].to_numpy(dtype=float)
    # Delta against the best of the four. Rows with any non-finite AIC cannot be
    # compared and stay invalid rather than propagating NaN into the features.
    finite = np.isfinite(aic).all(axis=1)
    best = np.where(finite, np.nanmin(aic, axis=1), np.nan)
    delta = aic - best[:, None]

    rmse = sub[RMSE_KEYS].to_numpy(dtype=float)
    rmse_ok = np.isfinite(rmse).all(axis=1)
    valid = finite & rmse_ok

    out.loc[common, DELTA_AIC_COLS] = np.where(valid[:, None], delta, 0.0)
    out.loc[common, RMSE_KEYS] = np.where(valid[:, None], rmse, 0.0)

    labels = sub["best_fit"].astype(str).to_numpy() if "best_fit" in sub.columns else np.array([""] * len(sub))
    for i, cls in enumerate(BEST_FIT_CLASSES):
        out.loc[common, ONEHOT_COLS[i]] = np.where(valid & (labels == cls), 1.0, 0.0)

    unknown = set(labels[valid]) - set(BEST_FIT_CLASSES)
    if unknown:
        raise ValueError(
            f"comparator emitted best_fit values outside the pinned 9 classes: "
            f"{sorted(unknown)}. The one-hot order is a frozen contract."
        )

    out.loc[common, "incumbent_valid"] = valid.astype(int)
    n_valid = int(out["incumbent_valid"].sum())
    logger.info("Incumbent block: %d of %d events valid (%.1f%%)",
                n_valid, len(out), 100 * n_valid / max(len(out), 1))
    return out.reset_index()


def event_dicts_from_snippets(snippet_paths, width_key="duration"):
    """Turn saved snippets into the dict form ``compare_events`` accepts.

    The comparator wants ``time, flux, flux_err, event_time, event_width``.
    ``filename`` is carried through so the outer join has a key.
    """
    events = []
    for path in snippet_paths:
        d = np.load(path, allow_pickle=True)
        events.append({
            "time": d["time"], "flux": d["flux"], "flux_err": d["flux_err"],
            "event_time": float(d["event_time"]),
            "event_width": float(d[width_key]) if width_key in d else 0.2,
            "filename": path.split("/")[-1].replace(".npz", ""),
        })
    return events
