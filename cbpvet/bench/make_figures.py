"""The headline two-panel figure, drawn ONLY from frozen artifacts.

Panel A: manual-inspection false positives per 1,000 light curves at the
incumbent's matched recall (OP-B, 0.15-0.30% depth stratum), with the
TIC-cluster bootstrap CI on the model bar and the incumbent-information arm
(b0) as a hollow validity-check bar.
Panel B: the dual - recall at the incumbent's own FP budget, same stratum.

Reads t0_core.json + t0_addendum_2026-08-06.json and nothing else, so the
figure cannot drift from the banked record. Spec: Session-Sweep_2026-08-06.md
section 6 (awaiting sign-off); wording rules per Novelty-Audit v2. The
denominator is the pinned per-1,000-test-files convention, stated on the
figure itself.

Usage: python -m cbpvet.bench.make_figures [out.png]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "data", "bench", "full")

LABELS = ["incumbent\n(OP-B)", "arm b0\n(incumbent\ninfo only)", "arm b\n(full\nfeatures)"]


def build(out_path):
    t0 = json.load(open(os.path.join(BENCH, "t0_core.json")))
    add = json.load(open(os.path.join(BENCH, "t0_addendum_2026-08-06.json")))

    fp_m0 = t0["m0"]["B"]["real_fp_flagged"]
    fp_b0 = t0["t0_matched_recall"]["b0"]["OPB_stratum"]["queue_fp_only"]
    fp_b = t0["t0_matched_recall"]["b"]["OPB_stratum"]["queue_fp_only"]
    ci = t0["t0_matched_recall"]["b"]["OPB_stratum"]["fp_ci_2p5_97p5"]
    n_test = t0["n_test_files"]
    n_real = t0["n_real_test_files"]
    d = add["dual_recall_at_m0_matched_fp"]

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.subplots_adjust(wspace=0.42)
    for a in ax:
        a.tick_params(axis="x", labelsize=10)
    vals = [1000 * f / n_test for f in (fp_m0, fp_b0, fp_b)]
    ci_lo, ci_hi = (c * n_real / n_test for c in ci)
    ax[0].bar(LABELS, vals, color=["#777777", "none", "#4878a8"],
              edgecolor=["#777777", "#777777", "#4878a8"], linewidth=1.4)
    ax[0].errorbar(2, vals[2], yerr=[[vals[2] - ci_lo], [ci_hi - vals[2]]],
                   fmt="none", ecolor="k", capsize=4, lw=1.2)
    ax[0].set(ylabel="manual-inspection false positives\nper 1,000 test-split files",
              title="A. Same sensitivity, less junk")
    ax[0].text(2, ci_hi * 1.05, "~8x fewer", ha="center",
               fontsize=13, fontweight="bold", color="#4878a8")
    # b0 validity bar carries its own banked whisker (t0_core b0 OPB_stratum
    # CI rescaled between the two banked conventions - exact under the
    # fixed-denominator bootstrap).
    b0ci = t0["t0_matched_recall"]["b0"]["OPB_stratum"]["fp_ci_2p5_97p5"]
    b0lo, b0hi = (c * n_real / n_test for c in b0ci)
    ax[0].errorbar(1, vals[1], yerr=[[vals[1] - b0lo], [b0hi - vals[1]]],
                   fmt="none", ecolor="#777777", capsize=4, lw=1.0)

    rec_m0 = d["b"]["OPB_stratum"]["m0_recall_same_cell"]
    rec_b0 = d["b0"]["OPB_stratum"]["model_recall_at_m0_fp"]
    rec_b = d["b"]["OPB_stratum"]["model_recall_at_m0_fp"]
    ax[1].bar(LABELS, [100 * r for r in (rec_m0, rec_b0, rec_b)],
              color=["#777777", "none", "#c44e52"],
              edgecolor=["#777777", "#777777", "#c44e52"], linewidth=1.4)
    n_strat = t0["m0"]["B"]["n_stratum"]
    ax[1].set(ylabel=f"recall on {n_strat:,} injected transits,\n0.15-0.30% depth (%)",
              ylim=(0, 100),
              title=f"B. Same junk budget ({fp_m0} FPs), more transits recovered")
    ax[1].text(2, 100 * rec_b + 2, f"{100 * rec_b:.0f}% vs {100 * rec_m0:.0f}%",
               ha="center", fontsize=13, fontweight="bold", color="#c44e52")

    fig.suptitle(f"Head-to-head on the frozen test split ({n_test} files; "
                 "A: matched recall, B: matched FP budget)", y=1.02,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        BENCH, "figures", "t0_two_panel.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("wrote", build(out))
