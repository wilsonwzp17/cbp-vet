"""Importance-sampled injection epochs, drawn from where real negatives live.

The shortcut this closes
------------------------
mono-cbp's injector draws the injection epoch uniformly over unmasked cadences,
and additionally requires the transit to land on real data rather than in a gap.
Both are sensible for measuring completeness. Both are ruinous for training data.

Real false positives are not uniform. They **cluster at data gaps and sector
edges**, because that is where detrending has the least information and goes
wrong, and they cluster near eclipses, because an imperfectly masked eclipse
wing is the single largest false-positive class (1,084 of mono-cbp's own 1,647).
So if positives are injected uniformly and negatives come from the real search,
then two features that have nothing to do with planets, namely "how far am I
from a gap" and "how far am I from an eclipse", separate the classes almost
perfectly. The model would learn those and we would never know from the
benchmark alone.

The fix: measure the real negatives' joint distribution over exactly those two
coordinates, then draw injection epochs from **that same distribution**. The
positives get placed where the negatives already are, so the two features carry
no label information and the model is forced onto the transit shape itself.

Two coordinates, deliberately joint
-----------------------------------
Marginal matching on each axis separately is not enough. If the real negatives
concentrate at "near a gap AND near an eclipse", matching each margin
independently would still leave that corner under-populated in the positives,
and the interaction would remain learnable. So the histogram is 2D and the draw
is bin-first over the joint table.

Who is exempt, and why
----------------------
Pair members. A pair's lead transit is drawn uniformly and its partner is placed
at ``lead + dt``, because the partner's position is set by orbital geometry, not
by our sampling choice. Forcing a pair into a histogram bin would distort the
very spacing distribution the pair model exists to reproduce.
"""

import logging

import numpy as np

logger = logging.getLogger("cbpvet.injection.epochs")

GAP_THRESHOLD_DAYS = 0.5   # must match experiments/07_search_frozen.py
MAX_BIN_REDRAWS = 20       # per recipe B3


def cadence_coordinates(time, phase, eclipse_params):
    """Per-cadence (phase distance to eclipse, time distance to gap or edge).

    Computed with the identical formulas used to build the histogram in
    experiments/07_search_frozen.py, so the sampler and the target distribution
    speak the same coordinates. Any divergence here would silently mis-target
    the sampling.
    """
    # --- distance in phase to the nearer eclipse edge ----------------------
    if phase is None or eclipse_params is None:
        d_phi = np.zeros_like(time)
    else:
        dists = []
        for pos, width in eclipse_params:
            if pos is None or width is None or not np.isfinite(pos) or not np.isfinite(width):
                continue
            centre = np.abs(((phase - pos + 0.5) % 1.0) - 0.5)
            dists.append(np.maximum(0.0, centre - width / 2.0))
        d_phi = np.min(np.vstack(dists), axis=0) if dists else np.zeros_like(time)

    # --- distance in time to the nearest edge or intra-sector gap ----------
    order = np.argsort(time)
    t_sorted = time[order]
    edges = [t_sorted[0], t_sorted[-1]]
    dt = np.diff(t_sorted)
    for j in np.where(dt > GAP_THRESHOLD_DAYS)[0]:
        edges.extend([t_sorted[j], t_sorted[j + 1]])
    edges = np.asarray(edges)
    d_gap = np.min(np.abs(time[:, None] - edges[None, :]), axis=1)

    return d_phi, d_gap


class HistogramEpochSampler:
    """Draw injection epochs bin-first from the real negatives' joint histogram.

    Per recipe B3: draw a bin with probability proportional to its count, then
    choose uniformly among the eligible unmasked cadences that fall in that bin.
    If the bin holds no eligible cadence for this light curve, redraw up to
    ``MAX_BIN_REDRAWS`` times, then fall back to the nearest non-empty bin, then
    to a uniform draw. Each fallback is counted so the campaign can report how
    often the target distribution could not be honoured.
    """

    def __init__(self, hist_npz_path, rng, catalogue=None):
        data = np.load(hist_npz_path)
        self.H = data["H"].astype(float)
        self.phase_edges = data["phase_edges"]
        self.gap_edges = data["gap_edges"]
        self.rng = rng
        self.catalogue = catalogue

        flat = self.H.ravel()
        total = flat.sum()
        if total <= 0:
            raise ValueError(f"Histogram in {hist_npz_path} is empty")
        self.bin_p = flat / total
        self.n_phase = self.H.shape[0]
        self.n_gap = self.H.shape[1]
        self.nonempty = np.flatnonzero(flat > 0)

        self._coord_cache = {}
        self.counters = {"bin_hit": 0, "redraw": 0, "nearest": 0, "uniform": 0}

    def _eclipse_params(self, tic):
        if self.catalogue is None:
            return None
        row = self.catalogue[self.catalogue["tess_id"] == tic]
        if row.empty:
            return None
        r = row.iloc[0]
        return [(r.get("prim_pos"), r.get("prim_width")), (r.get("sec_pos"), r.get("sec_width"))]

    def _bins_for_file(self, key, time, phase, tic):
        """Cache the per-cadence bin index for a light curve."""
        if key in self._coord_cache:
            return self._coord_cache[key]
        d_phi, d_gap = cadence_coordinates(time, phase, self._eclipse_params(tic))
        # np.digitize gives 1-based indices; clip into valid bin range.
        i_phi = np.clip(np.digitize(d_phi, self.phase_edges) - 1, 0, self.n_phase - 1)
        i_gap = np.clip(np.digitize(d_gap, self.gap_edges) - 1, 0, self.n_gap - 1)
        flat_idx = i_phi * self.n_gap + i_gap
        self._coord_cache[key] = flat_idx
        if len(self._coord_cache) > 64:          # bound memory across a worker
            self._coord_cache.pop(next(iter(self._coord_cache)))
        return flat_idx

    def __call__(self, time, mask_indices, phase, meta):
        key = (meta["tic"], meta["sector"])
        flat_idx = self._bins_for_file(key, time, phase, meta["tic"])
        eligible_bins = flat_idx[mask_indices]

        for attempt in range(MAX_BIN_REDRAWS):
            target = self.rng.choice(len(self.bin_p), p=self.bin_p)
            hits = mask_indices[eligible_bins == target]
            if len(hits):
                self.counters["bin_hit" if attempt == 0 else "redraw"] += 1
                return float(time[self.rng.choice(hits)])

        # Nearest non-empty bin that this light curve can actually populate.
        available = np.unique(eligible_bins)
        if len(available):
            weights = self.bin_p[available]
            if weights.sum() > 0:
                self.counters["nearest"] += 1
                pick = self.rng.choice(available, p=weights / weights.sum())
                hits = mask_indices[eligible_bins == pick]
                return float(time[self.rng.choice(hits)])

        self.counters["uniform"] += 1
        return float(time[self.rng.choice(mask_indices)])

    def report(self):
        total = sum(self.counters.values()) or 1
        return {k: v for k, v in self.counters.items()} | {
            "fraction_on_target": (self.counters["bin_hit"] + self.counters["redraw"]) / total
        }
