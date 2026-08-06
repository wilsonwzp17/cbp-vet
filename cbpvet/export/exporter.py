"""Assemble events from every provenance into one frozen HDF5 shard set.

The contract
------------
Real-search negatives, bank injections, and ELC injections all become records
with the same fields, computed by the same functions, in the same order. The
only thing allowed to differ between them is the flux array the local view is
cut from, and that MUST differ: an injected event only exists in the injected
array, while a real event only exists in the original cached file.

Everything else is shared, on purpose:

* the recurrence view is always cut from the ORIGINAL cached file, for both
  classes, which is what makes it sign-invariant;
* gap proximity is always measured on the FULL cached time array, never on a
  snippet, so it means the same thing for a 0.05 d and a 0.9 d event;
* the local view always uses the same half-width rule,
  ``max(3 * t_dur, 0.75 d)``. The real-search snippets already on disk are the
  finder's narrower stock windows, so real events are re-cut from the staged
  masked files at the wide width rather than being exported at a different
  width from the injections. Mixed widths across classes would be a provenance
  tell as clean as the inversion leak.
"""

import json
import logging
import os

import h5py
import numpy as np
import pandas as pd

from .. import physics
from . import features, schema
from .incumbent import INCUMBENT_COLS

logger = logging.getLogger("cbpvet.export")

DEDUP_TOL_DAYS = 0.0209


class EventExporter:
    """Builds records and writes them to HDF5 shards."""

    def __init__(self, staged_dir, catalogue, raw_catalogue, noise_screen=None,
                 e2_passed=False):
        self.staged_dir = staged_dir
        self.cat = catalogue.drop_duplicates("tess_id").set_index("tess_id")
        self.raw = raw_catalogue.drop_duplicates("tess_id").set_index("tess_id")
        self.noise = noise_screen.set_index("tic") if noise_screen is not None else None
        # E2 decides whether the four pair scalars are admitted as features. They
        # are always WRITTEN, so the decision stays auditable, but when E2 has
        # not passed they are marked invalid so no arm can consume them.
        self.e2_passed = e2_passed
        self._file_cache = {}
        self.stats = {"records": 0, "no_cached_file": 0, "no_host_row": 0}

    # ------------------------------------------------------------------
    def _cached(self, tic, sector):
        """Load and cache one staged light curve: time, flux, err, phase."""
        key = (int(tic), int(sector))
        if key in self._file_cache:
            return self._file_cache[key]
        path = os.path.join(self.staged_dir, f"TIC_{int(tic)}_{int(sector):02d}.txt")
        if not os.path.exists(path):
            self._file_cache[key] = None
            return None
        arr = np.loadtxt(path, skiprows=1)
        if arr.ndim != 2 or arr.shape[0] < 2:
            self._file_cache[key] = None
            return None
        out = {"time": arr[:, 0], "flux": arr[:, 1], "flux_err": arr[:, 2],
               "phase": arr[:, 3]}
        if len(self._file_cache) > 200:
            self._file_cache.pop(next(iter(self._file_cache)))
        self._file_cache[key] = out
        return out

    def _all_sectors(self, tic):
        """Every frozen sector of this TIC, concatenated, for the recurrence view.

        THE RECIPE SAYS SO, AND IT MATTERS ENORMOUSLY. Execution-Readiness W3.1b:
        "files = the TIC's frozen cached sectors from the catalog sectors column,
        NOT a directory glob".

        The first implementation passed only the event's OWN sector. Measured
        consequence, 2026-08-04:

            other cycles available, median   one sector 1.21   all sectors 2.54
            systems reaching >= 10 cycles    one sector 0      all sectors 79

        The 10-plus coverage tier is where the recurrence view is supposed to do
        its work, where the frozen split deliberately over-allocates to test at
        60/15/25, and where the MDE is computed. Under the single-sector version
        that tier was EMPTY, and the recurrence view scored ROC-AUC 0.552, which
        is chance. This was a starved feature, not a useless one.

        The catalogue's sectors column is used rather than a directory glob so
        the 122 newer sector-60-plus pulls stay out of the frozen set.
        """
        tic = int(tic)
        key = ("ALL", tic)
        if key in self._file_cache:
            return self._file_cache[key]
        sectors = []
        if tic in self.raw.index:
            raw_sectors = str(self.raw.loc[tic].get("sectors", ""))
            for tok in raw_sectors.replace(";", ",").split(","):
                tok = tok.strip()
                if tok and tok != "nan":
                    try:
                        sectors.append(int(float(tok)))
                    except ValueError:
                        continue
        parts = []
        for sec in sorted(set(sectors)):
            d = self._cached(tic, sec)
            if d is not None:
                parts.append(d)
        if not parts:
            self._file_cache[key] = None
            return None
        out = {k: np.concatenate([p[k] for p in parts]) for k in ("time", "flux", "phase")}
        order = np.argsort(out["time"])
        out = {k: v[order] for k, v in out.items()}
        out["n_sectors"] = len(parts)
        self._file_cache[key] = out
        return out

    def _host(self, tic):
        """Host-level constants, identical for every event of this TIC."""
        tic = int(tic)
        if tic not in self.cat.index:
            return None
        c, r = self.cat.loc[tic], self.raw.loc[tic] if tic in self.raw.index else None
        p_bin = float(c["period"])
        ecc = 0.0
        if r is not None:
            _, _, e = physics.eclipse_eccentricity(
                r.get("prim_pos_2g"), r.get("sec_pos_2g"),
                r.get("prim_width_2g"), r.get("sec_width_2g"))
            ecc = float(e) if np.isfinite(e) else 0.0
        pc = physics.p_crit(p_bin, ecc)
        depth_ratio = np.nan
        if self.noise is not None and tic in self.noise.index:
            depth_ratio = float(self.noise.loc[tic, "depth_ratio"])
        return {
            "p_bin": p_bin, "bjd0": float(c["bjd0"]), "ecc": ecc, "p_crit": pc,
            "tmag": float(r["Tmag"]) if r is not None and np.isfinite(r.get("Tmag", np.nan)) else np.nan,
            "morph": float(r["morph_coeff"]) if r is not None and np.isfinite(r.get("morph_coeff", np.nan)) else np.nan,
            "depth_ratio": depth_ratio,
            # Host-level and CONSTANT per TIC, as the unit test asserts.
            "log10_pp_over_pbin": float(np.log10(pc / p_bin)),
            "same_star_multi_allowed": features.same_star_multi_allowed(
                pc, p_bin, mass_ratio=0.3),
        }

    # ------------------------------------------------------------------
    def set_system_events(self, table, key=("tic",)):
        """Give the exporter the other flagged events of each system.

        Leg 3 needs, for every event, the other events the SAME search run
        flagged on the SAME system. Passing them in keeps build_record free of
        any notion of injection truth, so the feature is exactly as computable at
        deployment as it is here.
        """
        self._sys_events = {}
        self._sys_key = key
        if table is None or not len(table):
            return
        t = table.copy()
        t.columns = [c.lower() for c in t.columns]
        tcol = "event_time" if "event_time" in t.columns else "time"
        # GROUPING MATTERS. Real-search events of one TIC were all flagged by the
        # same run, so they were genuinely seen together and group by tic.
        # Campaign events must group by (tic, sector, model_idx) = ONE TEST:
        # pooling across tests would relate events from DIFFERENT injections that
        # a vetter would never see together, inventing pairs that do not exist.
        cols = [c for c in key if c in t.columns]
        for gk, g in t.groupby([t[c] for c in cols] if len(cols) > 1 else t[cols[0]]):
            gk = gk if isinstance(gk, tuple) else (gk,)
            deps = (g["depth"].to_numpy(dtype=float) if "depth" in g.columns
                    else np.full(len(g), np.nan))
            self._sys_events[tuple(int(x) for x in gk)] = (
                g[tcol].to_numpy(dtype=float), g["duration"].to_numpy(dtype=float), deps)

    def build_record(self, ev, label_source, local_flux_source=None):
        """One event to one record. ``ev`` is a dict-like row."""
        tic, sector = int(ev["tic"]), int(ev["sector"])
        cached = self._cached(tic, sector)
        host = self._host(tic)
        if cached is None:
            self.stats["no_cached_file"] += 1
            return None
        if host is None:
            self.stats["no_host_row"] += 1
            return None

        t_dur = float(ev["duration"])
        t_ev = float(ev["event_time"])
        phase_ev = float(ev["phase"])
        depth = float(ev["depth"])

        # --- local view: the ONLY thing whose source differs by class -------
        if local_flux_source is not None:
            lt, lf = local_flux_source
        else:
            lt, lf = cached["time"], cached["flux"]
        local, scatter = features.local_view(lt, lf, t_ev, t_dur)

        # --- recurrence: ALWAYS from the original cached flux, and from ALL
        # of this TIC's frozen sectors, not just the event's own. See
        # _all_sectors: the single-sector version left the 10-plus coverage
        # tier empty and the feature at chance.
        multi = self._all_sectors(tic) or cached
        recur, n_cycles, rec_scalars = features.recurrence_view(
            multi["time"], multi["flux"], multi["phase"], t_ev, phase_ev,
            t_dur, host["p_bin"], depth, scatter)

        # gap_proximity stays on the event's OWN sector. Measuring it on the
        # concatenated multi-sector array would treat every inter-sector break
        # as a gap edge, which is true but useless: the quantity is about the
        # event's local data continuity, and the recipe says "from the FULL
        # cached file's time array", singular.
        gp = features.gap_proximity(cached["time"], t_ev)

        rec = {
            "tic": tic, "sector": sector,
            "event_time": t_ev, "phase": phase_ev,
            "label": int(ev.get("label", 0)),
            "label_source": schema.LABEL_SOURCES.index(label_source),
            "inverted_lc": int(ev.get("inverted_lc", 0)),
            # --- core scalars ---
            "snr": float(ev["snr"]),
            "depth": depth,
            "t_dur": t_dur,
            "log_p_bin": float(np.log10(host["p_bin"])),
            "morph_coeff": host["morph"],
            "tmag": host["tmag"],
            "sin_phase": float(np.sin(2 * np.pi * phase_ev)),
            "cos_phase": float(np.cos(2 * np.pi * phase_ev)),
            "detrend_fraction": float(ev.get("n_detrend_detections", np.nan)) / 21.0,
            "skye_flag": float(ev["skye_flag"]) if label_source == "real_search" and "skye_flag" in ev else np.nan,
            "gap_proximity": gp,
            "local_scatter": scatter,
            "log1p_n_cycles": float(np.log1p(min(n_cycles, schema.N_CYCLES_CAP))),
            "rec_depth_med": rec_scalars["rec_depth_med"],
            "rec_depth_mad": rec_scalars["rec_depth_mad"],
            "rec_frac_dipping": rec_scalars["rec_frac_dipping"],
            # --- host-level constants ---
            "same_star_multi_allowed": host["same_star_multi_allowed"],
            "log10_pp_over_pbin": host["log10_pp_over_pbin"],
            "n_cycles_raw": n_cycles,
        }
        # --- gated pair scalars: OBSERVED quantities only (2026-08-06) --------
        # From the system's other flagged events, one code path for both
        # classes. The injection-truth versions below are AUDIT columns and are
        # excluded from training_scalars by schema + regression test.
        obs = self._observed_partner(
            tuple(int(ev[c]) for c in getattr(self, "_sys_key", ("tic",))
                  if c in ev) or (tic,), t_ev, t_dur, depth, host["p_bin"])
        obs_depth_ratio = (obs["depth_other"] / depth
                           if np.isfinite(obs["depth_other"]) and depth > 0 else np.nan)
        rec.update({
            "has_observed_pair": float(np.isfinite(obs["dt"])),
            "pair_dt_over_pbin": (obs["dt"] / host["p_bin"]
                                  if np.isfinite(obs["dt"]) else np.nan),
            "pair_duration_ratio": obs["ratio"],
            "pairing_depth_ratio_vs_expectation":
                features.pairing_depth_ratio_vs_expectation(
                    obs_depth_ratio, host["depth_ratio"]),
            # --- injection-truth AUDIT columns, never features ---
            # NOTE: in runs before 2026-08-06 the injector stamped pair_role
            # 'lead' on every recovered injection, paired or not; interpret via
            # tests_all.csv's partner_injected when auditing those runs.
            "truth_pair_role": str(ev.get("pair_role", "none")),
            "truth_partner_dt": float(ev.get("partner_dt", np.nan)),
            "truth_partner_ratio": float(ev.get("partner_ratio", np.nan)),
        })
        self.stats["records"] += 1
        return rec, local, recur

    # ------------------------------------------------------------------
    def _observed_partner(self, key, t_ev, t_dur, this_depth, p_bin):
        """Everything the pair features may legally know: OBSERVED flagged events.

        CORRECTION OF RECORD, 2026-08-06. The gated pair features were previously
        derived from injection bookkeeping (pair_role, partner_dt, partner_ratio).
        The pre-freeze stress test measured the consequence: has_pair_same_star
        EQUALLED the label, single-column val ROC-AUC 1.0000, because pair_role
        was stamped on every recovered injection and real events carried none.

        The doctrine this enforces: star identity is UNOBSERVABLE for a real
        event -- that is ELC's entire point -- so no observed feature may claim
        it. Same-star/cross-star classification lives ONLY in ELC audit columns.
        What a vetter can actually observe about a candidate pair is: whether
        another flagged event of the same system sits inside the per-system pair
        window, how far away, its duration ratio, and its depth ratio. All four
        are computed here, identically for both classes, from flagged events
        alone, so they are exactly as computable at deployment as in training.
        """
        sys_ev = getattr(self, "_sys_events", {}).get(key if isinstance(key, tuple) else (int(key),))
        if sys_ev is None:
            return {"ratio": np.nan, "dt": np.nan, "n": 0, "depth_other": np.nan}
        times, durs, deps = sys_ev
        keep = np.abs(times - float(t_ev)) > 1e-9      # exclude the event itself
        ratio, dt, n, dep = features.nearest_pair_partner(
            t_ev, t_dur, times[keep], durs[keep], p_bin, other_depths=deps[keep])
        return {"ratio": ratio, "dt": dt, "n": n, "depth_other": dep}

    def scalar_valid(self, rec):
        """Bit vector over the conditional scalars.

        Invalid means UNDEFINED, not missing. skye_flag has no meaning for an
        injection; the pair scalars have no admitted meaning until E2 passes.
        Marking them rather than imputing keeps the model from reading an
        imputed value as a real measurement.
        """
        bits = {}
        for name in schema.CONDITIONAL_SCALARS:
            if name in schema.GATED_PAIR_SCALARS and not self.e2_passed:
                bits[name] = 0
            else:
                v = rec.get(name, np.nan)
                bits[name] = int(np.isfinite(v)) if isinstance(v, (int, float, np.floating)) else 0
        return bits


def write_shard(path, records, locals_, recurs, incumbent_df=None):
    """Write one HDF5 shard: scalars table, both views, and the incumbent block."""
    df = pd.DataFrame(records)
    if incumbent_df is not None:
        for c in INCUMBENT_COLS + ["incumbent_valid"]:
            df[c] = incumbent_df[c].to_numpy() if c in incumbent_df else 0.0

    with h5py.File(path, "w") as h5:
        h5.create_dataset("local", data=np.stack(locals_).astype(np.float32),
                          compression="gzip", compression_opts=4)
        h5.create_dataset("recurrence", data=np.stack(recurs).astype(np.float32),
                          compression="gzip", compression_opts=4)
        grp = h5.create_group("scalars")
        for col in df.columns:
            vals = df[col].to_numpy()
            if vals.dtype == object:
                vals = vals.astype("S64")
            grp.create_dataset(col, data=vals, compression="gzip", compression_opts=4)
        h5.attrs["n_events"] = len(df)
        h5.attrs["local_shape"] = str(np.stack(locals_).shape)
        h5.attrs["recurrence_shape"] = str(np.stack(recurs).shape)
        h5.attrs["columns"] = json.dumps(list(df.columns))
    logger.info("Wrote %s: %d events", path, len(df))
    return df
