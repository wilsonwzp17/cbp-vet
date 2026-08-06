"""B2: the dual-inversion injection harness.

The problem this exists to solve
--------------------------------
mono-cbp's injector flips every light curve upside down before injecting::

    flux = ((flux - 1) * -1) + 1        # injector.py line 251, unconditional

Real dips become bumps, so any dip the search then finds must be the injected
one. That is correct for measuring completeness, and poison for training data.
If every positive lives in an inverted light curve and every negative lives in a
native one, then "is this light curve upside down" is a perfect predictor of the
label. Stellar variability, systematics, and detrending residuals all change
sign, so the shortcut is not subtle. A model could score beautifully on our
benchmark and be worthless on real data.

The fix, Amendment-1 section 3B.7: **every provenance runs 50/50 inverted and
native**, so inversion carries no information about the label. On the native
half the inversion trick no longer protects us, so instead we mask out every
event the real search already found before drawing an injection epoch. A
"recovery" still cannot be somebody else's dip.

Three things the stock injector hard-codes that had to be opened up
------------------------------------------------------------------
1. Inversion at line 251 is unconditional. Here it is a per-test coin flip.
2. The epoch is drawn internally at line 258 with
   ``np.random.choice(mask_indices)``, uniform over unmasked cadences. Uniform
   is wrong for us: real false positives cluster near gaps and sector edges,
   and mono-cbp additionally requires injections to land away from gaps. Inject
   uniformly and "distance to the nearest gap" separates the classes on its own.
   Here the draw is delegated to an injectable ``epoch_sampler``, so the
   campaign can importance-sample from the real negatives' own joint
   distribution of eclipse-phase distance and gap distance.
3. Only the event closest to the injection is recorded; every other flagged
   event in the same light curve is discarded. Those discarded events are
   free, genuine, in-the-wild negatives, and here they are kept.

Why event extraction is delegated to TransitFinderExt
-----------------------------------------------------
The injector has its own event-processing path (``_process_cb_injection``) that
does NOT group peaks across the 21 biweight windows the way the finder does. If
injected positives had their features computed by one code path and real
negatives by another, the difference between the code paths would itself be
learnable, which is the same failure as the inversion leak wearing a different
hat. So after detrending, the injected light curve is handed to the very same
``TransitFinderExt._process_cb_events`` that produced the real negatives.
Identical grouping, identical SNR, identical ``n_detrend_detections``.

Snippet width, and one obligation this places on the exporter
-------------------------------------------------------------
The output contract pins snippets at ``event_time +/- max(3 * t_dur, 0.75 d)``,
which is wider than the finder's stock window of ``max(event +/- 1.5 dur,
+/- 0.5 d)``. Wider is right, because the recurrence view needs sideband
continuum. But it means the real-search snippets already on disk are narrower
than these. **The exporter must re-extract real-event snippets at this same
width from the staged masked files**, which still exist, rather than mixing
widths across classes. Mixed widths would be one more provenance tell.
"""

import logging
import os

import numpy as np
import pandas as pd

from mono_cbp.injection_retrieval.injector import TransitInjector
from mono_cbp.utils import bin_to_long_cadence, get_eclipse_mask, time_to_phase
from mono_cbp.utils.detrending import detrend

from ..search import TransitFinderExt

logger = logging.getLogger("cbpvet.injection")

CADENCE_DAYS = 30.0 / 1440.0          # 0.0208 d, the long-cadence bin
SNIPPET_MIN_HALFWIDTH = 0.75          # days
SNIPPET_DUR_MULTIPLE = 3.0

# TICs that must never be injected into: they carry the known real planets the
# benchmark is graded on, so injecting into them would contaminate the test set.
DEFAULT_EXCLUDED_TICS = frozenset({260128333, 319011894, 172900988})


_STUB_PATH = os.path.join(os.path.dirname(__file__), "_stub_bank.npz")


def _stub_bank_path():
    """A one-model bank, written once, so the parent's eager loader is cheap."""
    if not os.path.exists(_STUB_PATH):
        t = np.arange(-1.0, 1.0 + 1e-9, CADENCE_DAYS)
        np.savez(
            _STUB_PATH, time=t, num_depths=1, num_durations=1,
            depth_range=(1e-3, 1e-2), duration_range=(0.1, 1.0), cadence_minutes=30,
            model_0_flux=np.zeros_like(t), model_0_depth=0.0,
            model_0_duration=0.1, model_0_impact_parameter=0.0, model_0_ror=0.0,
        )
    return _STUB_PATH


def load_bank_slice(path, lo, hi):
    """Load only models [lo, hi) from a bank npz.

    ``load_transit_models`` rebuilds all N model dicts regardless of need, which
    is 56.6 s on the 16,384-profile bank. A worker only ever touches its own
    contiguous slice, so it pays for that slice alone.
    """
    with np.load(path, allow_pickle=True) as data:
        return [
            {
                "flux": data[f"model_{i}_flux"],
                "depth": float(data[f"model_{i}_depth"]),
                "duration": float(data[f"model_{i}_duration"]),
                "impact_parameter": float(data[f"model_{i}_impact_parameter"]),
                "ror": float(data[f"model_{i}_ror"]),
            }
            for i in range(lo, hi)
        ]


def default_epoch_sampler(rng):
    """Uniform draw over unmasked cadences, matching stock behaviour.

    Used for the pilot and as the fallback. The campaign supplies an
    importance-sampling replacement built from the real negatives' histogram.
    """

    def _sample(time, mask_indices, phase, meta):
        return float(time[rng.choice(mask_indices)])

    return _sample


class DualInjector(TransitInjector):
    """TransitInjector with per-test inversion, injectable epochs, and full harvest.

    Args:
        transit_models_path: path to the v2 bank npz.
        real_events_df: the FULL unfiltered ``detected_events.txt`` from the
            frozen search. Unfiltered on purpose: on the native path we mask
            around every TCE the search can flag, not only the ones that survive
            the SNR and duration cuts, because any of them could be mistaken for
            our injection.
        rng_seed: pinned per worker; the campaign derives per-worker seeds from
            one campaign seed so the whole run is reproducible.
        epoch_sampler: callable ``(time, mask_indices, phase, meta) -> float``.
        p_invert: probability of inverting a given test. 0.5 is the ratified
            value; it is a parameter only so the pilot can assert against it.
    """

    def __init__(self, transit_models_path, real_events_df=None, rng_seed=0,
                 epoch_sampler=None, catalogue=None, config=None, TEBC=False,
                 p_invert=0.5, provenance="bank", run_id="run", load_models=True):
        # TransitInjector.__init__ eagerly calls load_transit_models, which
        # rebuilds every model dict by indexing the npz five times per model.
        # On the 16,384-profile bank that is 56.6 s, MEASURED 2026-08-04.
        # DualInjector never reads self.transit_models: the campaign passes the
        # flux array for each test explicitly, because workers only need their
        # own slice. So when load_models is False we hand the parent a one-model
        # stub instead, which is loaded in microseconds and never used. This was
        # worth 113 s of pure startup per worker.
        if not load_models:
            transit_models_path = _stub_bank_path()
        super().__init__(transit_models_path, catalogue=catalogue, config=config, TEBC=TEBC)
        self.rng = np.random.default_rng(rng_seed)
        self.rng_seed = rng_seed
        self.p_invert = p_invert
        self.provenance = provenance
        self.run_id = run_id
        self.epoch_sampler = epoch_sampler or default_epoch_sampler(self.rng)

        # Index the real events by (tic, sector) once; per-file lookup is hot.
        self.real_events = {}
        if real_events_df is not None and len(real_events_df):
            df = real_events_df.copy()
            df.columns = [c.lower() for c in df.columns]
            for (tic, sector), grp in df.groupby([df["tic"].astype(int), df["sector"].astype(int)]):
                self.real_events[(int(tic), int(sector))] = (
                    grp["time"].to_numpy(dtype=float),
                    grp["duration"].to_numpy(dtype=float),
                )
            logger.info("Indexed %d real events over %d (tic, sector) keys",
                        len(df), len(self.real_events))

        # The event extractor: the SAME class that produced the real negatives.
        self._finder = TransitFinderExt(
            catalogue=self.catalogue, sector_times=None, config=config, TEBC=False
        )

        self.event_rows = []   # every flagged event, positive and negative
        self.test_rows = []    # one row per injection test

    # ------------------------------------------------------------------
    def _mask_real_events(self, time, mask, tic, sector, duration_model):
        """Native path only: forbid epochs near events the real search already found.

        Exclusion half-width is the real event's own half-duration, plus half
        the injected duration, plus one cadence of slack. Two events whose
        centres are closer than that could be confused for one another under the
        t_dur/2 recovery rule, so an injection there could be scored as
        recovered when what was actually found was the pre-existing event.
        """
        key = (int(tic), int(sector))
        if key not in self.real_events:
            return mask, 0
        times, durations = self.real_events[key]
        blocked = np.zeros_like(mask)
        for t_real, dur_real in zip(times, durations):
            if not np.isfinite(dur_real):
                dur_real = 0.0
            half = dur_real / 2.0 + duration_model / 2.0 + CADENCE_DAYS
            blocked |= np.abs(time - t_real) < half
        return mask & ~blocked, int(blocked.sum())

    def _snippet(self, time, flux, flux_err, t_event, t_dur):
        """Wide event window, per the pinned output contract."""
        half = max(SNIPPET_DUR_MULTIPLE * float(t_dur), SNIPPET_MIN_HALFWIDTH)
        sel = np.abs(time - t_event) <= half
        return {
            "time": time[sel].astype(np.float64),
            "flux": flux[sel].astype(np.float64),
            "flux_err": flux_err[sel].astype(np.float64),
            "halfwidth": half,
        }

    # ------------------------------------------------------------------
    def process_file(self, file_path, flux_model, depth_model, duration_model,
                     model_idx=-1, snippet_dir=None, partner=None):
        """One injection test. Mirrors the stock body with the three deltas.

        Args:
            partner: optional dict ``{"dt": signed days, "ratio": depth ratio}``.
                When given, a SECOND transit is injected at ``lead + dt`` with
                delta-flux scaled by ``ratio``, reproducing the 1-2 punch: one
                conjunction, the planet crossing star 1 then star 2. The partner
                is skipped, and recorded as skipped, if it would fall outside the
                light curve; that is itself physical, since a real partner
                transit can land in a gap or after the sector ends.
        """
        file = os.path.basename(file_path)
        split_file = os.path.splitext(file)

        if split_file[1] == ".npz":
            data = np.load(file_path, allow_pickle=True)
            time = data[self.npz_keys["time"]]
            flux = data[self.npz_keys["flux"]]
            flux_err = data[self.npz_keys["flux_err"]]
            phase = data.get("phase", None)
            ecl_mask_raw = data.get("eclipse_mask", None)
            ecl_mask = ecl_mask_raw.astype(bool) if ecl_mask_raw is not None else None
        elif split_file[1] == ".txt":
            data = np.loadtxt(file_path, skiprows=1)
            if data.ndim != 2 or data.shape[0] < 2:
                return None
            time, flux, flux_err, phase = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            ecl_mask = data[:, 4].astype(bool) if data.shape[1] > 4 else None
        else:
            return None

        tic, sector = self._parse_filename(file)
        if tic is None:
            return None

        # Stock re-binning branch, kept verbatim in behaviour.
        finite = ~np.isnan(flux)
        if np.median(np.gradient(time[finite])) < self.transit_config["cadence_minutes"] / (60 * 24):
            time, flux, flux_err = bin_to_long_cadence(time, flux, flux_err)
            if self.catalogue is not None:
                row = self.catalogue[self.catalogue["tess_id"] == tic]
                if not row.empty:
                    phase = time_to_phase(time, row["period"].values[0], row["bjd0"].values[0])
                    prim = get_eclipse_mask(phase, row["prim_pos"].values[0], row["prim_width"].values[0])
                    sec = get_eclipse_mask(phase, row["sec_pos"].values[0], row["sec_width"].values[0])
                    ecl_mask = np.logical_or(prim, sec)

        nan_mask = ~np.isnan(flux * time * flux_err)
        mask = nan_mask & ~ecl_mask if ecl_mask is not None else nan_mask

        # ---- DELTA 1: inversion becomes a per-test coin flip -------------
        inverted = bool(self.rng.random() < self.p_invert)
        if inverted:
            flux = ((flux - 1) * -1) + 1

        # ---- DELTA 2: native path masks the real search's own events -----
        epoch_mask = mask
        n_blocked = 0
        if not inverted:
            epoch_mask, n_blocked = self._mask_real_events(time, mask, tic, sector, duration_model)

        mask_indices = np.where(epoch_mask)[0]
        if len(mask_indices) == 0:
            logger.debug("%s: no eligible cadences after masking", file)
            return None

        # ---- DELTA 3: the epoch draw is injectable ------------------------
        meta = {"tic": tic, "sector": sector, "file": file, "inverted": inverted,
                "duration_model": duration_model, "depth_model": depth_model}
        inj_time = self.epoch_sampler(time, mask_indices, phase, meta)

        flux_inj, inj_time = self._inject_transit(time, flux, flux_model, inj_time)

        # ---- the 1-2 punch: optional partner transit on the other star ----
        partner_time = np.nan
        partner_ratio = np.nan
        partner_injected = 0
        if partner is not None:
            want = inj_time + float(partner["dt"])
            # Require the partner centre to sit on real data, not merely inside
            # the time span: a centre inside a gap would be un-recoverable by
            # construction and would silently depress the paired recovery rate.
            on_data = np.abs(time - want) < CADENCE_DAYS
            if on_data.any():
                partner_ratio = float(partner["ratio"])
                # The partner must differ in DURATION as well as depth. The old
                # code injected flux_model * ratio, i.e. the same array scaled,
                # so every bank pair had a duration ratio of exactly 1.000 while
                # ELC's real pairs sit near 0.116. That single column would have
                # let probe 2 - a FREEZE GATE - separate bank from ELC positives
                # perfectly. Compressing the model in time reproduces the
                # physical fact that the partner crosses a smaller star on a
                # different chord.
                dur_ratio = float(partner.get("duration_ratio", 1.0))
                pm = flux_model * partner_ratio
                if abs(dur_ratio - 1.0) > 1e-6:
                    n = len(pm)
                    src = np.linspace(-1.0, 1.0, n)
                    pm = np.interp(src, src * dur_ratio, pm, left=0.0, right=0.0)
                flux_inj, partner_time = self._inject_transit(
                    time, flux_inj, pm, want
                )
                partner_injected = 1

        detrend_result = detrend(
            time, flux_inj, flux_err, method=self.transit_config["detrending_method"],
            fname=split_file[0], mask=mask,
            edge_cutoff=self.transit_config["edge_cutoff"],
            cos_win_len_max=self.transit_config["cosine"]["win_len_max"],
            cos_win_len_min=self.transit_config["cosine"]["win_len_min"],
            fap_threshold=self.transit_config["cosine"]["fap_threshold"],
            poly_order=self.transit_config["cosine"]["poly_order"],
            max_splines=self.transit_config["pspline"]["max_splines"],
            bi_win_len_max=self.transit_config["biweight"]["win_len_max"],
            bi_win_len_min=self.transit_config["biweight"]["win_len_min"],
        )

        # ---- event extraction via the REAL SEARCH's own code path ---------
        self._finder._clear_results()
        try:
            self._finder._process_cb_events(
                time, flux_inj, flux_err, phase, ecl_mask, mask,
                detrend_result, tic, sector, split_file[0], None,
            )
        except Exception as exc:  # one bad file must not kill a 60k campaign
            logger.warning("%s: event extraction failed (%s)", file, type(exc).__name__)
            return None

        res = self._finder.results
        n_events = len(res["event_times"])
        half_dur = duration_model / 2.0

        recovered = False
        recovered_partner = False
        for i in range(n_events):
            t_ev = float(res["event_times"][i])
            # The t_dur/2 recovery rule, applied to each injected transit
            # separately so a pair can be half-recovered. That is the honest
            # outcome and it is exactly what the depth-ratio screen predicts:
            # the shallower partner is often below the noise.
            is_lead = abs(t_ev - inj_time) < half_dur
            is_partner = partner_injected and abs(t_ev - partner_time) < half_dur
            is_injection = bool(is_lead or is_partner)
            recovered |= is_lead
            recovered_partner |= bool(is_partner)
            row = {
                "run_id": self.run_id,
                "provenance": self.provenance,
                "tic": int(tic),
                "sector": int(sector),
                "model_idx": int(model_idx),
                "inverted_lc": int(inverted),
                "event_time": t_ev,
                "phase": float(res["event_phases"][i]),
                "depth": float(res["event_depths"][i]),
                "duration": float(res["event_durations"][i]),
                "snr": float(res["event_snrs"][i]),
                "win_len": float(res["win_len_max_SNR"][i]),
                "det_dependence": int(res["det_dependence"][i]),
                "n_detrend_detections": int(res["n_detrend_detections"][i]),
                # injected truth, present on every row so the label is auditable
                "inj_time": float(inj_time),
                "inj_depth": float(depth_model),
                "inj_duration": float(duration_model),
                "label": int(is_injection),
                # 2026-08-06 semantic fix: 'lead' requires an actually-injected
                # partner. Before this, every recovered unpaired injection was
                # stamped 'lead', which made pair_role a copy of the label; the
                # exporter no longer consumes pair_role as a feature at all
                # (audit column truth_pair_role only), but the run artifact
                # should still mean what its name says.
                "pair_role": ("lead" if (is_lead and partner_injected)
                              else ("partner" if is_partner else "none")),
                "pair_id": f"{self.run_id}_{tic}_{sector}_{model_idx}" if partner_injected else "",
                "partner_dt": float(partner["dt"]) if partner is not None else np.nan,
                "partner_ratio": partner_ratio,
                "partner_duration_ratio": (float(partner.get("duration_ratio", np.nan))
                                           if partner is not None else np.nan),
                "n_cadences_blocked": n_blocked,
            }
            self.event_rows.append(row)

            if snippet_dir is not None:
                snip = self._snippet(time, flux_inj, flux_err, t_ev, row["duration"])
                name = (f"{self.run_id}_{self.provenance}_TIC{tic}_S{sector}"
                        f"_m{model_idx}_e{i}.npz")
                np.savez(os.path.join(snippet_dir, name), label=row["label"],
                         inverted_lc=int(inverted), tic=tic, sector=sector,
                         event_time=t_ev, **snip)
                row["snippet"] = name

        self.test_rows.append({
            "run_id": self.run_id,
            "provenance": self.provenance,
            "tic": int(tic),
            "sector": int(sector),
            "model_idx": int(model_idx),
            "inverted_lc": int(inverted),
            "inj_time": float(inj_time),
            "inj_depth": float(depth_model),
            "inj_duration": float(duration_model),
            "recovered": int(recovered),
            "partner_requested": int(partner is not None),
            "partner_injected": partner_injected,
            "partner_time": partner_time,
            "partner_dt": float(partner["dt"]) if partner is not None else np.nan,
            "partner_ratio": partner_ratio,
            "partner_duration_ratio": (float(partner.get("duration_ratio", np.nan))
                                       if partner is not None else np.nan),
            "recovered_partner": int(recovered_partner),
            "recovered_pair": int(recovered and recovered_partner),
            "n_events_flagged": n_events,
            "n_cadences_blocked": n_blocked,
        })
        return recovered

    # ------------------------------------------------------------------
    def inject_fixed_events(self, file_path, elc_events, model_meta=None,
                            snippet_dir=None):
        """Inject a set of transits at TIMES ELC computed, rather than at a drawn epoch.

        This is the second entry point, used for ELC-generated positives. The
        difference from ``process_file`` is that nothing is sampled: ELC has
        already decided when each transit happens, how deep it is, how long it
        lasts, and crucially WHICH STAR was crossed. So the whole set of a
        model's transits goes in together, the light curve is detrended once and
        searched once, and each transit is labelled separately.

        Labelling each separately is the honest choice and it matters for the
        pair analysis: a conjunction can easily be half-recovered, with the deep
        crossing of the bright star found and the shallow crossing of the faint
        one missed. Collapsing that to one label per model would hide exactly the
        effect the depth-ratio screen exists to measure.

        Args:
            elc_events: list of dicts with ``time``, ``depth``, ``duration`` and
                ``star``; ``cycle`` when available, so pairs can be grouped.
        """
        file = os.path.basename(file_path)
        split_file = os.path.splitext(file)
        data = np.loadtxt(file_path, skiprows=1)
        if data.ndim != 2 or data.shape[0] < 2:
            return None
        time, flux, flux_err, phase = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        ecl_mask = data[:, 4].astype(bool) if data.shape[1] > 4 else None

        tic, sector = self._parse_filename(file)
        if tic is None:
            return None

        nan_mask = ~np.isnan(flux * time * flux_err)
        mask = nan_mask & ~ecl_mask if ecl_mask is not None else nan_mask

        # Same 50/50 rule as the bank arm: inversion must carry no label
        # information for ELC positives either, or probe 1 fails by provenance.
        inverted = bool(self.rng.random() < self.p_invert)
        if inverted:
            flux = ((flux - 1) * -1) + 1

        # Keep only transits that land on real data. A transit in a gap is not a
        # failure, it is a real observational outcome, but it cannot be labelled
        # recovered or missed so it must not enter the denominator.
        usable = []
        flux_inj = flux.copy()
        for ev in elc_events:
            t_ev = float(ev["time"])
            dur = max(float(ev.get("duration", 0.2)), 2 * CADENCE_DAYS)
            on_data = np.abs(time - t_ev) < CADENCE_DAYS
            if not on_data.any():
                continue
            # Analytic segment shaped like the ELC transit: a smooth dip of the
            # right depth and duration, added as delta-flux.
            x = (time - t_ev) / (dur / 2.0)
            seg = np.where(np.abs(x) <= 1.0,
                           -float(ev["depth"]) * (1.0 - 0.35 * x ** 2), 0.0)
            flux_inj = flux_inj + seg
            usable.append(ev)

        if not usable:
            return None

        detrend_result = detrend(
            time, flux_inj, flux_err, method=self.transit_config["detrending_method"],
            fname=split_file[0], mask=mask,
            edge_cutoff=self.transit_config["edge_cutoff"],
            cos_win_len_max=self.transit_config["cosine"]["win_len_max"],
            cos_win_len_min=self.transit_config["cosine"]["win_len_min"],
            fap_threshold=self.transit_config["cosine"]["fap_threshold"],
            poly_order=self.transit_config["cosine"]["poly_order"],
            max_splines=self.transit_config["pspline"]["max_splines"],
            bi_win_len_max=self.transit_config["biweight"]["win_len_max"],
            bi_win_len_min=self.transit_config["biweight"]["win_len_min"],
        )

        self._finder._clear_results()
        try:
            self._finder._process_cb_events(
                time, flux_inj, flux_err, phase, ecl_mask, mask,
                detrend_result, tic, sector, split_file[0], None)
        except Exception as exc:
            logger.warning("%s: ELC event extraction failed (%s)", file, type(exc).__name__)
            return None

        res = self._finder.results
        meta = model_meta or {}
        found = np.zeros(len(usable), dtype=bool)

        for i in range(len(res["event_times"])):
            t_flag = float(res["event_times"][i])
            matched = -1
            for j, ev in enumerate(usable):
                if abs(t_flag - float(ev["time"])) < max(float(ev.get("duration", 0.2)), 2 * CADENCE_DAYS) / 2.0:
                    matched = j
                    found[j] = True
                    break
            ev = usable[matched] if matched >= 0 else {}
            row = {
                "run_id": self.run_id, "provenance": "elc",
                "tic": int(tic), "sector": int(sector),
                "model_idx": meta.get("model_id", -1),
                "inverted_lc": int(inverted),
                "event_time": t_flag,
                "phase": float(res["event_phases"][i]),
                "depth": float(res["event_depths"][i]),
                "duration": float(res["event_durations"][i]),
                "snr": float(res["event_snrs"][i]),
                "win_len": float(res["win_len_max_SNR"][i]),
                "det_dependence": int(res["det_dependence"][i]),
                "n_detrend_detections": int(res["n_detrend_detections"][i]),
                "inj_time": float(ev["time"]) if matched >= 0 else np.nan,
                "inj_depth": float(ev["depth"]) if matched >= 0 else np.nan,
                "inj_duration": float(ev.get("duration", np.nan)) if matched >= 0 else np.nan,
                "label": int(matched >= 0),
                "elc_star": int(ev.get("star", 0)) if matched >= 0 else 0,
                "elc_cycle": int(ev.get("cycle", -1)) if matched >= 0 else -1,
                "sigma": meta.get("sigma", np.nan),
                "pair_role": "none", "pair_id": "", "partner_dt": np.nan,
                "partner_ratio": np.nan, "n_cadences_blocked": 0,
            }
            self.event_rows.append(row)
            if snippet_dir is not None:
                snip = self._snippet(time, flux_inj, flux_err, t_flag, row["duration"])
                name = f"{self.run_id}_elc_TIC{tic}_S{sector}_{meta.get('model_id','m')}_e{i}.npz"
                np.savez(os.path.join(snippet_dir, name), label=row["label"],
                         inverted_lc=int(inverted), tic=tic, sector=sector,
                         event_time=t_flag, **snip)
                row["snippet"] = name

        # One test row per INJECTED transit, so the denominator is transits.
        for j, ev in enumerate(usable):
            self.test_rows.append({
                "run_id": self.run_id, "provenance": "elc",
                "tic": int(tic), "sector": int(sector),
                "model_idx": meta.get("model_id", -1),
                "inverted_lc": int(inverted),
                "inj_time": float(ev["time"]), "inj_depth": float(ev["depth"]),
                "inj_duration": float(ev.get("duration", np.nan)),
                "elc_star": int(ev.get("star", 0)),
                "elc_cycle": int(ev.get("cycle", -1)),
                "sigma": meta.get("sigma", np.nan),
                "recovered": int(found[j]),
                "n_events_flagged": len(res["event_times"]),
                "partner_requested": 0, "partner_injected": 0,
                "partner_time": np.nan, "partner_dt": np.nan,
                "partner_ratio": np.nan, "recovered_partner": 0,
                "recovered_pair": 0, "n_cadences_blocked": 0,
            })
        return int(found.sum())

    # ------------------------------------------------------------------
    def events_frame(self):
        return pd.DataFrame(self.event_rows)

    def tests_frame(self):
        return pd.DataFrame(self.test_rows)
