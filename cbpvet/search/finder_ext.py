"""TransitFinder subclass that persists the detrending-window group size.

Why this file exists
--------------------
mono-cbp groups the peaks found across the 21 biweight detrending windows and
keeps only a binary summary of how large each group was::

    det_dep = 1 if len(group) <= det_dependence_threshold else 0   # finder.py:492

``len(group)`` itself, the number of detrending windows in which the event was
independently detected, is discarded. That count is a strong vetting feature: a
real astrophysical dip survives most windows, a detrending artefact survives
few. The search over the frozen 1,946 light curves is a never-re-run product,
so if the count is not captured on that run it is unavailable for every
real-search negative for the life of the benchmark.

mono-cbp originals must not be edited, so ``_process_cb_events`` is copied
verbatim from the installed source with exactly one line added (marked
CBP-VET). ``tests/test_finder_ext_parity.py`` asserts the copy still reproduces
the stock finder's event table column for column.

Pinned against mono-cbp git 891af27 (installed v0.1.9).
"""

import logging
import os

import numpy as np

from mono_cbp.transit_finding.finder import (
    EVENT_GROUPING_TOLERANCE,
    VAR_MAD_WINDOW,
    TransitFinder,
)
from mono_cbp.utils import get_var_mad, monofind, split_tol
from mono_cbp.utils.plotting import plot_event, plot_no_events

logger = logging.getLogger("cbpvet.search")

EXTRA_COLUMN = "N_DETREND_DETECTIONS"


class TransitFinderExt(TransitFinder):
    """TransitFinder that also records ``n_detrend_detections`` per event.

    The extra count is appended as a trailing column of the output file so that
    every existing mono-cbp reader of ``detected_events.txt`` keeps working on
    the columns it already knows.
    """

    @staticmethod
    def _initialise_results():
        results = TransitFinder._initialise_results()
        results["n_detrend_detections"] = []
        return results

    def process_directory(self, data_dir, output_file="output.txt", output_dir=None,
                          plot_output_dir=None):
        """Run the stock pipeline, then append the extra column to the output file.

        The parent sorts its output with ``np.lexsort((time, tic))`` before
        writing. That permutation is reproduced here from the same result lists
        so the appended column stays aligned row for row.
        """
        df = super().process_directory(
            data_dir, output_file=output_file, output_dir=output_dir,
            plot_output_dir=plot_output_dir,
        )

        n_detrend = self.results["n_detrend_detections"]
        if len(n_detrend) == 0:
            logger.warning("No events detected; no extra column written")
            return df

        # Reproduce finder._save_results' sort exactly: primary key tic (as the
        # string it was stored as), secondary key event time (as float).
        tics = np.array([str(t) for t in self.results["tics"]])
        times = np.asarray(self.results["event_times"], dtype=float)
        sort_idx = np.lexsort((times, tics))
        n_sorted = np.asarray(n_detrend, dtype=int)[sort_idx]

        path = os.path.join(output_dir or os.getcwd(), output_file)
        with open(path) as fh:
            lines = fh.read().rstrip("\n").split("\n")
        header, body = lines[0], lines[1:]
        if len(body) != len(n_sorted):
            raise RuntimeError(
                f"Row-count mismatch appending {EXTRA_COLUMN}: file has {len(body)} "
                f"rows, finder recorded {len(n_sorted)} events"
            )
        with open(path, "w") as fh:
            fh.write(f"{header} {EXTRA_COLUMN}\n")
            for line, n in zip(body, n_sorted):
                fh.write(f"{line} {n}\n")
        logger.info("Appended %s to %s (%d rows)", EXTRA_COLUMN, path, len(body))

        df[EXTRA_COLUMN.lower()] = n_sorted
        return df

    # ------------------------------------------------------------------
    # Verbatim copy of TransitFinder._process_cb_events (finder.py 404-516)
    # at mono-cbp git 891af27. The ONLY change is the line marked CBP-VET.
    # Do not refactor: parity with the original is asserted by test.
    # ------------------------------------------------------------------
    def _process_cb_events(self, time, flux, flux_err, phase, ecl_mask, mask,
                           detrend_result, tic, sector, fname, plot_output_dir):
        flatten_lcs, trend_lcs, bi_win_lens, _ = detrend_result
        mad = self.transit_config['mad_threshold']

        # Storage for all detected events across all biweight windows
        all_peaks = []
        event_data_all = []

        # Loop over biweight windows
        for index, lc in enumerate(flatten_lcs):
            win_len = bi_win_lens[index]
            var_mad = get_var_mad(lc, VAR_MAD_WINDOW)
            peaks, meta = monofind(time[mask], lc, mad=mad, var_mad=var_mad)
            all_peaks.append(peaks)

            # Extract event data for each peak
            for j in range(len(peaks)):
                event_data = self._extract_event_data(
                    time, flux_err, phase, lc, mask,
                    peaks[j], meta, j, tic, sector
                )
                event_data['win_len'] = win_len
                event_data['flat_lc_idx'] = index
                event_data['var_mad'] = var_mad
                event_data_all.append(event_data)

        # Flatten peaks list
        all_peaks_flat = [p for ps in all_peaks for p in ps]

        if len(all_peaks_flat) == 0:
            # Plot no events if requested
            if self.transit_config['generate_vetting_plots'] and plot_output_dir:
                plot_no_events(time, flatten_lcs[-1], flux, flux_err, trend_lcs[-1],
                               fname, mad=mad, var_mad=get_var_mad(flatten_lcs[-1], VAR_MAD_WINDOW),
                               ecl_mask=ecl_mask, mask=mask, output_dir=plot_output_dir)
            return []

        # Group events detected at similar times across different window lengths
        all_peaks_flat_sorted = np.sort(all_peaks_flat)
        time_sorted_idx = np.argsort([e['time'] for e in event_data_all])
        event_data_sorted = [event_data_all[i] for i in time_sorted_idx]

        events_grouped = split_tol(all_peaks_flat_sorted, EVENT_GROUPING_TOLERANCE)

        # Select highest SNR event from each group
        # NOTE: All events are saved, filtering can be done later with filter_events()
        events = []
        start_idx = 0
        for group in events_grouped:
            group_events = event_data_sorted[start_idx:start_idx+len(group)]
            snrs = [e['snr'] for e in group_events]
            max_snr_idx = np.argmax(snrs)
            best_event = group_events[max_snr_idx]

            # Calculate detrending dependence flag
            det_dep = 1 if len(group) <= self.transit_config['filters']['det_dependence_threshold'] else 0

            # Save ALL events (no filtering)
            events.append(best_event)
            self.results['tics'].append(str(tic))
            self.results['sectors'].append(str(sector))
            self.results['event_times'].append(best_event['time'])
            self.results['event_phases'].append(best_event['phase'])
            self.results['event_depths'].append(best_event['depth'])
            self.results['event_durations'].append(best_event['duration'])
            self.results['event_snrs'].append(best_event['snr'])
            self.results['win_len_max_SNR'].append(round(best_event['win_len'], 1))
            self.results['det_dependence'].append(det_dep)
            self.results['n_detrend_detections'].append(len(group))  # CBP-VET

            # Save event snippet if requested
            if self.transit_config['generate_event_snippets']:
                snippet = self._create_event_snippet(
                    time, flux_err, flatten_lcs[best_event['flat_lc_idx']],
                    mask, best_event, tic, sector, len(events)
                )
                self.results['event_snippets'].append(snippet)

            # Plot ONLY if event passes filters
            if self.transit_config['generate_vetting_plots'] and plot_output_dir:
                flat_lc_idx = best_event['flat_lc_idx']
                peaks = all_peaks[flat_lc_idx]
                plot_event(time, best_event['time'], flatten_lcs[flat_lc_idx],
                           flux, flux_err, trend_lcs[flat_lc_idx], fname, mad,
                           best_event['var_mad'], best_event['depth'],
                           best_event['duration'], best_event['phase'],
                           best_event['snr'], peaks, len(events),
                           ecl_mask=ecl_mask, mask=mask, output_dir=plot_output_dir)

            start_idx += len(group)

        return events
