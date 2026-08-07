"""Fixture 2 for the one-shot: the export-host and Kepler-host code paths.

Trap recorded on first run: the frozen exporter zero-pads sectors in staged
filenames (TIC_x_05.txt); a fixture without the padding builds zero records.

Why this exists (punch-list item 1, 2026-08-07): the original fixture
declares both host lists empty, so the most complex code in 26_one_shot.py
(the frozen-exporter pass over a host's staged files, and the per-event
cycle-subsampled Kepler leg) had ZERO executed coverage and would have run
for the first time ever on the real Aug-11 data. This fixture builds one
fully synthetic host of each kind and runs both paths end to end, including
the comparator, write_shard, expected_rows asserts, D1 matching on exported
pools, and the C2 regression check: the capped Kepler export's
n_cycles-bearing scalars must DIFFER from the uncapped secondary's (the
defect the reviewers proved was silently absent before the per-event
staging fix).

Everything here is synthetic: fake TICs (900000011 TESS-like, 900000012
Kepler-like), generated light curves with hand-placed box dips, a synthetic
events file in the real detected_events.txt header format, and catalogue
rows in config form. No real data, no sealed contact.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "data", "oneshot", "fixtures_hosts")
SEED = 20260812

TIC_E = 900000011      # "TESS" export host
TIC_K = 900000012      # "Kepler" host (quarter-style files)
HDR = ("TIME FLUX FLUX_ERR PHASE\n")
EV_HDR = ("TIC SECTOR TIME PHASE DEPTH DURATION SNR WIN_LEN_MAX_SNR "
          "DET_DEPENDENCE SKYE_FLAG N_DETREND_DETECTIONS\n")


def write_lc(path, t0, t1, period, dips, rng, cad=0.02):
    """A synthetic masked light curve: flat noise + box dips at given times."""
    t = np.arange(t0, t1, cad)
    f = 1.0 + rng.normal(0, 3e-4, size=len(t))
    for td, depth, dur in dips:
        f[np.abs(t - td) <= dur / 2] -= depth
    ph = ((t - t0) / period) % 1.0
    err = np.full(len(t), 3e-4)
    with open(path, "w") as fh:
        fh.write(HDR)
        np.savetxt(fh, np.column_stack([t, f, err, ph]), fmt="%.8f")


def write_events(path, rows):
    with open(path, "w") as fh:
        fh.write(EV_HDR)
        for r in rows:
            fh.write(" ".join(str(x) for x in r) + "\n")


def main():
    os.makedirs(FIX, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- export host: one sector file, 2 events (1 matches a known time) --
    e_lc = os.path.join(FIX, "lc_export")
    os.makedirs(e_lc, exist_ok=True)
    p_e = 7.0
    write_lc(os.path.join(e_lc, f"TIC_{TIC_E}_05.txt"), 100.0, 127.0, p_e,
             [(105.0, 0.004, 0.25), (118.0, 0.003, 0.2)], rng)
    ev_e = os.path.join(FIX, f"events_{TIC_E}.txt")
    write_events(ev_e, [
        (TIC_E, 5, 105.0, 0.71, 0.004, 0.25, 9.1, 3.0, 0, 0, 20),
        (TIC_E, 5, 118.0, 0.57, 0.003, 0.20, 7.4, 3.0, 0, 0, 19),
    ])

    # ---- kepler host: two "quarter" files, 3 events, ~40 cycles of period 2.0
    k_lc = os.path.join(FIX, "lc_kepler")
    os.makedirs(k_lc, exist_ok=True)
    p_k = 2.0
    write_lc(os.path.join(k_lc, f"TIC_{TIC_K}_01.txt"), 200.0, 240.0, p_k,
             [(205.0, 0.005, 0.15), (221.0, 0.004, 0.15)], rng)
    write_lc(os.path.join(k_lc, f"TIC_{TIC_K}_02.txt"), 240.0, 280.0, p_k,
             [(262.0, 0.004, 0.15)], rng)
    ev_k = os.path.join(FIX, f"events_{TIC_K}.txt")
    write_events(ev_k, [
        (TIC_K, 1, 205.0, 0.5, 0.005, 0.15, 8.0, 3.0, 0, 0, 21),
        (TIC_K, 1, 221.0, 0.5, 0.004, 0.15, 7.0, 3.0, 0, 0, 20),
        (TIC_K, 2, 262.0, 0.0, 0.004, 0.15, 7.2, 3.0, 0, 0, 20),
    ])

    # ---- times: export host 1 of 2 events at a known time (H7 rule);
    #      kepler host 2 of 3 (one out-of-window row filtered)
    import csv
    times_t = os.path.join(FIX, "times_tess.csv")
    with open(times_t, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["planet", "tic", "t_transit_btjd",
                                           "t_unc_days", "in_our_cache_window"])
        w.writeheader()
        w.writerow({"planet": "FIXE b", "tic": TIC_E, "t_transit_btjd": 105.06,
                    "t_unc_days": 0.0, "in_our_cache_window": "yes"})
    times_k = os.path.join(FIX, "times_kepler.csv")
    with open(times_k, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["planet", "tic", "t_transit_btjd",
                                           "t_unc_days", "in_our_cache_window"])
        w.writeheader()
        w.writerow({"planet": "FIXK b", "tic": TIC_K, "t_transit_btjd": 205.02,
                    "t_unc_days": 0.0, "in_our_cache_window": "yes"})
        w.writerow({"planet": "FIXK b", "tic": TIC_K, "t_transit_btjd": 262.05,
                    "t_unc_days": 0.0, "in_our_cache_window": "yes"})
        w.writerow({"planet": "FIXK b", "tic": TIC_K, "t_transit_btjd": 999.0,
                    "t_unc_days": 0.0, "in_our_cache_window": "no (fixture)"})

    import hashlib
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    real_hlog_path = os.path.join(REPO, "data", "bench", "full",
                                  "preshot_hashlog_2026-08-07.json")
    real_hlog = json.load(open(real_hlog_path))

    cat_e = {"tess_id": TIC_E, "period": p_e, "bjd0": 100.0, "prim_pos": 1.0,
             "prim_width": 0.01, "sec_pos": 0.5, "sec_width": 0.01,
             "Tmag": 10.0, "morph_coeff": 0.15, "sectors": "5"}
    cat_k = {"tess_id": TIC_K, "period": p_k, "bjd0": 200.0, "prim_pos": 1.0,
             "prim_width": 0.01, "sec_pos": 0.5, "sec_width": 0.01,
             "Tmag": 11.0, "morph_coeff": 0.15, "sectors": "1,2"}

    cfg = {
        "run_label": "FIXTURE2_HOSTS_NOT_REAL",
        "out_dir": os.path.join(FIX, "out"),
        "repo_root": REPO,
        "hash_log": real_hlog_path,
        "tau": real_hlog["operating_threshold"]["value"],
        "times_tables": [
            {"mission": "TESS", "path": times_t, "sha256": sha(times_t)},
            {"mission": "Kepler", "path": times_k, "sha256": sha(times_k)}],
        "frozen_shard": os.path.join(REPO, "data", "oneshot", "fixtures",
                                     "fixture_shard.h5"),
        "frozen_shard_hosts": [],
        "export_hosts": [
            {"name": "FIXE", "mission": "TESS", "kind": "planet",
             "events": ev_e, "events_sha256": sha(ev_e),
             "staged_lc": e_lc, "catalogue_row": cat_e,
             "use_noise": False, "expected_rows": 2}],
        "kepler_hosts": [
            {"name": "FIXK", "mission": "Kepler", "kind": "planet",
             "events": ev_k, "events_sha256": sha(ev_k),
             "staged_lc": k_lc, "catalogue_row": cat_k,
             "p_bin": p_k, "subsample_seed": 7, "use_noise": False,
             "expected_rows": 3}],
        "tess_train_n_cycles": [2, 3, 4],   # far below the ~40 available cycles
        "declared_tier": "FIXTURE2",
        "in_window_only": True,
    }
    cfg_path = os.path.join(FIX, "fixture2_config.json")
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    expected = {
        "FIXE": {"D2": 1, "D1": 1},
        "FIXK": {"D2": 2, "D1": 2},
        "notes": "pass counts not pre-pinned (synthetic features, real model); "
                 "the test asserts structure, counts, and the C2 regression: "
                 "capped vs uncapped n_cycles scalars must differ.",
    }
    json.dump(expected, open(os.path.join(FIX, "fixture2_expected.json"), "w"),
              indent=2)
    print(f"fixture2 written under {FIX}")


if __name__ == "__main__":
    main()
