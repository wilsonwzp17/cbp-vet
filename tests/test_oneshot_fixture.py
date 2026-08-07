"""The 3B.10-iv pre-committed fixture anchor for the one-shot script.

Runs experiments/26_one_shot.py end-to-end against the synthetic fixtures
(the ONLY permitted execution mode before the real Aug-11 run) and diffs the
D1/D2/pass counts against fixture_expected.json. Any disagreement is a bug
by the pre-registered definition. Skips cleanly if the fixtures have not
been generated on this machine.
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "data", "oneshot", "fixtures")


@pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "fixture_config.json")),
                    reason="fixtures not generated (run experiments/26_fixtures.py)")
def test_oneshot_reproduces_fixture_expectations(tmp_path):
    cfg = json.load(open(os.path.join(FIX, "fixture_config.json")))
    cfg["out_dir"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.json"
    json.dump(cfg, open(cfg_path, "w"))

    r = subprocess.run([sys.executable,
                        os.path.join(REPO, "experiments", "26_one_shot.py"),
                        "--config", str(cfg_path)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]

    out = json.load(open(tmp_path / "oneshot_results.json"))
    exp = json.load(open(os.path.join(FIX, "fixture_expected.json")))
    for h in out["per_host"]:
        e = exp[h["host"]]
        assert h["D2_known_times_in_windows"] == e["D2"], h["host"]
        assert h["D1_flagged_at_known_times"] == e["D1"], h["host"]
        assert h["D1_vetter_pass"] == e["D1_pass"], h["host"]
    # the candidate must not enter the mission rollup
    assert out["per_mission"]["TESS"]["D1"] == exp["FIX-A"]["D1"]


@pytest.mark.skipif(not os.path.exists(os.path.join(
        REPO, "data", "oneshot", "fixtures_hosts", "fixture2_config.json")),
    reason="host fixtures not generated (run experiments/26_fixtures_hosts.py)")
def test_oneshot_export_and_kepler_paths(tmp_path):
    """Fixture 2: the export-host and Kepler-subsample paths + C2 regression."""
    import h5py
    import numpy as np

    fix2 = os.path.join(REPO, "data", "oneshot", "fixtures_hosts")
    cfg = json.load(open(os.path.join(fix2, "fixture2_config.json")))
    cfg["out_dir"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.json"
    json.dump(cfg, open(cfg_path, "w"))

    r = subprocess.run([sys.executable,
                        os.path.join(REPO, "experiments", "26_one_shot.py"),
                        "--config", str(cfg_path)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]

    out = json.load(open(tmp_path / "oneshot_results.json"))
    exp = json.load(open(os.path.join(fix2, "fixture2_expected.json")))
    hosts = {h["host"]: h for h in out["per_host"]}
    for name in ("FIXE", "FIXK"):
        assert hosts[name]["D2_known_times_in_windows"] == exp[name]["D2"], name
        assert hosts[name]["D1_flagged_at_known_times"] == exp[name]["D1"], name
    assert hosts["FIXK_uncapped"]["kind"] == "secondary"
    assert out["per_mission"]["Kepler"]["D1"] == exp["FIXK"]["D1"]
    assert "FIXK" in out["kepler_ncycles_ks"]

    # The C2 regression: the capped export's cycle counts must be genuinely
    # smaller than the uncapped secondary's (the subsample must not be a no-op).
    with h5py.File(tmp_path / "oneshot_FIXK.h5") as f:
        capped = f["scalars"]["n_cycles_raw"][:].astype(float)
    with h5py.File(tmp_path / "oneshot_FIXK_uncapped.h5") as f:
        uncapped = f["scalars"]["n_cycles_raw"][:].astype(float)
    assert (capped < uncapped).all(), (capped, uncapped)
