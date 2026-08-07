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
