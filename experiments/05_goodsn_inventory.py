#!/usr/bin/env python
"""
05_goodsn_inventory.py -- coverage inventory for all 1,020 good-S/N systems.

Inventory-first (no downloads): one MAST query per TIC recording every
available sector and pipeline product, so the bulk pull can be sized before
committing disk and bandwidth.

Usage: conda activate mono-cbp && python experiments/05_goodsn_inventory.py
Resumable: appends to the CSV and skips TICs already inventoried.
"""
from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import lightkurve as lk

DATA = Path(os.environ.get("CBP_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
TICS = DATA / "goodsn_tics.txt"
OUT = DATA / "goodsn_pilot" / "inventory_1020.csv"
FIELDS = ["tic", "n_sectors", "sectors", "n_tess_spoc", "n_qlp", "n_spoc2min", "max_sector"]


def main() -> None:
    tics = [int(l) for l in open(TICS) if l.strip()]
    done = set()
    if OUT.exists():
        with open(OUT) as fh:
            done = {int(r["tic"]) for r in csv.DictReader(fh)}
    mode = "a" if done else "w"
    with open(OUT, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for i, tic in enumerate(tics):
            if tic in done:
                continue
            try:
                tbl = lk.search_lightcurve(f"TIC {tic}", mission="TESS").table
                secs, auth = {}, {"TESS-SPOC": 0, "QLP": 0, "SPOC": 0}
                for r in tbl:
                    try:
                        s = int(str(r["mission"]).split()[-1])
                    except (ValueError, IndexError):
                        continue
                    a = str(r["author"])
                    secs.setdefault(s, set()).add(a)
                    if a in auth:
                        auth[a] += 1
                w.writerow({"tic": tic, "n_sectors": len(secs),
                            "sectors": ";".join(map(str, sorted(secs))),
                            "n_tess_spoc": auth["TESS-SPOC"], "n_qlp": auth["QLP"],
                            "n_spoc2min": auth["SPOC"],
                            "max_sector": max(secs) if secs else 0})
            except Exception as e:
                w.writerow({"tic": tic, "n_sectors": -1, "sectors": f"ERR:{type(e).__name__}",
                            "n_tess_spoc": 0, "n_qlp": 0, "n_spoc2min": 0, "max_sector": 0})
            if (i + 1) % 50 == 0:
                fh.flush()
                print(f"{i+1}/{len(tics)} inventoried", flush=True)
    print("inventory complete:", OUT)


if __name__ == "__main__":
    main()
