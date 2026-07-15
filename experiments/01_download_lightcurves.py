#!/usr/bin/env python
"""
01_download_lightcurves.py -- resumable bulk download of the EB-sample light curves.

Wraps mono-cbp's own catalogue_to_lc_files, which SKIPS files already present, so
re-running resumes where it left off and retries only what failed. Downloads the
crowding-corrected SAP light curves (TESS-SPOC) for every system/sector in the
591-EB parent catalogue, into a git-ignored cache.

Paths are resolved from environment variables with sensible fallbacks (no hardcoded
home path), so this runs on any machine.

Usage:
    conda activate mono-cbp
    python experiments/01_download_lightcurves.py --limit 1     # smoke test (one system)
    python experiments/01_download_lightcurves.py               # full run (background/overnight)
Env overrides:
    MONO_CBP_DIR  (default: ~/mono-cbp)
    CBP_DATA_DIR  (default: <repo>/data)
"""
import argparse
import os
from pathlib import Path

from mono_cbp.utils.data import load_catalogue, catalogue_to_lc_files

MONO_CBP = Path(os.environ.get("MONO_CBP_DIR", Path.home() / "mono-cbp"))
CATALOGUE = MONO_CBP / "catalogues" / "TEBC_morph_05_P_7.csv"
DATA_ROOT = Path(os.environ.get("CBP_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CACHE = DATA_ROOT / "lc_cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N systems (smoke test); default = all")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    cat = load_catalogue(str(CATALOGUE), TEBC=True)
    if args.limit:
        cat = cat.iloc[: args.limit]

    print(f"Catalogue: {len(cat)} systems -> cache {CACHE}")
    print("Resumable: files already in the cache are skipped; only missing ones download.")
    catalogue_to_lc_files(cat, output_path=str(CACHE))
    n = len(list(CACHE.glob("*.txt")))
    print(f"Done. Cache now holds {n} light-curve files.")


if __name__ == "__main__":
    main()
