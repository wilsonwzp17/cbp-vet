"""B11: stage the deployment run's inputs so the Aug-9 fork cannot crash or corrupt.

The two verified failure modes this script exists to prevent
------------------------------------------------------------
1. IN-PLACE CORRUPTION. mono-cbp's masker rewrites light-curve files where they
   stand. The 63 deployment TICs' files live in the frozen cache, which is
   never written (invariant 2). So everything is staged as a COPY (cp
   semantics, never a symlink - a symlink would route the masker's write
   straight back into the cache).
2. THE SKYE IndexError. The Skye metric indexes
   ``sector_times['start_time'].values[sector - 1]`` (finder.py ~705). The
   shipped sector_times.csv ends at sector 98; the staged QLP files reach
   sector 103. Any event in sectors 99-103 kills the run. The fix pinned in
   Execution-Readiness B11: an EXTENDED CSV in cbp-vet (the mono-cbp original
   is never edited) with contiguous rows 1..max_staged - shipped rows kept
   verbatim, missing rows computed as min/max BTJD over that sector's staged
   files - passed to the pipeline as ``sector_times_path``.

Also enforced here, per the same recipe:
- mis-named cache files are EXCLUDED from staging BY RULE: a filename whose
  sector token is not a plausible TESS sector number (1..150) cannot be
  searched (the Skye metric indexes by sector). Exactly one such file exists
  in the 63-TIC set (its sector token is a 4-digit non-sector; investigate
  separately, never search it) and the count is asserted;
- the staged inventory is manifested with SHA-256 hashes so the Aug-9 driver
  can assert it is running on exactly this staging.

Expected numbers (verified against disk 2026-08-06): 63 TICs, 317 matching
cache files, 316 staged after the exclusion, max real sector 103. The
extended CSV keeps shipped rows value-identical (pandas re-serializes a few
rows' float formatting; byte-diff checked, value-equal).
"""

import hashlib
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIC_LIST = os.path.join(REPO, "data", "goodsn_in_tebc.txt")
CACHES = [os.path.join(REPO, "data", "lc_cache"),
          os.path.join(REPO, "data", "lc_cache_qlp")]
MONO_SECTOR_TIMES = os.path.expanduser("~/mono-cbp/catalogues/sector_times.csv")
STAGE = os.path.join(REPO, "data", "deploy_staged")
STAGE_LC = os.path.join(STAGE, "lc")
MAX_PLAUSIBLE_SECTOR = 150          # mis-named files excluded by rule, not by name


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tics = {ln.strip() for ln in open(TIC_LIST) if ln.strip()}
    assert len(tics) == 63, f"expected 63 TICs, got {len(tics)}"

    matched, excluded = [], []
    for cache in CACHES:
        for f in sorted(os.listdir(cache)):
            m = re.match(r"TIC_(\d+)_(\d+)\.txt$", f)
            if not (m and m.group(1) in tics):
                continue
            if int(m.group(2)) > MAX_PLAUSIBLE_SECTOR:
                excluded.append(os.path.join(cache, f))
                continue
            matched.append((cache, f, int(m.group(2))))
    assert len(matched) + len(excluded) == 317, (
        f"expected 317 cache files for the 63 TICs, got "
        f"{len(matched) + len(excluded)}")
    assert len(excluded) == 1, f"expected exactly 1 excluded file: {excluded}"

    os.makedirs(STAGE_LC, exist_ok=True)
    manifest = {}
    for cache, f, sector in matched:
        src = os.path.join(cache, f)
        dst = os.path.join(STAGE_LC, f)
        shutil.copy2(src, dst)          # a real copy; the masker may write it
        assert not os.path.islink(dst)
        manifest[f] = {"source": os.path.relpath(src, REPO),
                       "sector": sector, "sha256_staged": sha256(dst)}
    max_sector = max(s for _, _, s in matched)

    # ---- extended sector times ------------------------------------------
    st = pd.read_csv(MONO_SECTOR_TIMES)
    shipped_max = int(st.Sector.max())
    assert list(st.Sector) == list(range(1, shipped_max + 1)), \
        "shipped sector_times is not contiguous from 1; indexing assumption dead"

    per_sector = {}
    for _, f, sector in matched:
        if sector <= shipped_max:
            continue
        t = np.loadtxt(os.path.join(STAGE_LC, f), skiprows=1, usecols=0)
        lo, hi = float(np.min(t)), float(np.max(t))
        if sector in per_sector:
            per_sector[sector] = (min(per_sector[sector][0], lo),
                                  max(per_sector[sector][1], hi))
        else:
            per_sector[sector] = (lo, hi)
    missing = [s for s in range(shipped_max + 1, max_sector + 1)
               if s not in per_sector]
    assert not missing, (
        f"sectors {missing} have no staged files to bound; values[sector-1] "
        f"indexing needs contiguous rows - stop and resolve, do not guess")

    rows = [{"Sector": s, "start_time": lo, "end_time": hi}
            for s, (lo, hi) in sorted(per_sector.items())]
    ext = pd.concat([st, pd.DataFrame(rows)], ignore_index=True)
    assert list(ext.Sector) == list(range(1, max_sector + 1))
    ext_path = os.path.join(STAGE, "sector_times_extended.csv")
    ext.to_csv(ext_path, index=False)

    summary = {
        "built": "2026-08-06",
        "recipe": "Execution-Readiness_2026-07-30.md B11",
        "n_tics": len(tics),
        "n_cache_matches": len(matched) + len(excluded),
        "n_staged": len(matched),
        "excluded": [os.path.basename(e) for e in excluded],
        "max_staged_sector": max_sector,
        "shipped_sector_rows": shipped_max,
        "extended_rows_added": sorted(per_sector),
        "sector_times_extended": os.path.relpath(ext_path, REPO),
        "sector_times_extended_sha256": sha256(ext_path),
        "staged_files": manifest,
    }
    with open(os.path.join(STAGE, "staging_manifest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"staged {len(matched)} files ({len(tics)} TICs), excluded "
          f"{[os.path.basename(e) for e in excluded]}")
    print(f"sector_times extended {shipped_max} -> {max_sector} rows "
          f"(added {sorted(per_sector)}) at {ext_path}")


if __name__ == "__main__":
    main()
