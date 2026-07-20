# cbp-vet

Machine-learning vetting for circumbinary planet transit candidates in TESS
eclipsing-binary light curves. Builds on the open-source
[mono-cbp](https://github.com/bdrdavies/mono-cbp) search pipeline
(Davies et al. 2026, arXiv:2604.09435).

Circumbinary planet transits are aperiodic, so standard periodic search tools
fail and candidate vetting has relied on manual inspection. This project trains
classifiers to separate real transits from false positives, benchmarked against
mono-cbp's heuristic vetting.

Work in progress (summer 2026). First results expected August 2026; the
conference abstract is in [docs/abstract.md](docs/abstract.md).

## Layout

- `experiments/` — numbered scripts, one per step (data validation, downloads,
  injection profiles)
- `cbpvet/` — the package (dataset export, benchmark, models); being built
- `configs/` — pinned package versions
- `data/` — light-curve cache and outputs (not tracked)

GPL-3.0, matching mono-cbp.
