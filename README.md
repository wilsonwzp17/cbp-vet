# cbp-vet

Machine-learning vetting for circumbinary planet transit candidates in TESS
eclipsing-binary light curves. Builds on the open-source
[mono-cbp](https://github.com/bdrdavies/mono-cbp) search pipeline
(Davies et al. 2026, arXiv:2604.09435).

Circumbinary planet transits are aperiodic, so standard periodic search tools
fail and candidate vetting has relied on manual inspection. This project trains
classifiers to separate real transits from false positives, benchmarked against
mono-cbp's heuristic vetting.

**Status (August 2026):** the benchmark is frozen (`bench-v1`: 188,732 labelled
events behind seven adversarial leak probes and a frozen by-system split), and
the first head-to-head is banked: at the incumbent heuristics' own recall on
the headline depth stratum, the trained model cuts manual-inspection false
positives per 1,000 light curves by roughly 8x, with the dual reading (recall
at the incumbent's false-positive budget) confirming the same margin. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the module map, the data flow, and the
invariants; `data/bench/full/` artifacts (manifest, datasheet, errata, T0
tables) document every rule and every known limitation.

## Layout

- `cbpvet/` — the package: search extension, dual-injection engine, calibrated
  pair models, dataset export + schema, leak probes, frozen split, model arms
- `experiments/` — numbered drivers in dependency order; each docstring states
  what it produces, why, and the traps it avoids
- `notebooks/` — the executed end-to-end walkthrough of the data engine
- `tests/` — invariant guards (verbatim-copy parity, arm feature contract,
  sign-invariance, split integrity)
- `configs/` — pinned package versions
- `data/` — light-curve cache and outputs (not tracked)

GPL-3.0, matching mono-cbp.
