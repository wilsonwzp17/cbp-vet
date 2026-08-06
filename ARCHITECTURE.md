# CBP-Vet architecture

**What this is.** The machine-learning vetter for circumbinary-planet transit candidates, built on the open-source mono-cbp search pipeline. This document is the build-on-top map: every module's job, the data flow, the invariants you must not break, and where the frozen benchmark's rules live.

## The one-paragraph idea

mono-cbp's search flags candidate events; its published run needed a human to eyeball 1,647 of them to find one planet. We manufacture the labelled dataset nobody has (real search negatives + injected positives from two calibrated sources), freeze it behind adversarial leak probes, and train models that rank the queue. The banked head-to-head: at the incumbent's own recall on the headline depth stratum, the full model cuts manual-inspection false positives per 1,000 light curves by ~8× (see `data/bench/full/t0_core.json`).

## Data flow

```
frozen 1,945 light curves (data/search_frozen/staged, masked COPIES - never the cache)
   │  experiments/07_search_frozen.py  → real negatives + (phase,gap) histogram
   ▼
bank v2 (08_bank_v2.py: 16,384 Sobol profiles)     ELC models (09_elc_driver.py)
   │            pair model: cbpvet/injection/pair_model.py (ELC-calibrated v3)
   ▼                                                   ▼
60k dual campaign + pair re-injection (10_campaign.py)  ELC batch (16_elc_batch.py)
   │  cbpvet/injection/dual_injector.py: 50/50 inversion, importance-sampled
   │  epochs (epoch_sampler.py), full harvest, pair support
   ▼
EXPORTER (13_export.py → cbpvet/export/*): one record per event, same code
   │  both classes; local [2,201] + recurrence [2,64] + 39 scalars;
   │  frozen TIC split (bench/split.py); rebalance flag (injection/rebalance.py)
   ▼
bench_full.h5  ──►  probes + diagnostics (14_probes.py → bench/probes.py)
   │                 certified on the frozen split
   ▼
FREEZE (MANIFEST_bench-v1.md + DATASHEET + FREEZE_bench-v1.json hashes)
   ▼
arms + T0 (cbpvet/models/arms.py + 20_m0_m1_t0.py): M0 operating points,
M1 grid (selection on val PR-AUC only), matched-recall head-to-head, banked.
```

## Modules

| Module | Job | The invariant it owns |
|---|---|---|
| `cbpvet/search/finder_ext.py` | mono-cbp's finder + persist `n_detrend_detections` | verbatim copy of `_process_cb_events` + one line; parity asserted by test |
| `cbpvet/injection/dual_injector.py` | per-test inversion, injectable epochs, full harvest, pairs, `inject_fixed_events` for ELC | one event-extraction code path for both classes (the finder's) |
| `cbpvet/injection/pair_model.py` | pair spacing / depth / duration models | all three ELC-calibrated (v3); history of wrong models kept in comments |
| `cbpvet/injection/epoch_sampler.py` | bin-first draws from the real negatives' joint histogram | joint, not marginal; fallbacks counted |
| `cbpvet/injection/rebalance.py` | probe-1 remedy | flag, never delete |
| `cbpvet/export/schema.py` | the record definition | `training_scalars` is THE feature gate; `AUDIT_COLUMNS` forbidden; withheld list; the star-identity doctrine |
| `cbpvet/export/features.py` | views + scalars | recurrence from ORIGINAL flux (sign-invariance tests); gap from the full file; observed-pair quantities only |
| `cbpvet/export/incumbent.py` | the 17-column block | outer join by reproduced comparator key; failures keep rows |
| `cbpvet/export/exporter.py` | record assembly + HDF5 | per-source `pair_model_version` read from each run's own summary |
| `cbpvet/bench/split.py` | frozen split | by TIC, stratified, 10-plus tier 60/15/25, hashed |
| `cbpvet/bench/probes.py` | seven probes + MDE + diagnostics | a probe must FAIL to predict; certified on the frozen split; folded AUC disclosed |
| `cbpvet/models/arms.py` | feature-model arms | `load_matrix` is the single gate; arm contract = fitted features == `training_scalars` |
| `cbpvet/physics.py` | eccentricity from eclipses, Holman–Wiegert, period draws | corrected Winn relation; mu = 0.3 pinned |

Experiments `07`–`20` are the drivers, numbered in dependency order; each docstring states what it produces, why, and the traps it avoids (these docstrings are the primary in-repo documentation and record every measured defect).

## The invariants (break these and the benchmark stops meaning anything)

1. **mono-cbp originals and ELC's distribution are never edited.** ELC runs are fingerprint-guarded (110 files SHA-256'd before/after every run).
2. **The frozen cache is never written.** Everything stages COPIES (the masker rewrites in place).
3. **One code path for both classes** for every feature; anywhere they differ, the difference is learnable.
4. **The recurrence view reads the original flux** — sign-invariant by construction, tested.
5. **Injection truth is never a feature.** `truth_*` and `AUDIT_COLUMNS` are audit only; star identity is unobservable for real events.
6. **The frozen split is the only split**; test is touched once, at T0.
7. **Never re-run the frozen search**; its outputs are the benchmark's denominators.
8. **Pre-named responses, never silent edits.** Gate failures spend their named response; changes are dated annotations.

## The frozen benchmark

`data/bench/full/`: `bench_full.h5` (188,732 events), `MANIFEST_bench-v1.md` (all rules), `DATASHEET_bench-v1.md` (honest limitations), `FREEZE_bench-v1.json` (SHA-256 of every frozen artifact), `probe_results.json`, `t0_core.json` + `t0_addendum_2026-08-06.json` (the dual metric, the pinned-denominator restatement, and the frozen model checkpoints in `models/`, written by `experiments/21_freeze_models.py` only after reproducing every banked number), and `ERRATA_bench-v1.md` — post-freeze corrections live there, never as edits to frozen files, so the hashes stay true. Planning, decisions, and the full defect history live in the project docs folder (see the manifest's pointers), notably the build tutorial's error registers — 26 resolved defects, each with cause, evidence, and fix.

## Building on top

- New features: add to `schema.py`, compute in `features.py` identically for both classes, screen with a probe/diagnostic **before** offering to models.
- New models/arms: load through `arms.load_matrix` only; add the arm to `ARM_FEATURES`; selection on val PR-AUC; test split only at a banked evaluation.
- New injection sources: subclass or extend `DualInjector`; record your own `pair_model_version`-style pins in the run summary; expect the provenance probe to interrogate you.
- Kepler/other missions: the feature functions are mission-agnostic (verified on Kepler-shaped input); use `19_kepler_smoke.py` as the porting template.
