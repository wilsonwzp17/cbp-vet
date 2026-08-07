# CBP-Vet architecture

The module map for the machine-learning vetter built on top of the mono-cbp
search pipeline. What each module does, how the data flows, the invariants that
must not break, and where the frozen benchmark's rules live.

## The idea

mono-cbp's search flags candidate events, and its published run needed a human
to look at 1,647 of them to find one planet. We build the labelled dataset
nobody has, from real search negatives plus injected positives from two
calibrated sources, freeze it behind leak probes, and train models that rank
the queue. At the incumbent's own recall on the headline depth stratum, the
full model cuts manual-inspection false positives per 1,000 light curves by
about 8x. See `data/bench/full/t0_core.json`.

## Data flow

```
frozen 1,945 light curves (data/search_frozen/staged, masked COPIES, never the cache)
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
| `cbpvet/search/finder_ext.py` | mono-cbp's finder + persist `n_detrend_detections` | verbatim copy of `_process_cb_events` plus one line; parity asserted by test |
| `cbpvet/injection/dual_injector.py` | per-test inversion, injectable epochs, full harvest, pairs, `inject_fixed_events` for ELC | one event-extraction code path for both classes, the finder's |
| `cbpvet/injection/pair_model.py` | pair spacing, depth and duration models | all three ELC-calibrated (v3); the wrong earlier models are kept in comments |
| `cbpvet/injection/epoch_sampler.py` | bin-first draws from the real negatives' joint histogram | joint, not marginal; fallbacks counted |
| `cbpvet/injection/rebalance.py` | probe-1 remedy | flag, do not delete |
| `cbpvet/export/schema.py` | the record definition | `training_scalars` is the feature gate; `AUDIT_COLUMNS` forbidden; withheld list; the star-identity rule |
| `cbpvet/export/features.py` | views and scalars | recurrence from ORIGINAL flux (sign-invariance tests); gap from the full file; observed-pair quantities only |
| `cbpvet/export/incumbent.py` | the 17-column block | outer join by reproduced comparator key; failures keep rows |
| `cbpvet/export/exporter.py` | record assembly and HDF5 | per-source `pair_model_version` read from each run's own summary |
| `cbpvet/bench/split.py` | frozen split | by TIC, stratified, 10-plus tier 60/15/25, hashed |
| `cbpvet/bench/probes.py` | seven probes, MDE, diagnostics | a probe must FAIL to predict; certified on the frozen split; folded AUC disclosed |
| `cbpvet/models/arms.py` | feature-model arms | `load_matrix` is the single gate; arm contract is fitted features == `training_scalars` |
| `cbpvet/physics.py` | eccentricity from eclipses, Holman-Wiegert, period draws | corrected Winn relation; mu = 0.3 pinned |

Experiments `07` to `24` are the drivers, numbered in dependency order. Each
docstring says what it produces, why, and the traps it avoids, and those
docstrings are the primary in-repo documentation. `21` freezes the T0 models
and completes the pinned harness. `22` to `24` are the deployment runway:
staging plus the extended sector-times fix (proven in both directions), the
parameterized mask-then-find driver, and the sealed completion for the third
real-planet host.

## Invariants

If any of these break, the benchmark stops meaning anything.

1. mono-cbp originals and ELC's distribution are never edited. ELC runs are
   fingerprint-guarded, with 110 files SHA-256'd before and after every run.
2. The frozen cache is never written. Everything stages copies, because the
   masker rewrites in place.
3. One code path for both classes, for every feature. Anywhere they differ, a
   model can learn the difference.
4. The recurrence view reads the original flux, so it is sign-invariant. There
   is a test.
5. Injection truth is never a feature. `truth_*` and `AUDIT_COLUMNS` are audit
   only, and star identity is unobservable for real events.
6. The frozen split is the only split. Test is touched once, at T0.
7. The frozen search is never re-run. Its outputs are the benchmark's
   denominators.
8. Responses to gate failures are named in advance. Changes are dated
   annotations, not silent edits.

## The frozen benchmark

`data/bench/full/` holds `bench_full.h5` (188,732 events), `MANIFEST_bench-v1.md`
(all rules), `DATASHEET_bench-v1.md` (the limitations), `FREEZE_bench-v1.json`
(SHA-256 of every frozen artifact), `probe_results.json`, and `t0_core.json`
plus `t0_addendum_2026-08-06.json`, which carry the dual metric, the
pinned-denominator restatement, and the frozen model checkpoints in `models/`.
The addendum is written by `experiments/21_freeze_models.py` only after it
reproduces every banked number.

`ERRATA_bench-v1.md` holds post-freeze corrections. They live there instead of
as edits to the frozen files, so the hashes stay true. Planning, decisions and
the defect history live in the project docs folder, which is not tracked here.

## The deployment and one-shot layer (experiments 22 to 28)

Everything below points the frozen model at data it has never seen. This layer
adds no new learning and never edits the frozen chain. It inherits the
invariants above and adds three of its own.

```
data/deploy_staged (COPIES, extended sector-times 1..103; 22_stage_deployment.py)
   │  23_deploy_run.py: same mask→find chain as the frozen search (invariant 3)
   ▼
deployment TCEs ──► 25_deploy_score.py: FROZEN exporter (unmodified) + comparator
   │                → scoring shard → hash-asserted frozen checkpoint →
   ▼                ranked_all + shortlist v1 (LOCAL ONLY, 15 pinned columns)
detectability (28_detectability.py): dual 50/50 injections into the same masked
   staged copies (a fourth provenance, injector-rows-only label), per-system
   golden + aggregate, disjoint-from-frozen headline split
one-shot (26_one_shot.py, RUNS ONCE): opening hash asserts → pools (frozen-shard
   rows + fresh frozen-exporter passes) → score once at the frozen tau →
   D1/D2 matching vs transcribed PUBLIC times tables → Wilson intervals,
   per-mission rollup → hashed outputs. Reviewed against synthetic fixtures
   (26_fixtures.py) only, until the day it runs.
```

9. The scoring model is hash-asserted before any work. Every ranking or
   evaluation refuses to start unless the checkpoint hashes to the pre-shot
   log.
10. Scoring uses the same single feature gate as training. `arms.score_matrix`
    carries the identical forbidden-column contract, with no ad-hoc feature
    lists.
11. The sealed evaluation is fixture-reviewed and never dry-run. The one-shot
    script is exercised only on synthetic fixtures with hand-computed expected
    outputs until its single real execution.

## Building on top

- New features: add to `schema.py`, compute in `features.py` identically for
  both classes, and screen with a probe or diagnostic before offering them to
  models.
- New models or arms: load through `arms.load_matrix` only, add the arm to
  `ARM_FEATURES`, select on validation PR-AUC, and touch the test split only at
  a banked evaluation.
- New injection sources: subclass or extend `DualInjector`, record your own
  `pair_model_version`-style pins in the run summary, and expect the provenance
  probe to interrogate you.
- Kepler or other missions: the feature functions are mission-agnostic, and
  verified on Kepler-shaped input. Use `19_kepler_smoke.py` as the porting
  template.
