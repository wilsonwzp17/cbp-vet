# Ingestion head-start — TOI-1338 (TIC 260128333) S6

**Date:** 2026-07-06 · **Script:** `experiments/00_ingest_toi1338.py` · **Env:** conda `mono-cbp`
(mono_cbp 0.1.9, lightkurve 2.6.0, numpy 1.26.4) · **Result: PASS**

## What was tested
Can we rebuild a mono-cbp input light curve *from raw MAST* and recover the same
monotransit the bundled example data yields? This is the load-bearing assumption
for the Week-1/2 dataset factory.

## Recipe (mirrors `mono_cbp.utils.data.lc_to_txt`, reused not re-implemented)
1. Ephemeris from `catalogues/TEBC_morph_05_P_7.csv`: P = 14.608583 d,
   bjd0 = 1585.1667 BTJD; prim (pos 1.000, width 0.024), sec (pos 0.454, width 0.022).
2. `lightkurve.search_lightcurve("TIC 260128333", mission="TESS", sector=6,
   author="TESS-SPOC").download(flux_column="sap_flux", quality_bitmask="hard")`.
3. Crowding correction with CROWDSAP = 0.98405, FLFRCSAP = 0.84036:
   `flux = (sap - (1-CROWDSAP)*median)/FLFRCSAP`, then normalise to median 1.
4. Phase via `time_to_phase`; drop NaNs → **963 cadences**, 30-min, span 1468.30–1490.03 BTJD.
5. Save NPZ (time/flux/flux_err/phase/eclipse_mask) → `EclipseMasker(force=True)` → `TransitFinder`.

## Numbers
| Check | Result |
|---|---|
| Reconstruction vs shipped NPZ | 963/966 timestamps matched; **median \|Δflux\| = 2.3e-8**, 95th pct 6.2e-8 |
| Reference event (shipped control) | t=1483.9075, depth 0.311%, dur 0.292 d, **SNR 14.62** |
| Reference event (rebuilt from MAST) | t=1483.9075, depth 0.311%, dur 0.292 d, **SNR 14.62** (Δ=0.00 on all) |

Both light curves also flag the weak t=1471.574 TCE (SNR 3.67) — below the min_snr=5
cut, as expected; it drops out at the vetting stage.

## Takeaways for Week 1
- Raw-MAST → mono-cbp search works on the build machine; the exact flux recipe is the
  CROWDSAP/FLFRCSAP-corrected SAP flux with `quality_bitmask='hard'` (not plain SAP).
- The shipped example NPZs were produced by this same recipe (Δflux ~ 1e-8 confirms it).
- Events are read cleanly from `pipeline.transit_finder.results` (tics/sectors/event_times/
  event_depths/event_durations/event_snrs) — the exporter's raw feed for the benchmark.
- Next scale step (Week 1): repeat across TOI-1338's other sectors + the 2 Kepler CBP hosts,
  and measure runtime per light curve.
