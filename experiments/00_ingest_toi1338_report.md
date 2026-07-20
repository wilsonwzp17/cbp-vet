# Pipeline validation: TOI-1338 (TIC 260128333), sector 6

Script: `00_ingest_toi1338.py`. Run 2026-07-06. Result: pass.

The test: rebuild a light curve from raw MAST data and check that the mono-cbp
search recovers the same transit that the pipeline's bundled example data
yields. This validates the whole data path before building anything on it.

Recipe: TESS-SPOC sector 6, SAP flux with the CROWDSAP/FLFRCSAP crowding
correction and quality_bitmask='hard', normalized to median 1, phased on the
catalogue ephemeris (P = 14.608583 d).

Results:

| Check | Outcome |
|---|---|
| Rebuilt vs bundled light curve | 963/966 timestamps matched, median flux difference 2.3e-8 |
| Reference transit (bundled) | t = 1483.9075, depth 0.311%, duration 0.292 d, SNR 14.62 |
| Reference transit (rebuilt) | identical to the digit |

The near-zero flux difference confirms the bundled example data was produced by
this same recipe. Note the correct flux is the crowding-corrected SAP, not
plain SAP.
