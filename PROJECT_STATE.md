# PROJECT_STATE — cbp-vet ledger

The running log of work on this project. Every working session appends a dated
block: what was done, key numbers, what's blocked/open, and the next action.
Every session starts by reading this file plus the relevant week brief.

Ledger block template:

```
## YYYY-MM-DD (Wk N, session M)
Done: ...
Numbers: ...
Blocked/Open: ...
Next: ...
```

**Project docs.** The detailed planning docs (execution plan, novelty audit,
week worksheets, proposal, abstract variants) are kept in a private working
folder and are intentionally *not* committed to this public repo before T0 —
see `docs/internal/README.md`. The public surface stays modest until the T0
report ships (build-Week 3).

**Two standing rules.** (1) Keep this ledger current. (2) Never unfreeze the
benchmark after it is frozen (build-Week 2) — any change requires a versioned
`bench-vN` with written justification.

---

## 2026-07-04 (Wk 0)
Done: Novelty gate closed at high confidence (adversarial sweep + NASA ADS
  scan + Semantic Scholar — mono-cbp citations = 0). Abstract drafted in three
  lengths; internal proposal v2 + one-page mentor summary produced.
Numbers: mono-cbp funnel confirmed 7,176 TCEs → 2,387 → 1,647 → 1 candidate
  (TIC 319011894, S7); 512 EBs / 3,808 light curves; 14 known transiting CBPs.
Blocked/Open: mentor email (pass 1) not yet sent; abstract deadline Fri Jul 10.
Next: send mentor email; stand up repo + environment; watch run 1.

## 2026-07-06 (Wk 0, session — repo + ingestion head-start)
Done: Public repo skeleton created (this commit). Confirmed the mono-cbp build
  environment on the work machine: conda env `mono-cbp`, mono_cbp 0.1.9,
  lightkurve 2.6.0, numpy 1.26.4, wotan, batman 2.5.1, lmfit 1.3.4, astropy 6.0.1
  (no `exoplanet` — only needed later for injection profiles). All 6 example
  notebooks run; bundled example data = TIC 260128333 (TOI-1338), sectors 6–7.
  Reference S6 event verified in examples/results/detected_events.txt:
  t = 1483.907 BTJD, depth 0.31%, duration 0.29 d, SNR 14.62.
  Wrote + ran experiments/00_ingest_toi1338.py (rebuild the S6 light curve from
  raw MAST → mono-cbp search → recover the reference event). ACCEPTANCE: PASS.
Numbers: TEBC parent catalogue TEBC_morph_05_P_7.csv = 592 systems; TOI-1338
  ephemeris P = 14.60858 d, bjd0 = 1585.1667 (BTJD). Rebuilt-from-MAST light
  curve matches the shipped example NPZ to median |Δflux| = 2.3e-8; both recover
  the reference event t=1483.9075, depth 0.311%, dur 0.292 d, SNR 14.62 (Δ=0.00).
  Flux recipe = CROWDSAP/FLFRCSAP-corrected SAP (bitmask 'hard'), not plain SAP.
Repo LIVE (public): https://github.com/wilsonwzp17/cbp-vet (16 files; internal
  strategy docs + data/npz excluded, verified on GitHub). Git identity set to
  Wilson Wu + private no-reply email; both initial commits re-stamped to it.
Blocked/Open: mentor email still to send (critical path to Jul 10).
Next: send mentor email; on Jul 10 close Week 0 and generate the Week-1 worksheet
  (scale ingestion to TOI-1338's other sectors + 2 Kepler CBP hosts, time per LC).
