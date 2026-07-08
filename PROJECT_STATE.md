# PROJECT_STATE — cbp-vet ledger

The running log of work on this project. Every working session appends a dated
block: what was done, key numbers, and the next action. Every session starts by
reading this file plus the relevant week brief.

Ledger block template:

```
## YYYY-MM-DD (Wk N)
Done: ...
Numbers: ...
Next: ...
```

**Project docs.** The detailed planning docs (execution plan, novelty audit,
week plans, proposal, abstract drafts) are kept in a private working folder and
are intentionally *not* committed to this public repo before T0. The public
surface stays modest until the T0 report ships (build-Week 3).

**Two standing rules.** (1) Keep this ledger current. (2) Never unfreeze the
benchmark after it is frozen (build-Week 2); any change requires a versioned
`bench-vN` with written justification.

---

## 2026-07-04 (Wk 0)
Done: Novelty gate closed at high confidence (adversarial literature sweep +
  citation checks; the mono-cbp paper has 0 citations). Abstract drafted in
  several lengths; internal proposal + one-page summary produced.
Numbers: mono-cbp funnel confirmed 7,176 TCEs -> 2,387 -> 1,647 -> 1 candidate
  (TIC 319011894, S7); 512 EBs / 3,808 light curves; 14 known transiting CBPs.
Next: stand up repo + environment; get input on the abstract.

## 2026-07-06 (Wk 0)
Done: Public repo skeleton created and committed (package layout, GPL-3.0,
  modest README, ledger, env freeze). Confirmed the mono-cbp build environment:
  conda env `mono-cbp`, mono_cbp 0.1.9, lightkurve 2.6.0, numpy 1.26.4, wotan,
  batman 2.5.1, lmfit 1.3.4, astropy 6.0.1. All 6 example notebooks run; bundled
  example data = TIC 260128333 (TOI-1338), sectors 6-7. Wrote + ran
  experiments/00_ingest_toi1338.py (rebuild the S6 light curve from raw MAST ->
  mono-cbp search -> recover the reference event). ACCEPTANCE: PASS.
Numbers: TEBC parent catalogue = 592 systems; TOI-1338 P = 14.60858 d,
  bjd0 = 1585.1667 BTJD. Rebuilt-from-MAST light curve matches the shipped
  example NPZ to median |dflux| = 2.3e-8; both recover the reference event
  t=1483.9075, depth 0.311%, dur 0.292 d, SNR 14.62. Flux recipe =
  CROWDSAP/FLFRCSAP-corrected SAP (bitmask 'hard'), not plain SAP.
Next: push the repo live; get input on the abstract; scale ingestion at Week 1.

## 2026-07-07 (Wk 0)
Done: Repo live and public. Abstract reviewed and the ML-vetting focus endorsed;
  oral-talk format and a 250-word cap confirmed; first submission Fri Jul 10
  (final Tue Jul 14). Abstract revised to a ~230-word general-audience draft
  (keeps the "vetting" scope, no over-claims). Reviewed an inbound data-handling
  notebook set; queued a Week-1 pipeline cross-check.
Next: finish and submit the abstract (Fri Jul 10); then generate the Week-1 agenda.
