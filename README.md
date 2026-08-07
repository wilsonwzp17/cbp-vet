# cbp-vet

Machine-learning vetting for circumbinary planet transit candidates in TESS
eclipsing-binary light curves. Built on the open-source
[mono-cbp](https://github.com/bdrdavies/mono-cbp) search pipeline
(Davies et al. 2026, arXiv:2604.09435).

## The problem

A circumbinary planet orbits two stars, so its transits do not repeat on a
fixed period. Folding-based search tools cannot see them. mono-cbp handles the
search by flagging convincing dips one at a time, which moves the cost
downstream. Its published run flagged 7,176 candidate events, and after
automated filtering a human still had to look at 1,647 of them by eye to find
one real candidate.

This project treats vetting as a ranking problem. Train a model to do the
triage, so a human looks at a hundred plots instead of 1,647 and still finds
the planet.

## Why the dataset had to be built

There is no labelled training data for this. About fourteen transiting
circumbinary planets are known. That is a test set, not a training set, and it
is used as one. They stay sealed until a single scripted evaluation.

Negatives come from a frozen replication of the mono-cbp search over 1,946
eclipsing-binary light curves, with the event rate reproduced to 0.2 percent.
Positives are injected, from a 16,384-profile Sobol shape bank whose pair
physics is calibrated against a dynamical integrator and checked against
published spacings.

A dataset built from injections fails when the building leaves a mark a model
can read instead of the physics. The marks we knew about are designed out: 50/50
dual-inversion injection, importance-sampled epoch placement, and one code path
for both classes. For the ones we did not know about, seven probes run before
the dataset is allowed to freeze, and the pass condition is inverted. A probe
has to fail to predict the label for the dataset to pass. Three of the seven
failed, and they stay in the table with the response each one was assigned
beforehand.

The result froze as bench-v1: 188,732 events over 584 systems, every artifact
SHA-256 hashed. Corrections found after the freeze go in an errata file beside
it, so the hashes stay true.

## The result

At the incumbent pipeline's own recall on the 0.15 to 0.30 percent depth
stratum, where circumbinary transits actually live, the ranker cuts
manual-inspection false positives from 65 to 8. Read the other way, at the
incumbent's own false-positive budget it recovers 78 percent of true shallow
transits against the incumbent's 35.

The control is what makes that mean anything. A second model given only the
incumbent's own 17 features lands at 63, so it matches the incumbent instead of
beating it. That is what a fair harness should produce.

Eight is a small count and the interval around it is wide, [4.0, 71.8] per
1,000 real-negative files. That interval is quoted every time the headline is.
Both selected models are checkpointed with SHA-256 hashes, and every later use
asserts the hash before scoring.

## Layout

- `cbpvet/` the package: search extension, dual-injection engine, calibrated
  pair models, dataset export and schema, leak probes, frozen split, model arms
  with a single feature gate for training and scoring, figure code
- `experiments/` numbered drivers in dependency order. Each docstring says what
  it produces, why it exists, and the traps it avoids
- `notebooks/` executed walkthroughs. Start with
  `CBP-Vet_Data-Engine_Walkthrough.ipynb`. `CBP-Vet_Input-Anatomy.ipynb` opens
  up what the model actually reads, and `CBP-Vet_Showcase.ipynb` is the short
  version
- `tests/` invariant guards: verbatim-copy parity, arm feature contract,
  sign-invariance, split integrity
- `ARCHITECTURE.md` the module map, the data flow, and the invariants the
  benchmark depends on
- `data/` light-curve cache and outputs, not tracked

## Running it

Open `notebooks/CBP-Vet_Data-Engine_Walkthrough.ipynb` and run it top to
bottom. Every cell loads a real frozen artifact and nothing is mocked. The
environment is pinned in `configs/`.

GPL-3.0, matching mono-cbp.
