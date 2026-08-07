# cbp-vet

Machine-learning vetting for circumbinary planet transit candidates in TESS
eclipsing-binary light curves. Builds on the open-source
[mono-cbp](https://github.com/bdrdavies/mono-cbp) search pipeline
(Davies et al. 2026, arXiv:2604.09435).

## The problem, and the idea

A circumbinary planet orbits two stars at once. Its transits do not repeat
on a clock, so every folding-based search tool is blind to them. mono-cbp
solved the search half: it flags convincing dips one at a time. The cost
moved downstream: its published run flagged 7,176 candidate events, and
after automated filtering a human still had to inspect 1,647 of them by eye
to find one real candidate. The human queue is the instrument's limiting
element.

This project turns vetting into a ranking problem: train a model to do the
triage, so a human looks at a hundred plots instead of 1,647 without losing
the planet. The obstacle is that nobody has labelled training data — there
are only about fourteen known transiting circumbinary planets, which is not
a training set but a test set, and that is exactly how they are treated
here: sealed until a single scripted evaluation.

## How the dataset was manufactured honestly

Real negatives come from a faithful, frozen replication of the incumbent
search over 1,946 eclipsing-binary light curves (event rate reproduced to
0.2 percent). Synthetic positives come from a 16,384-profile Sobol shape
bank whose pair physics — the "1-2 punch" of one conjunction crossing both
stars — is calibrated against a dynamical integrator's ground truth and
cross-checked against published statistics.

Injection-based datasets fail by leaving procedural fingerprints a model
can read instead of physics. The known fingerprints are designed out
(50/50 dual-inversion injection, importance-sampled epoch placement, one
code path for both classes), and the unknown ones are hunted by **seven
adversarial probes whose pass condition is inverted**: a probe must FAIL
to predict the label for the dataset to pass. Failures are disclosed and
dispositioned, never hidden. The result froze as **bench-v1**: 188,732
events, 584 systems, every artifact SHA-256 hashed; later corrections live
in an errata file beside the freeze so the hashes stay true forever.

## The banked result, and why to believe it

At the incumbent pipeline's own recall on the 0.15–0.30 percent depth
stratum (where circumbinary transits actually live), the trained ranker
cuts manual-inspection false positives roughly **eightfold** — and the
design carries its own control: a second model given *only* the
incumbent's own 17 features reproduces the incumbent instead of beating
it, so the harness is provably fair and the gain comes from the added
physics context. The dual reading holds too: at the incumbent's own
false-positive budget, the model recovers 78 percent of true shallow
transits against the incumbent's 35. The uncertainty is honest and wide
(eight false positives is a small count) and is always quoted. Both
selected models are checkpointed with SHA-256 hashes, and every downstream
use asserts the hash before scoring.

## What the repo contains

- `cbpvet/` — the package: search extension, dual-injection engine,
  calibrated pair models, dataset export + schema (with the audit-column
  doctrine), leak probes, frozen split, model arms with a single feature
  gate for training and scoring, figure code
- `experiments/` — numbered drivers in dependency order (07–28); each
  docstring states what it produces, why it exists, and the traps it
  avoids — they are the primary in-repo documentation
- `notebooks/` — executed walkthroughs: the data engine end to end
  (`CBP-Vet_Data-Engine_Walkthrough.ipynb`, the place to start), an input
  anatomy teaching notebook, and a demo variant
- `tests/` — invariant guards (verbatim-copy parity, arm feature
  contract, sign-invariance, split integrity)
- `ARCHITECTURE.md` — the module map, data flow, and the eleven
  invariants that make the benchmark mean something
- `data/` — light-curve cache and outputs (untracked)

## Run the demo

Open `notebooks/CBP-Vet_Data-Engine_Walkthrough.ipynb` and run it top to
bottom: every cell loads real frozen artifacts, nothing is mocked, and the
story reads in the same order as this README. The environment is pinned in
`configs/`.

GPL-3.0, matching mono-cbp.
