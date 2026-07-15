# Abstract (submitted 2026-07-10; final wording resubmitted 2026-07-14)

Submitted through the registration form on Fri Jul 10, 2026, and resubmitted with
two small line-edits at the Jul 14 final deadline (237 words, under the 250-word cap).
The text below is the final wording. The submission receipt is kept in the private
project records, not in this public repo.

---

Circumbinary planets, planets that orbit both stars of a binary system, are among the hardest planets to detect. Most known planets were found by the transit method: when a planet's orbit is edge-on, it passes in front of a star and blocks a little light. Around a single star these transits repeat on a fixed period, so automated searches can average the data to boost the signal. Around a binary they do not: the transits are aperiodic and vary in depth and duration, so standard periodic search tools fail. Consequently, all 14 currently known transiting circumbinary planets were discovered by visual inspection, and none since 2021. The open-source mono-cbp pipeline automates the transit search in TESS eclipsing-binary light curves, but its candidate vetting is still heuristic and ends in manual inspection, which does not scale to the thousands of eclipsing binaries now catalogued. We are developing a machine-learning vetting layer, an approach mature for single-star surveys but not previously applied to the circumbinary case, that separates real transits from false positives and ranks the remaining candidates to cut the manual workload. The training data come from mono-cbp's injection-recovery framework, augmented with a physics-based model of circumbinary transits, and we benchmark it against the current heuristic vetting on injected transits and the known circumbinary planets. In this talk we describe how the vetting layer is trained and show preliminary results. All code and data will be released openly.
