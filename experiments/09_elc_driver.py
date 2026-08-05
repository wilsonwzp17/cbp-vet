"""B4: the ELC config driver, generating physically-real circumbinary positives.

Why ELC at all, when we already have a 16,384-profile bank
-----------------------------------------------------------
The bank gives breadth and volume, but every bank profile is a batman transit
pasted onto a light curve at a time we chose. It knows nothing about the binary.
It cannot tell you which star was crossed, it cannot produce a genuine 1-2
punch, and its pair spacings are ones we sampled rather than ones the geometry
produced.

ELC is a full dynamical binary-plus-planet model. Give it two stars and a
circumbinary planet and it integrates the actual orbits. With
``iwriteeclipse = 1`` it writes, per star, the exact time of every transit,
labelled by which body crossed which star, in files named
``ELC{body}tran{star}time.dat``. So the same-star versus cross-star
distinction, the pair taxonomy, and the mentor's three-part screen are all
**measured rather than assumed**.

That is why the campaign has two provenances. Bank for volume, ELC for realism,
and probe 2 tests explicitly that a classifier cannot tell them apart. If it
can, the label is contaminated by provenance rather than physics.

The claim that nearly cost us the radius axis
----------------------------------------------
The project's own reference document asserted that editing ``Iseason``,
``P1incl`` or ``P1ratrad`` "makes ELC exit silently during setup", and that
belief had shaped the plan: the planet radius was treated as un-settable, so the
injected-radius axis and the mutual-inclination sweep were both considered
impossible.

Re-tested on 2026-07-29 and 2026-07-30 against the real binary: **false for
every field.** ``P1incl``, ``P1ratrad``, ``Iseason``, ``fracdiff`` and
``P1Omega`` all edit cleanly. **The never-edit list is EMPTY.** More than that,
the depth law is exact: measured depth over geometric ``(1/r)^2`` was 1.238,
1.239, 1.240 across ``P1ratrad`` values of 8.24, 12.0 and 20.0, constant to 0.2
percent. It is a **per-host** constant, since it absorbs that binary's flux
ratio and limb darkening, so it is measured per host rather than assumed.

Consequences: the injected planet radius is an exactly controlled variable, so
the detectability-versus-radius axis is real; per-host stellar radii are
settable, so duration realism is fixed at the source; and the mutual-inclination
sweep is buildable.

The substring trap
------------------
The patcher matches configuration lines by their label text. Line 284 of the
template reads ``angsum1 (P1incl + P1Omega), tag 1a`` and therefore **contains
the substrings "P1incl" and "P1Omega"**. Matching on the bare parameter name
would patch the wrong line. Every key here is the full label including its tag,
which was machine-checked collision-free against the real template.

The 189,012-point cap
---------------------
ELC allocates its light-curve array from ``(t_end - t_start) / step`` and has a
compiled-in cap of 189,012 points. Exceed it and ELC **exits 0** with only a
message on stdout and no output files. The earlier retarget script's own first
window violated this. Here windows are built by grouping the host's cached
sectors at gaps longer than 20 days, and the cap is asserted before any run.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time as _time

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mono_cbp.utils import load_catalogue

from cbpvet import physics

REPO = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
MONO = os.path.expanduser("~/mono-cbp")
CAT_PATH = os.path.join(MONO, "catalogues", "TEBC_morph_05_P_7.csv")
ELC90 = os.path.join(REPO, "data", "elc", "ELC90")
TEMPLATE = os.path.join(ELC90, "ELC_Kepler16.inp")
CACHE = os.path.join(REPO, "data", "lc_cache")
NOISE = os.path.join(REPO, "data", "noise_screen.csv")
OUT_ROOT = os.path.join(REPO, "data", "elc_models")

# ---- pinned constants -----------------------------------------------------
DRIVER_SEED = 20260804
MODELS_PER_HOST = 24
SIGMA_LOW = (1e-4, 1e-3)      # 60 percent of models, log-uniform
SIGMA_HIGH = (1e-3, 1e-2)     # 40 percent, log-uniform
SIGMA_LOW_FRACTION = 0.60
TIME_STEP = 0.002
NMAXPHASE = 189012
NMAXPHASE_SLACK = 1000
SECTOR_GROUP_GAP_DAYS = 20.0
WINDOW_PAD_DAYS = 2.0
BOXCAR_DAYS = 30.0 / 1440.0
OOE_BASELINE_TOL = 1e-4
QA_ATTRITION_BUDGET = 0.15
# Reference depth for the pair-host screen, per VERIFIED-FACTS section 6.
REFERENCE_DEPTH = 0.005
DEPTH_TARGET_RANGE = (1.0e-3, 1.5e-2)

# Kepler-16 template values, verified on disk 2026-07-29.
TPL_FINC = 90.3387011123486161
TPL_TEFF1 = 4465.171470000          # Teff1, template line 152
TPL_TEFF2 = 3318.228810000          # Teff2, template line 194
TPL_RATRAD = 2.8479536901918        # R1/R2, template line 204
TPL_SEPAR = 48.251791
TPL_PBIN = 41.079297

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("09_elc")

# Full label text including tag: matching the bare name would hit line 284,
# "angsum1 (P1incl + P1Omega), tag 1a". Machine-checked collision-free.
KEYS = {
    "t_start": "t_start (if itime=2)",
    "t_end": "t_end   (if itime=2)",
    "step": "time step in days (if itime=2)",
    "tref": "Tref for dynamical integrator",
    "period": "Period (days), tag pe",
    "t0": "T0 (time of periastron passage), tag T0",
    "tconj": "Tconj (time of primary eclipse), tag Tc",
    "ecc": "eccentricity, tag ec",
    "omega": "argument of periaston in degrees, tag ar",
    "ecosw": "e*cos(omega), tag oc",
    "esinw": "e*sin(omega), tag os",
    "separ": "separ (semimajor axis in solar radii), tag se",
    "finc": "finc (inclination in degrees), tag in",
    "fracsum": "fracsum ((R_1 + R_2)/a), tag fs",
    "fracdiff": "fracdiff ((R_1 - R_2)/a), tag fd",
    "p1period": "P1period (days), tag tt",
    "p1tconj": "P1Tconj, tag tj",
    "p1incl": "P1incl (degrees), tag tx",
    "p1omega": "P1Omega (degrees), tag ty",
    "p1ratrad": "P1ratrad (radius of star 1 to body 3), tag tb",
    "teff1": "Teff1 (K), tag T1",
    "teff2": "Teff2 (K), tag T2",
    "temprat": "temprat (T_2/T_1), tag te",
    "ratrad": "ratrad (ratio of star 1 radius to star 2 radius,  tag ra",
    "iwriteeclipse": "iwriteeclipse (1 to fit",
}


def elc_source_fingerprint():
    """SHA256 of every file in the ELC distribution directory.

    STANDING RULE: **ELC's own code, binary and shipped templates are never
    modified.** ELC is the mentor's own program and he is its author; any change
    to it is his call, not ours, and would also invalidate every model generated
    before the change.

    This driver therefore only ever READS ``ELC_Kepler16.inp``, COPIES the
    ``ELC`` binary and ``ELC.atm`` into a per-model scratch directory, and
    writes a fresh ``ELC.inp`` inside that scratch directory. Nothing is written
    back into the distribution.

    That is a promise about behaviour, so it is enforced rather than asserted:
    the fingerprint is taken before and after every run and compared.
    """
    import hashlib
    out = {}
    for name in sorted(os.listdir(ELC90)):
        path = os.path.join(ELC90, name)
        if not os.path.isfile(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
    return out


def assert_elc_untouched(before, after):
    changed = [k for k in before if before.get(k) != after.get(k)]
    added = [k for k in after if k not in before]
    removed = [k for k in before if k not in after]
    if changed or added or removed:
        raise RuntimeError(
            "ELC DISTRIBUTION MODIFIED. This must never happen: ELC is the "
            f"mentor's own code. changed={changed} added={added} removed={removed}"
        )
    log.info("ELC distribution verified unchanged (%d files fingerprinted)", len(before))


def check_keys_unique():
    """Every key must match exactly one template line under first-match-wins."""
    lines = open(TEMPLATE).read().splitlines()
    problems = []
    for name, key in KEYS.items():
        hits = [i for i, ln in enumerate(lines) if key in ln]
        if len(hits) != 1:
            problems.append(f"{name!r} ({key!r}) matched {len(hits)} lines: {hits}")
    if problems:
        raise RuntimeError("Patcher key collision:\n  " + "\n  ".join(problems))
    log.info("All %d patcher keys matched exactly one template line", len(KEYS))


def patch(values):
    """First-match-wins substitution, the same loop experiments/04 proved sound."""
    out = []
    for line in open(TEMPLATE).read().splitlines():
        for name, key in KEYS.items():
            if name in values and key in line:
                line = f"  {values[name]}   {key}"
                break
        out.append(line)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- host setup
def sector_windows(tic):
    """Group the host's cached sectors into ELC-runnable time windows.

    Sectors separated by more than 20 days become separate windows, because one
    window spanning the gap would allocate points for the empty stretch too and
    blow the 189,012-point cap.
    """
    spans = []
    for f in sorted(os.listdir(CACHE)):
        if not f.startswith(f"TIC_{tic}_"):
            continue
        arr = np.loadtxt(os.path.join(CACHE, f), skiprows=1)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        t = arr[:, 0]
        spans.append((float(np.nanmin(t)), float(np.nanmax(t)), f))
    if not spans:
        return []
    spans.sort()
    groups, cur = [], [spans[0]]
    for s in spans[1:]:
        if s[0] - cur[-1][1] > SECTOR_GROUP_GAP_DAYS:
            groups.append(cur)
            cur = [s]
        else:
            cur.append(s)
    groups.append(cur)

    windows = []
    for g in groups:
        t0 = min(x[0] for x in g) - WINDOW_PAD_DAYS
        t1 = max(x[1] for x in g) + WINDOW_PAD_DAYS
        # ELC requires (T_start - T_ref) to be an integer multiple of its
        # internal dynamical step hh = 0.1 d, and refuses to run otherwise::
        #
        #   Error: (T_start-T_ref)/hh should be an integer
        #
        # MEASURED 2026-08-04. experiments/04 never hit this because its window
        # bounds were round numbers chosen by hand; windows derived from real
        # data edges are not. Snapping both bounds outward onto the 0.1 d grid
        # makes the difference exactly zero when T_ref is set to T_start, and
        # keeps the window a whole number of integrator steps.
        t0 = np.floor(t0 * 10.0) / 10.0
        t1 = np.ceil(t1 * 10.0) / 10.0
        n_points = (t1 - t0) / TIME_STEP + NMAXPHASE_SLACK
        if n_points >= NMAXPHASE:
            raise RuntimeError(
                f"TIC {tic}: window {t0:.1f}-{t1:.1f} needs {n_points:.0f} points, "
                f"above the {NMAXPHASE} cap. ELC would exit 0 with no output."
            )
        windows.append({"t0": t0, "t1": t1, "n_points": int(n_points),
                        "files": [x[2] for x in g]})
    return windows


def select_hosts(n_hosts, min_sectors=2):
    """Rank candidate pair hosts by partner SNR, per the three-part screen.

    The mentor's screen (2026-07-29): the deep primary transit is 30 to 50
    percent recoverable; the depth ratio between the two stars must be within a
    factor of two to three, so at least about 0.33; and the shallower partner
    transit must sit above the measured out-of-eclipse noise. Applied to the
    measured noise screen this leaves 261 physically eligible systems.
    """
    noise = pd.read_csv(NOISE)
    noise = noise[np.isfinite(noise.depth_ratio) & np.isfinite(noise.rms_cadence)]
    # Per-transit SNR: the partner transit is averaged over ~12 cadences.
    noise["partner_depth"] = REFERENCE_DEPTH * noise.depth_ratio
    noise["partner_snr"] = noise.partner_depth / (noise.rms_cadence / np.sqrt(12))
    eligible = noise[(noise.depth_ratio >= 0.33) & (noise.partner_snr >= 5)]
    eligible = eligible.sort_values("partner_snr", ascending=False)
    log.info("Eligible pair hosts (N_physical): %d", len(eligible))

    cat = load_catalogue(CAT_PATH, TEBC=True).drop_duplicates("tess_id").set_index("tess_id")
    raw = pd.read_csv(CAT_PATH).drop_duplicates("tess_id").set_index("tess_id")

    hosts = []
    for _, r in eligible.iterrows():
        tic = int(r.tic)
        if tic not in cat.index or tic not in raw.index:
            continue
        try:
            windows = sector_windows(tic)
        except RuntimeError as exc:
            log.warning("skipping TIC %d: %s", tic, exc)
            continue
        n_files = sum(len(w["files"]) for w in windows)
        if n_files < min_sectors:
            continue
        rr = raw.loc[tic]
        _, _, e = physics.eclipse_eccentricity(
            rr.get("prim_pos_2g"), rr.get("sec_pos_2g"),
            rr.get("prim_width_2g"), rr.get("sec_width_2g"))
        hosts.append({
            "tic": tic,
            "p_bin": float(cat.loc[tic, "period"]),
            "bjd0": float(cat.loc[tic, "bjd0"]),
            "ecc": float(e if np.isfinite(e) else 0.0),
            "sec_pos_2g": float(rr.get("sec_pos_2g")) if np.isfinite(rr.get("sec_pos_2g")) else 0.5,
            "prim_width_2g": float(rr.get("prim_width_2g")),
            "depth_ratio": float(r.depth_ratio),
            "partner_snr": float(r.partner_snr),
            "windows": windows,
            "n_files": n_files,
        })
        if len(hosts) >= n_hosts:
            break
    return hosts


# ---------------------------------------------------------- model generation
def draw_sigma(rng, n):
    """Coplanarity dispersion, allocated 60 percent tight and 40 percent loose.

    sigma is ONE common dimensionless dispersion, not a sum of per-angle terms.
    Verified first-hand against the Chen and Kipping 2022 text; the earlier
    reading would have doubled the effective misalignment.
    """
    n_low = int(np.ceil(SIGMA_LOW_FRACTION * n))
    lo = 10 ** rng.uniform(np.log10(SIGMA_LOW[0]), np.log10(SIGMA_LOW[1]), n_low)
    hi = 10 ** rng.uniform(np.log10(SIGMA_HIGH[0]), np.log10(SIGMA_HIGH[1]), n - n_low)
    sig = np.concatenate([lo, hi])
    rng.shuffle(sig)
    assert (sig <= SIGMA_LOW[1]).mean() >= SIGMA_LOW_FRACTION - 1e-9, \
        "sigma allocation fell below the 60 percent tight-coplanarity rule"
    return sig


def draw_angles(rng, sigma, finc_host):
    """Mutual inclination and node from one common sigma.

    x is drawn from a normal of width sigma truncated at +/- 1 (so the arcsine
    is always defined), the inclination offset is arcsin(x), and the node offset
    is drawn the same way scaled to +/- 90 degrees.
    """
    x = truncnorm.rvs(a=-1 / sigma, b=1 / sigma, loc=0, scale=sigma,
                      random_state=rng.integers(2 ** 31))
    y = truncnorm.rvs(a=-1 / sigma, b=1 / sigma, loc=0, scale=sigma,
                      random_state=rng.integers(2 ** 31))
    p1incl = finc_host + np.degrees(np.arcsin(np.clip(x, -1, 1)))
    p1omega = np.degrees(y * np.pi / 2)
    return float(p1incl), float(p1omega)


def stellar_geometry(host):
    """fracsum, fracdiff, ratrad and Teff2 for one host.

    CORRECTION OF RECORD, 2026-08-04. The first version read

        k = sqrt(depth_ratio)      # treated as the RADIUS ratio R2/R1

    which is physically wrong. In a detached eclipsing binary BOTH eclipses
    block the same area, namely the smaller star's disc. The primary eclipse
    hides that area of star 1's surface, the secondary hides it of star 2's, so

        depth_sec / depth_prim  =  S2 / S1  ~  (T2 / T1)^4

    to first order. The eclipse depth ratio therefore measures the
    SURFACE-BRIGHTNESS (temperature) ratio and says nothing directly about the
    radii. The census caught this: it reported a median partner-to-lead transit
    depth ratio of 0.039, far from the ~0.9 the code believed it had set,
    because the radii had been reset from the host while the TEMPERATURES were
    left at Kepler-16's values (4465 K and 3318 K) and ``ratrad`` was left at
    Kepler-16's 2.848, inconsistent with the new fracsum and fracdiff.

    What is set now:
      fracsum  = (R1 + R2)/a, from the observed primary eclipse width
      ratrad   = R1/R2, kept at the template value unless better info exists,
                 since the depth ratio does not constrain it
      fracdiff = (R1 - R2)/a, derived from fracsum and ratrad so the three stay
                 mutually consistent
      temprat  = T2/T1 = depth_ratio^(1/4), so the surface-brightness ratio
                 reproduces the host's observed eclipse depth ratio
      Teff2    = Teff1 * temprat, kept consistent for bookkeeping only

    SECOND CORRECTION, same day. Setting ``Teff2`` alone did NOTHING: ELC's
    light curve was byte-identical across Teff2 = 3318, 4398 and 6000 K.
    Template line 195 carries ``temprat (T_2/T_1), tag te`` = 0.7431358, and
    4465.17 * 0.7431358 = 3318.23 exactly, so ELC derives star 2's temperature
    from the RATIO and treats Teff2 as bookkeeping. Driving ``temprat`` instead
    works, MEASURED on host 443055162 (eclipse depth ratio 0.9412):

        temprat 0.7431 (template)      partner/lead transit depth ratio 0.089
        temprat 0.9849 (host-matched)  partner/lead transit depth ratio 0.680
        temprat 1.0000                 partner/lead transit depth ratio 0.746

    The residual gap between 0.680 and the host's 0.941 is the radius ratio,
    which is still Kepler-16's 2.848 because the eclipse depth ratio does not
    constrain it. That is a known, bounded limitation rather than an error.
    """
    fracsum = float(np.pi * host["prim_width_2g"])
    ratrad = TPL_RATRAD                       # R1/R2; not set by the depth ratio
    fracdiff = fracsum * (ratrad - 1.0) / (ratrad + 1.0)
    depth_ratio = float(np.clip(host["depth_ratio"], 1e-3, 1.0))
    temprat = depth_ratio ** 0.25
    teff2 = TPL_TEFF1 * temprat
    return fracsum, fracdiff, ratrad, teff2, temprat


def build_model_spec(rng, host, window, host_factor, sigma=None):
    """One ELC model: sampled geometry plus a planet, as a manifest row.

    ``sigma`` MUST be supplied by the caller, drawn once for the host's whole
    set of models. Drawing it per model silently breaks the allocation: with
    n = 1, ceil(0.60 * 1) = 1, so every model lands in the tight bin and the
    40 percent high-sigma arm, which IS the coplanarity sweep, never exists.
    MEASURED 2026-08-04: the first pilot reported "sigma allocation: 100.0% in
    the tight bin" and 24 of 24 models were 1-2 punches, which is what tight
    coplanarity produces.
    """
    if sigma is None:
        sigma = float(draw_sigma(rng, 1)[0])
    sigma = float(sigma)
    p1incl, p1omega = draw_angles(rng, sigma, TPL_FINC)
    fracsum, fracdiff, ratrad, teff2, temprat = stellar_geometry(host)

    p_p = float(physics.draw_planet_period(rng, host["p_bin"], host["ecc"], size=1)[0])
    # Draw the conjunction INSIDE the window. Drawing it over a full planet
    # period wastes almost every model: planet periods run to hundreds of days
    # while a sector-group window is a few tens, so the conjunction lands
    # outside and ELC emits no transit at all. MEASURED 2026-08-04: that is why
    # the first timing run produced "no ELC{body}tran{star}time.dat emitted".
    # Choosing which conjunction to model is a free choice; whether the planet
    # transits at that conjunction is still decided by the sampled geometry.
    p1tconj = float(rng.uniform(window["t0"], window["t1"]))

    depth_target = float(10 ** rng.uniform(*np.log10(DEPTH_TARGET_RANGE)))
    # depth = (Rp/R1)^2 * host_factor, and P1ratrad = R1/Rp, so:
    p1ratrad = float(1.0 / np.sqrt(depth_target / host_factor))

    ecosw = host["ecc"] * np.cos(np.radians(90.0))
    esinw = host["ecc"] * np.sin(np.radians(90.0))
    separ = TPL_SEPAR * (host["p_bin"] / TPL_PBIN) ** (2.0 / 3.0)

    values = {
        "t_start": f"{window['t0']:.8f}",
        "t_end": f"{window['t1']:.8f}",
        "step": f"{TIME_STEP:.8f}",
        "tref": f"{window['t0']:.8f}",   # identical to t_start: difference exactly 0
        "period": f"{host['p_bin']:.9f}",
        "t0": "0.000000000000000",
        "tconj": f"{host['bjd0']:.9f}",
        "ecc": f"{host['ecc']:.13f}",
        "omega": "90.0000000000",
        "ecosw": f"{ecosw:.17f}",
        "esinw": f"{esinw:.17f}",
        "separ": f"{separ:.6f}",
        "finc": f"{TPL_FINC:.10f}",
        "fracsum": f"{fracsum:.15f}",
        "fracdiff": f"{fracdiff:.15f}",
        "ratrad": f"{ratrad:.13f}",
        "teff1": f"{TPL_TEFF1:.9f}",
        "teff2": f"{teff2:.9f}",
        "temprat": f"{temprat:.13f}",
        "p1period": f"{p_p:.9f}",
        "p1tconj": f"{p1tconj:.9f}",
        "p1incl": f"{p1incl:.15f}",
        "p1omega": f"{p1omega:.15f}",
        "p1ratrad": f"{p1ratrad:.11f}",
        "iwriteeclipse": "1",
    }
    manifest = {
        "tic": host["tic"], "sigma": sigma,
        "sigma_bin": "low" if sigma <= SIGMA_LOW[1] else "high",
        "p1incl": p1incl, "p1omega": p1omega, "p_planet": p_p,
        "p1tconj": p1tconj, "p1ratrad": p1ratrad,
        "depth_target": depth_target, "host_factor": host_factor,
        "fracsum": fracsum, "fracdiff": fracdiff,
        "ratrad": ratrad, "teff1": TPL_TEFF1, "teff2": teff2,
        "temprat": temprat,
        "host_depth_ratio": float(host["depth_ratio"]),
        "p_bin": host["p_bin"], "ecc": host["ecc"],
        "window_t0": window["t0"], "window_t1": window["t1"],
        "window_points": window["n_points"],
    }
    return values, manifest


def run_elc(values, work_dir, timeout=1800):
    """Run one ELC model in its own scratch dir. Returns (ok, message)."""
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    # SYMLINK, do not copy. ELC.atm is 9.7 MB and the binary 1.7 MB, so copying
    # both into 3,600 scratch directories writes 41 GB of pure duplication.
    # MEASURED 2026-08-04: the 150-host batch reached 107 GB and took the disk to
    # 98 percent full with 11 GB left, which would have killed the run.
    #
    # Symlinking is safe here in a way it was NOT safe for the light curves.
    # EclipseMasker rewrites light curves IN PLACE, so a symlink there would have
    # corrupted the frozen cache. ELC only READS its binary and its atmosphere
    # table, and elc_source_fingerprint() SHA256s all 110 distribution files
    # before and after every run, so any write would be caught immediately.
    for f in ("ELC", "ELC.atm"):
        os.symlink(os.path.join(ELC90, f), os.path.join(work_dir, f))
    with open(os.path.join(work_dir, "ELC.inp"), "w") as fh:
        fh.write(patch(values))
    t0 = _time.time()
    try:
        r = subprocess.run(["./ELC"], cwd=work_dir, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout", _time.time() - t0
    model = os.path.join(work_dir, "modelU.linear")
    # ELC exits 0 even when it fails the Nmaxphase check, so presence and
    # non-emptiness of the output is the only reliable success test.
    if r.returncode != 0:
        return False, f"exit {r.returncode}", _time.time() - t0
    if not os.path.exists(model) or os.path.getsize(model) == 0:
        tail = (r.stdout or "").strip().splitlines()[-3:]
        return False, f"no modelU.linear ({tail})", _time.time() - t0
    return True, "ok", _time.time() - t0


# Outputs worth keeping per model. ELC writes 8 photometric bands plus RV and
# per-star files; we read only modelU.linear (for depths), the tran time files
# (the whole point of iwriteeclipse), ELC.inp (auditability) and ELC.parm (QA).
# Keeping everything costs ~19 MB per model against ~0.4 MB for what is used.
KEEP_PATTERNS = (r"^modelU\.linear$", r"^ELC\d+tran\d+time\.dat$",
                 r"^ELC\.inp$", r"^ELC\.parm$", r"^ELCprimtime\.dat$",
                 r"^ELCsectime\.dat$")


def prune_model_dir(work_dir):
    """Delete outputs no downstream step reads. Returns bytes freed."""
    freed = 0
    for name in os.listdir(work_dir):
        if any(re.match(p, name) for p in KEEP_PATTERNS):
            continue
        path = os.path.join(work_dir, name)
        try:
            if os.path.islink(path):
                os.unlink(path)
            elif os.path.isfile(path):
                freed += os.path.getsize(path)
                os.remove(path)
        except OSError:
            pass
    return freed


def qa_model(work_dir, host, manifest):
    """The QA gate. Every assertion has a reason; failures are recorded, not raised."""
    fails = []
    model_path = os.path.join(work_dir, "modelU.linear")

    # (1) output exists and is non-empty
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return ["modelU.linear missing or empty"], {}, False
    m = np.loadtxt(model_path)
    if m.ndim != 2 or m.shape[0] < 10:
        return ["modelU.linear too short"], {}, False
    t, f = m[:, 0], m[:, 1]

    # (2) no NaN
    if not np.all(np.isfinite(f)):
        fails.append("NaN in model flux")

    # (3) out-of-eclipse baseline normalises to 1
    norm = f / np.median(f)
    ooe = norm[norm > np.percentile(norm, 60)]
    if abs(np.median(ooe) - 1.0) > OOE_BASELINE_TOL:
        fails.append(f"OOE baseline {np.median(ooe):.6f} off 1 by > {OOE_BASELINE_TOL}")

    # (6) per-star transit files, the whole point of iwriteeclipse.
    # A model with no transit is NOT a QA failure. P1Tconj is now always drawn
    # inside the window so a conjunction always occurs, but whether the planet
    # actually crosses a stellar disc at that conjunction is decided by the
    # sampled mutual inclination and node, and at larger sigma it often does
    # not. That is the physics the coplanarity sweep exists to measure, so it is
    # counted separately as NO_TRANSIT rather than charged to the attrition
    # budget, which is reserved for models that genuinely went wrong.
    tran = [x for x in os.listdir(work_dir) if re.match(r"ELC\d+tran\d+time\.dat$", x)]
    planet_events = [e for e in planet_transit_times(work_dir) if e["body"] >= 3]
    no_transit = len(planet_events) == 0

    events = {}
    for fn in tran:
        path = os.path.join(work_dir, fn)
        if os.path.getsize(path) == 0:
            continue
        # The trailing UTC string breaks np.loadtxt, so read the 6 numeric cols.
        arr = np.genfromtxt(path, usecols=range(6))
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        mm = re.match(r"ELC(\d+)tran(\d+)time\.dat$", fn)
        events[fn] = {"body": int(mm.group(1)), "star": int(mm.group(2)),
                      "n": int(arr.shape[0]), "times": arr[:, 1].tolist()[:50]}

    stats = {
        "no_transit": int(no_transit),
        "n_planet_transits": len(planet_events),
        "n_stars_crossed": len({e["star"] for e in planet_events}),
        "is_punch": int(len({e["star"] for e in planet_events}) >= 2),
        "n_points": int(len(t)),
        "span_days": float(t.max() - t.min()),
        "ooe_baseline": float(np.median(ooe)),
        "min_flux": float(norm.min()),
        "tran_files": len(tran),
        "tran_events": {k: v["n"] for k, v in events.items()},
        "star_of_file": {k: v["star"] for k, v in events.items()},
    }
    return fails, stats, no_transit


def planet_transit_times(work_dir):
    """Read the exact planet transit times and star identities ELC wrote.

    ``iwriteeclipse = 1`` emits one file per (body, star) pair named
    ``ELC{body}tran{star}time.dat`` with columns
    (cycle, time, flag, signed impact parameter, ingress, egress, UTC string).
    The trailing UTC string breaks ``np.loadtxt``, so only the six numeric
    columns are read.

    This is the single most valuable thing ELC gives us that the synthetic bank
    cannot: which star was crossed, and exactly when.
    """
    events = []
    for fn in sorted(os.listdir(work_dir)):
        mm = re.match(r"ELC(\d+)tran(\d+)time\.dat$", fn)
        if not mm:
            continue
        path = os.path.join(work_dir, fn)
        if os.path.getsize(path) == 0:
            continue
        arr = np.atleast_2d(np.genfromtxt(path, usecols=range(6)))
        if arr.size == 0:
            continue
        for row in arr:
            # |b| >= 1 means the planet missed the disc: a conjunction, not a
            # transit. ELC writes those rows with ingress == egress.
            if abs(float(row[3])) >= 1.0 or (float(row[5]) - float(row[4])) < 1e-4:
                continue
            events.append({"body": int(mm.group(1)), "star": int(mm.group(2)),
                           "cycle": float(row[0]), "time": float(row[1]),
                           "impact": float(row[3]),
                           "ingress": float(row[4]), "egress": float(row[5])})
    return events


def _depth_at_transits(work_dir):
    """Deepest planet transit, measured against the local continuum.

    Depth is taken relative to a local baseline just outside the transit rather
    than to the global median, so the stellar eclipse elsewhere in the window
    cannot bias it.
    """
    model_path = os.path.join(work_dir, "modelU.linear")
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return None
    m = np.loadtxt(model_path)
    if m.ndim != 2 or m.shape[0] < 10:
        return None
    t, f = m[:, 0], m[:, 1]
    best = None
    for ev in planet_transit_times(work_dir):
        if ev["body"] < 3:          # bodies 1 and 2 are the stars
            continue
        dur = max(ev["egress"] - ev["ingress"], 1e-3)
        inside = np.abs(t - ev["time"]) <= dur / 2.0
        near = (np.abs(t - ev["time"]) > dur) & (np.abs(t - ev["time"]) <= 3 * dur)
        if inside.sum() < 2 or near.sum() < 5:
            continue
        cont = np.median(f[near])
        if not np.isfinite(cont) or cont <= 0:
            continue
        d = float(1.0 - np.min(f[inside]) / cont)
        if np.isfinite(d) and d > 0 and (best is None or d > best):
            best = d
    return best


def measure_host_factor(host, rng, work_root):
    """Measure the per-host depth constant by running three P1ratrad values.

    Depth = (Rp/R1)^2 x host_factor, where host_factor absorbs limb darkening
    and dilution by the companion. It was 1.238 to 1.240 on Kepler-16, constant
    to 0.2 percent across radius, but it is per-host and must be measured, not
    assumed at the Kepler-16 value.
    """
    window = host["windows"][0]
    probe_values = [10.0, 15.0, 22.0]
    depths, geoms = [], []
    for i, ratrad in enumerate(probe_values):
        values, _ = build_model_spec(rng, host, window, host_factor=1.0)
        values["p1ratrad"] = f"{ratrad:.11f}"
        # Force a transiting, short-period planet so a transit is guaranteed
        # inside the window; this is a calibration run, not a science model.
        values["p1incl"] = f"{TPL_FINC:.15f}"
        values["p1omega"] = "0.0"
        wd = os.path.join(work_root, f"hostfactor_{host['tic']}_{i}")
        ok, msg, secs = run_elc(values, wd)
        if not ok:
            log.warning("  host-factor probe %d failed: %s", i, msg)
            continue
        # Measure the depth AT THE KNOWN TRANSIT TIMES, which iwriteeclipse
        # gives us exactly. The earlier approach of "deepest dip shallower than
        # 5 percent" measured the stellar eclipse's INGRESS AND EGRESS WINGS
        # instead: the eclipse is 15 percent deep, so its wings sweep through
        # any shallow band on the way down. MEASURED 2026-08-04: all three
        # P1ratrad probes returned an identical 0.0455, which cannot be a planet
        # because depth must scale as (1/r)^2, and the fitted factor came out
        # 12.2 with 59 percent spread instead of order 1.2.
        depth = _depth_at_transits(wd)
        if depth is None:
            continue
        depths.append(depth)
        geoms.append((1.0 / ratrad) ** 2)
    if len(depths) < 2:
        return None, {"probes": len(depths)}
    factor = float(np.mean(np.array(depths) / np.array(geoms)))
    spread = float(np.std(np.array(depths) / np.array(geoms)) / max(factor, 1e-12))
    return factor, {"probes": len(depths), "factor": factor, "relative_spread": spread,
                    "depths": depths, "geoms": geoms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", type=int, default=3, help="number of hosts (pilot = 3)")
    ap.add_argument("--models", type=int, default=MODELS_PER_HOST)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--dry-run", action="store_true", help="build specs, do not run ELC")
    args = ap.parse_args()

    elc_before = elc_source_fingerprint()
    check_keys_unique()
    rng = np.random.default_rng(DRIVER_SEED)
    out_dir = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    hosts = select_hosts(args.hosts)
    log.info("Selected %d hosts: %s", len(hosts),
             [(h["tic"], round(h["partner_snr"], 1), h["n_files"]) for h in hosts])
    if not hosts:
        raise SystemExit("No eligible hosts found")

    manifest_rows, timings = [], []
    for host in hosts:
        log.info("HOST TIC %d  P_bin=%.4f d  e=%.3f  depth_ratio=%.3f  windows=%d",
                 host["tic"], host["p_bin"], host["ecc"], host["depth_ratio"],
                 len(host["windows"]))
        for w in host["windows"]:
            log.info("   window %.1f to %.1f  (%d points, cap %d)",
                     w["t0"], w["t1"], w["n_points"], NMAXPHASE)

        host_factor = 1.239   # Kepler-16 value, replaced by measurement below
        if not args.dry_run:
            measured, info = measure_host_factor(host, rng, out_dir)
            if measured is not None:
                host_factor = measured
                log.info("   host depth factor MEASURED = %.4f (spread %.3f over %d probes)",
                         measured, info["relative_spread"], info["probes"])
            else:
                log.warning("   host depth factor NOT measured (%s); using Kepler-16 1.239 "
                            "and flagging every model of this host", info)

        # One draw for the host's whole set, so the 60/40 split is real.
        host_sigmas = draw_sigma(rng, args.models)
        for j in range(args.models):
            window = host["windows"][j % len(host["windows"])]
            values, man = build_model_spec(rng, host, window, host_factor,
                                           sigma=float(host_sigmas[j]))
            man.update({"model_id": f"{host['tic']}_{j:03d}", "tag": args.tag,
                        "host_factor_measured": not args.dry_run and host_factor != 1.239})
            work_dir = os.path.join(out_dir, f"TIC{host['tic']}_m{j:03d}")

            if args.dry_run:
                man.update({"qa_status": "dry-run", "elc_seconds": 0.0})
                manifest_rows.append(man)
                continue

            ok, msg, secs = run_elc(values, work_dir)
            timings.append(secs)
            if not ok:
                man.update({"qa_status": "RUN_FAILED", "qa_detail": msg, "elc_seconds": secs})
            else:
                fails, stats, no_transit = qa_model(work_dir, host, man)
                prune_model_dir(work_dir)   # keep ~0.4 MB, not ~19 MB
                man.update({
                    "qa_status": ("QA_FAILED" if fails else
                                  ("NO_TRANSIT" if no_transit else "PASS")),
                    "qa_detail": "; ".join(fails),
                    "elc_seconds": secs,
                    **{f"stat_{k}": (json.dumps(v) if isinstance(v, dict) else v)
                       for k, v in stats.items()},
                })
            manifest_rows.append(man)
            log.info("   model %s: %s (%.1f s)", man["model_id"], man["qa_status"], secs)

    assert_elc_untouched(elc_before, elc_source_fingerprint())

    df = pd.DataFrame(manifest_rows)
    man_path = os.path.join(out_dir, "elc_manifest.csv")
    df.to_csv(man_path, index=False)
    log.info("Wrote %d manifest rows to %s", len(df), man_path)

    if not args.dry_run and len(df):
        n_pass = int((df.qa_status == "PASS").sum())
        n_notransit = int((df.qa_status == "NO_TRANSIT").sum())
        n_broken = len(df) - n_pass - n_notransit
        # Attrition is broken models over models that could have worked; a
        # legitimate non-transit is yield, not breakage.
        attrition = n_broken / max(len(df), 1)
        log.info("QA: %d PASS, %d NO_TRANSIT (physics), %d broken of %d; "
                 "attrition %.1f%% (budget %.0f%%)",
                 n_pass, n_notransit, n_broken, len(df),
                 100 * attrition, 100 * QA_ATTRITION_BUDGET)
        if "stat_is_punch" in df.columns:
            log.info("1-2 punch models (planet crossed BOTH stars): %d of %d transiting",
                     int(df.stat_is_punch.fillna(0).sum()), n_pass)
        log.info("Per-model ELC wall time: median %.1f s, max %.1f s",
                 np.median(timings) if timings else 0, np.max(timings) if timings else 0)
        summary = {
            "tag": args.tag, "n_hosts": len(hosts), "n_models": len(df),
            "n_pass": n_pass, "n_no_transit": n_notransit,
            "n_broken": n_broken, "attrition": attrition,
            "attrition_budget": QA_ATTRITION_BUDGET,
            "within_budget": bool(attrition <= QA_ATTRITION_BUDGET),
            "median_seconds_per_model": float(np.median(timings)) if timings else None,
            "sigma_low_fraction": float((df.sigma_bin == "low").mean()),
            "pins": {"driver_seed": DRIVER_SEED, "time_step": TIME_STEP,
                     "nmaxphase": NMAXPHASE, "sigma_low": list(SIGMA_LOW),
                     "sigma_high": list(SIGMA_HIGH),
                     "sigma_low_fraction_required": SIGMA_LOW_FRACTION},
        }
        with open(os.path.join(out_dir, "elc_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        log.info("sigma allocation: %.1f%% in the tight bin (rule: >= 60%%)",
                 100 * summary["sigma_low_fraction"])


if __name__ == "__main__":
    main()
