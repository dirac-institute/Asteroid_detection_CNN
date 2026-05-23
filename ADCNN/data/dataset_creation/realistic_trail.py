"""Realistic asteroid-trail renderer for source injection.

LSST's stock `make_galsim_trail` renders a trail as a perfectly uniform,
infinitely-thin, perfectly-straight `galsim.Box(L, 1e-6)` convolved with the PSF.
The sim-to-real diagnostic (experiments/explore_simreal_gap) showed this is too
idealized: real trail-candidates have systematically lower line-coherence features
(oriented-aggregator mean, matched-filter SNR, integrated logit, elongation) than
these clean injections, so the stage-2 RF trained on them rejects real trails.

This module renders a trail as a sum of PSF point-components along the path with:
  - a non-uniform LIGHT CURVE (rotational modulation) along the trail,
  - TAPERED ends (acceleration / partial-detection at the extremities),
  - a slight CURVATURE (parabolic lateral deflection).
GalSim convolves the components with the PSF and the injection engine applies the
same WCS transform + flux convention as the stock trail, so this is a drop-in
replacement installed by monkeypatching `inject_engine.make_galsim_trail`.

All morphology parameters are drawn from PHYSICAL priors (light-curve amplitudes,
mild curvature) seeded per-injection — they are NOT fit to the real test set, so
training on the resulting synthetic data introduces no test leakage.
"""
from __future__ import annotations
import numpy as np

# physical priors (fixed a priori; not tuned to real data) ----------------------
LC_AMP_MAX = 0.6      # max fractional light-curve modulation (≈0.5 mag pk-pk)
LC_NCYCLES = (0.5, 2.5)   # rotational cycles spanned along the trail
TAPER_FRAC = (0.05, 0.25)  # fraction of each end that tapers (cosine)
CURV_SAGITTA = 0.06   # max curvature sagitta as fraction of trail length
N_PER_PIX = 2.0       # path samples per pixel
_SEED_SALT = 0x9E3779B1


def _trail_profile(L_pix: float, rng: np.random.Generator):
    """Return (s, perp, w): along-track positions, perpendicular offsets (curvature)
    and per-sample flux weights (light curve x taper), all in trail-frame pixels.
    w is normalized to sum 1."""
    L = float(L_pix)
    n = max(int(np.ceil(L * N_PER_PIX)) + 1, 11)
    s = np.linspace(-0.5 * L, 0.5 * L, n)
    u = (s + 0.5 * L) / max(L, 1e-6)            # 0..1 along trail

    # light curve: sinusoidal rotational modulation
    amp = rng.uniform(0.0, LC_AMP_MAX)
    ncyc = rng.uniform(*LC_NCYCLES)
    phase = rng.uniform(0.0, 2 * np.pi)
    lc = 1.0 + amp * np.sin(2 * np.pi * ncyc * u + phase)

    # tapered ends (cosine ramp over a random fraction at each end)
    tf = rng.uniform(*TAPER_FRAC)
    taper = np.ones_like(u)
    ramp = u < tf
    taper[ramp] = 0.5 * (1 - np.cos(np.pi * u[ramp] / tf))
    ramp2 = u > (1 - tf)
    taper[ramp2] = 0.5 * (1 - np.cos(np.pi * (1 - u[ramp2]) / tf))

    w = np.clip(lc, 0.05, None) * taper
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    # slight curvature: parabolic lateral deflection, random sign/magnitude
    kappa = rng.uniform(-CURV_SAGITTA, CURV_SAGITTA)
    perp = kappa * L * (1.0 - (2.0 * s / max(L, 1e-6)) ** 2)
    return s, perp, w


def make_galsim_trail_realistic(source_data, wcs, sky_coords, inst_flux,
                                trail_thickness: float = 1e-6):
    """Drop-in replacement for lsst.source.injection.inject_engine.make_galsim_trail.

    Builds a non-uniform, slightly-curved, tapered trail as a galsim.Add of shifted
    DeltaFunctions (each becomes a PSF after the engine's PSF convolution), then
    applies the SAME beta-rotation, flux convention and WCS transform as the stock
    trail so downstream behaviour is identical apart from morphology.
    """
    import galsim
    from lsst.geom import arcseconds

    L = float(source_data["trail_length"])
    # Per-injection RNG. injection_id alone is NOT unique across panels — it is the
    # per-detector row index 0..n_inject-1 (simulate.py adds it as `k`),
    # so seeding on it would repeat the same morphology draw on every panel. Mix the
    # sky position (ra/dec) and L into the seed so each injected trail gets a distinct
    # light curve / taper / curvature, using only integer arithmetic so the result is
    # reproducible across worker processes (Python's hash() is salted for some types).
    ra = float(sky_coords.getRa().asDegrees())
    dec = float(sky_coords.getDec().asDegrees())
    try:
        sid = int(source_data["injection_id"])
    except Exception:
        sid = 0
    seed = (sid * 2654435761
            + int(round(ra * 1e5)) * 2246822519
            + int(round((dec + 90.0) * 1e5)) * 3266489917
            + int(round(L * 100)) * 668265263
            + _SEED_SALT) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    s, perp, w = _trail_profile(L, rng)
    comps = [galsim.DeltaFunction(flux=float(wi)).shift(float(si), float(pi))
             for si, pi, wi in zip(s, perp, w)]
    obj = galsim.Add(comps)
    try:
        obj = obj.rotate(source_data["beta"] * galsim.degrees)
    except (KeyError, TypeError):
        pass
    # same flux convention as stock make_galsim_trail (Box carried flux*L)
    obj = obj.withFlux(inst_flux * L)
    linear_wcs = wcs.linearizePixelToSky(sky_coords, arcseconds)
    mat = linear_wcs.getMatrix()
    obj = obj.transform(mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])
    obj *= 1.0 / np.abs(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    return obj


def install(verbose: bool = True):
    """Monkeypatch the stock trail renderer with the realistic one. Call once
    before ExposureInjectTask.run()."""
    import lsst.source.injection.inject_engine as eng
    if getattr(eng, "_realistic_trail_installed", False):
        return
    eng._stock_make_galsim_trail = eng.make_galsim_trail
    eng.make_galsim_trail = make_galsim_trail_realistic
    eng._realistic_trail_installed = True
    if verbose:
        print("[realistic_trail] installed non-uniform/tapered/curved trail renderer",
              flush=True)
