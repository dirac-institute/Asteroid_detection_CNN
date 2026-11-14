"""Diffim mask planes: plane definitions from the MASK HDU header and per-detection plane fractions.

Current afw writes each plane as MSKN#### (name) + MSKM#### (bit VALUE, test ``mask & value``);
older files carry MP_<name> = bit index. Plane names are mapped to the column names used on the
research branch (SATURATED -> SAT, ...). A detection's fraction for a plane is the fraction of its
sample points whose mask has that plane set: 25 points along the trail, each the OR over +-1 px
perpendicular, or the 3x3 box around (x, y) for a point source (length < 1 px or not finite).
NaN where the mask is unavailable or the plane is not defined in that file.
"""
from __future__ import annotations

import numpy as np

ALIASES = {"SATURATED": "SAT", "COSMIC_RAY": "CR", "INTERPOLATED": "INTRP",
           "UNMASKED_NAN": "UNMASKEDNAN", "DETECTION_EDGE": "EDGE"}
# Instrumental planes: a trail on any of them is suspect (flag only, never a cut).
ART_PLANES = ["SPIKE", "SAT", "CR", "STREAK", "CROSSTALK", "SAT_TEMPLATE", "BAD", "SUSPECT", "EDGE",
              "NO_DATA", "VIGNETTED", "INTRP", "ITL_DIP", "UNMASKEDNAN", "HIGH_VARIANCE",
              "DETECTED_NEGATIVE"]
COLUMNS = [f"m_{p}" for p in ART_PLANES] + ["art_frac"]
NSAMP = 25       # points along the trail
HALFWID = 1      # +- px perpendicular to the trail


def plane_values(header) -> dict[str, int]:
    """{plane name: bit value} from a MASK HDU header, names mapped through ALIASES."""
    idx = {k[4:]: str(v).strip() for k, v in header.items() if k.startswith("MSKN") and k[4:].isdigit()}
    if idx:
        raw = {name: int(header["MSKM" + i]) for i, name in idx.items()}
    else:
        raw = {k[3:]: 1 << int(v) for k, v in header.items() if k.startswith("MP_")}
    if not raw:
        raise ValueError("mask header defines no planes (no MSKN/MSKM or MP_ cards)")
    return {ALIASES.get(n, n): v for n, v in raw.items()}


def fractions(mask, values, x, y, beta_deg, length_px) -> dict:
    """Per-detection plane fractions {m_<PLANE>: ..., art_frac: ...} for every plane in ART_PLANES.

    x, y (0-based px), beta_deg (from +x) and length_px are scalars or equal-length arrays; the
    values are floats or arrays to match. mask=None (unreadable) gives NaN everywhere. art_frac is
    the max over the defined planes."""
    scalar = np.ndim(x) == 0
    x, y, b, L = (np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (x, y, beta_deg, length_px))
    n = len(x)
    out = {c: np.full(n, np.nan) for c in COLUMNS}
    if mask is not None and n:
        H, W = mask.shape
        ok = np.isfinite(x) & np.isfinite(y)
        trail = ok & np.isfinite(L) & np.isfinite(b) & (L >= 1.0)
        x0, y0 = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
        rad = np.radians(np.where(trail, b, 0.0))
        dx, dy = np.cos(rad)[:, None], np.sin(rad)[:, None]
        ts = np.linspace(-0.5, 0.5, NSAMP)[None, :] * np.where(trail, L, 0.0)[:, None]
        xs = np.clip(np.round(x0[:, None] + ts * dx), 0, W - 1)
        ys = np.clip(np.round(y0[:, None] + ts * dy), 0, H - 1)
        acc = np.zeros((n, NSAMP), dtype=np.int64)
        for w in range(-HALFWID, HALFWID + 1):
            xw = np.clip(xs + np.round(-w * dy), 0, W - 1).astype(np.intp)
            yw = np.clip(ys + np.round(w * dx), 0, H - 1).astype(np.intp)
            acc |= mask[yw, xw].astype(np.int64)
        off = np.arange(-1, 2)
        bx = np.clip(np.round(x0)[:, None, None] + off[None, None, :], 0, W - 1).astype(np.intp)
        by = np.clip(np.round(y0)[:, None, None] + off[None, :, None], 0, H - 1).astype(np.intp)
        box = mask[by, bx].reshape(n, 9).astype(np.int64)
        point = ok & ~trail
        for p in ART_PLANES:
            if p in values:
                v = values[p]
                f = np.where(trail, np.mean((acc & v) != 0, axis=1), np.mean((box & v) != 0, axis=1))
                out[f"m_{p}"] = np.where(trail | point, f, np.nan)
        out["art_frac"] = np.fmax.reduce(np.stack([out[f"m_{p}"] for p in ART_PLANES]), axis=0)
    return {c: float(v[0]) for c, v in out.items()} if scalar else out
