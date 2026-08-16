#!/usr/bin/env python3
"""Golden-style trail-overlay figures for a 2-visit alert stream (QA / vetting figures).

One PNG per alert, one panel per member epoch. Each panel is a ZScale diffim cutout with:

  * the ADCNN detection itself (cyan): measured trail endpoints (``ra0/dec0 -> ra1/dec1``
    from the dets catalog), score / matched-filter SNR / trail length, plus a LIVE forced
    trail-capsule SNR re-measured on the pixels (``pixel_vet.forced_at0`` -- the same
    statistic the production pixel vet uses);
  * every other ADCNN detection in the field of view (orange, score-tagged);
  * every OTHER member epoch's position on this panel (yellow dashed capsule + forced
    ``snr_at0``): persistent flux there says "static", empty pixels say "it moved" -- for
    long-gap pairs whose counterpart lands on another detector, the off-panel direction
    and distance are annotated instead;
  * the motion solution as a green arrow, a 10 arcsec scale bar;
  * for ``staticVeto``-flagged members (alert schema >= 1.4), the vetoing mag<20 static
    from the catalog as a red dashed circle at the veto radius.

Pixels are read through ``pixel_vet.PanelStore`` (Butler-datastore URIs, local or S3
in-memory via diffim_io), so this runs in the torch env -- no LSST stack needed.
QA-only: reads the alert stream and dets catalog, never modifies pipeline outputs.

Usage (from the repo root):
    PYTHONPATH=. python -m ADCNN.qa.trail_overlays \
        --alerts ADCNN/pipelines/heliolinc/run_embargo_night/expt_staticveto/prod05/alerts.jsonl \
        --dets   ADCNN/pipelines/heliolinc/run_embargo_night/adcnn_dets_masked.csv \
        --out-dir ADCNN/pipelines/heliolinc/run_embargo_night/expt_staticveto/prod05/report \
        [--static-catalog .../static_catalog.parquet] [--static-mag-max 20] \
        [--static-radius-arcsec 3] [--half-px 95] [--dpi 160]
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

from ADCNN.linking.pixel_vet import PanelStore, forced_at0, PXSCALE

EXPTIME_S = 30.0      # per-snap exposure used for the capsule trail length
HALFW_PX = 2.0        # capsule half-width (matches the production pixel vet)


def _unit(ra, dec):
    ra, dec = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def match_det_row(dets, epoch, tol_arcsec=1.0):
    """The dets-catalog row for one alert member epoch (visit+detector, nearest position).

    Fail-loud: raises ValueError if no det on that panel lies within ``tol_arcsec`` --
    an alert member that cannot be traced back to the dets catalog is a bookkeeping bug,
    never something to paper over with an empty panel."""
    sub = dets[(dets.visit == epoch["visit"]) & (dets.detector == epoch["detector"])]
    if not len(sub):
        raise ValueError(f"no dets on visit {epoch['visit']} det {epoch['detector']}")
    du = _unit(sub.ra.to_numpy(), sub.dec.to_numpy()) - _unit([epoch["ra"]], [epoch["dec"]])[0]
    i = int(np.argmin(np.linalg.norm(du, axis=1)))
    sep = np.degrees(2 * np.arcsin(min(np.linalg.norm(du[i]) / 2, 1))) * 3600
    if sep > tol_arcsec:
        raise ValueError(f"nearest det {sep:.2f}\" from alert member "
                         f"(visit {epoch['visit']} det {epoch['detector']}) > {tol_arcsec}\"")
    return sub.iloc[i]


def capsule_outline(wcs, ra, dec, rate_degday, pa_deg, exptime_s=EXPTIME_S, halfw_px=HALFW_PX):
    """Closed 5-point pixel path of the forced-photometry capsule (rotated rectangle)
    centred on (ra, dec), long axis along the motion PA, length = rate x exptime.
    Returns ((xs, ys), (x0, y0))."""
    x0, y0 = map(float, wcs.world_to_pixel_values(ra, dec))
    dd = 1e-4
    ra1 = ra + dd * np.sin(np.radians(pa_deg)) / max(np.cos(np.radians(dec)), 1e-9)
    dec1 = dec + dd * np.cos(np.radians(pa_deg))
    x1, y1 = map(float, wcs.world_to_pixel_values(ra1, dec1))
    ux, uy = x1 - x0, y1 - y0
    n = np.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    px, py = -uy, ux
    L = max(rate_degday * exptime_s / 86400.0 * 3600.0 / PXSCALE, 2.0)
    h, w = L / 2, halfw_px
    corners = [(x0 + sx * h * ux + sy * w * px, y0 + sx * h * uy + sy * w * py)
               for sx, sy in [(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]]
    return list(zip(*corners)), (x0, y0)


def _raft(path):
    m = re.search(r"_(R\d\d)_(S\d\d)", str(path))
    return f"{m.group(1)}_{m.group(2)}" if m else "?"


def _night_label(night_mjd):
    from astropy.time import Time
    return Time(float(night_mjd), format="mjd").iso[:10]


def render_alert(alert, dets, store, out_path, *, statics=None, half_px=95, dpi=160,
                 rank=None):
    """Render one alert to ``out_path``. ``statics`` = (kdtree, ra, dec, mag, radius_arcsec)
    for the staticVeto marker, or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    eps = sorted(alert["epochs"], key=lambda e: e["mjd"])
    rows = [match_det_row(dets, e) for e in eps]
    rate, pa = alert["motion"]["rate_degday"], alert["motion"]["pa_deg"]
    sv = alert.get("staticVeto") or {}
    label = ("CONFIDENT" if (alert.get("fpp") or {}).get("confident") else
             ("CLEAN" if sv.get("nStaticMembers", 0) == 0 else "STATIC-FLAGGED"))

    fig, axes = plt.subplots(1, len(eps), figsize=(6.6 * len(eps), 7.2), squeeze=False)
    zs = ZScaleInterval()
    for k, (ax, ep, me) in enumerate(zip(axes[0], eps, rows)):
        other = rows[1 - k] if len(rows) == 2 else rows[(k + 1) % len(rows)]
        panel = store.get(me.fits_path)
        img, var, mask, w, badval = panel
        f_det = forced_at0(panel, float(me.ra), float(me.dec), rate, pa,
                           exptime_s=EXPTIME_S, halfw_px=HALFW_PX)
        cx, cy = float(me.x), float(me.y)
        ox, oy = map(float, w.world_to_pixel_values(other.ra, other.dec))
        on_panel = (0 <= ox < img.shape[1]) and (0 <= oy < img.shape[0])
        near = on_panel and np.hypot(ox - cx, oy - cy) < 2 * half_px - 25
        mx, my = ((cx + ox) / 2, (cy + oy) / 2) if near else (cx, cy)
        x0, x1 = int(max(0, mx - half_px)), int(min(img.shape[1], mx + half_px))
        y0, y1 = int(max(0, my - half_px)), int(min(img.shape[0], my + half_px))
        cut = np.asarray(img[y0:y1, x0:x1], dtype=float)
        vmin, vmax = zs.get_limits(cut[np.isfinite(cut)])
        ax.imshow(cut, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                  extent=[x0, x1, y0, y1])

        same = dets[(dets.visit == ep["visit"]) & (dets.detector == ep["detector"])]
        for _, r in same.iterrows():
            if not (x0 < r.x < x1 and y0 < r.y < y1):
                continue
            is_me = abs(r.x - cx) < 0.5 and abs(r.y - cy) < 0.5
            col = "#00e5ff" if is_me else "#ff9800"
            (ex0, ey0), (ex1, ey1) = w.all_world2pix([[r.ra0, r.dec0], [r.ra1, r.dec1]], 0)
            ax.plot([ex0, ex1], [ey0, ey1], color=col, lw=2.2 if is_me else 1.4,
                    solid_capstyle="round", alpha=0.95 if is_me else 0.8)
            ax.add_patch(plt.Circle((r.x, r.y), 14, fill=False, color=col,
                                    lw=2.0 if is_me else 1.2, alpha=0.95 if is_me else 0.8))
            if is_me:
                # forced_at0 returns None when the position is off-panel / the cutout degenerate --
                # that is its documented contract, and 8 of 9 campaign nights lost their ENTIRE
                # report package to an unguarded subscript of it (best-effort catch upstream, so the
                # first off-panel alert silently ended the report for the night).
                _fd = (f"forced capsule SNR = {f_det['snr']:+.1f}$\\sigma$ ({f_det['n_good']} px)"
                       if f_det else "forced capsule SNR unmeasurable (off panel)")
                ax.annotate(f"S={r.score:.2f}  mfSNR={r.mf_snr:.1f}  L={r.length:.1f}px\n" + _fd,
                            (r.x, r.y), xytext=(18, 16), textcoords="offset points",
                            color=col, fontsize=9, fontweight="bold")
            else:
                ax.annotate(f"S={r.score:.2f}", (r.x, r.y), xytext=(16, 12),
                            textcoords="offset points", color=col, fontsize=8)

        if on_panel:
            f_other = forced_at0(panel, float(other.ra), float(other.dec), rate, pa,
                                 exptime_s=EXPTIME_S, halfw_px=HALFW_PX)
            if near:
                (cxs, cys), _ = capsule_outline(w, float(other.ra), float(other.dec), rate, pa)
                ax.plot(cxs, cys, color="#ffee58", lw=1.6, ls="--")
                ax.add_patch(plt.Circle((ox, oy), 14, fill=False, color="#ffee58",
                                        lw=1.4, ls=":"))
                _fo = (f"forced snr_at0 = {f_other['snr']:+.2f}$\\sigma$ "
                       f"({f_other['n_good']} px clean, badfrac {f_other['badfrac']:.0%})"
                       if f_other else "forced snr_at0 unmeasurable (off panel / degenerate)")
                ax.annotate("position @ other epoch\n" + _fo,
                            (ox, oy), xytext=(-16, -56) if k == 0 else (-120, -56),
                            textcoords="offset points", color="#c8b900", fontsize=8.5)
            else:
                _fo = (f"forced snr_at0 there = {f_other['snr']:+.2f}$\\sigma$"
                       if f_other else "forced snr_at0 there unmeasurable")
                ax.annotate(f"other epoch on this detector, "
                            f"{np.hypot(ox - cx, oy - cy) * PXSCALE:.0f}\" away (off cutout)\n" + _fo,
                            (0.03, 0.03), xycoords="axes fraction",
                            color="#c8b900", fontsize=8.5)
        else:
            sep = np.degrees(2 * np.arcsin(min(np.linalg.norm(
                _unit([ep["ra"]], [ep["dec"]])[0]
                - _unit([other.ra], [other.dec])[0]) / 2, 1))) * 3600
            ax.annotate(f"other epoch {sep:.0f}\" away -- different detector (off panel)",
                        (0.03, 0.03), xycoords="axes fraction", color="#c8b900", fontsize=8.5)

        if near:
            ax.annotate("", xy=(ox, oy) if k == 0 else (cx, cy),
                        xytext=(cx, cy) if k == 0 else (ox, oy),
                        arrowprops=dict(arrowstyle="->", color="#76ff03", lw=1.6, alpha=0.9))
        else:
            dr = np.array([ox - cx, oy - cy])
            dr = dr / max(np.hypot(*dr), 1e-9) * 55
            base = (cx, cy) if k == 0 else (cx - dr[0], cy - dr[1])
            ax.annotate("", xy=(base[0] + dr[0], base[1] + dr[1]), xytext=base,
                        arrowprops=dict(arrowstyle="->", color="#76ff03", lw=1.6, alpha=0.9))

        mem = next((m for m in sv.get("members", []) if m["visit"] == ep["visit"]), None)
        if mem and mem.get("isStatic") and statics is not None:
            tree, sra, sdec, smag, rad = statics
            _, i = tree.query(_unit([ep["ra"]], [ep["dec"]])[0], k=1)
            xs, ys = map(float, w.world_to_pixel_values(sra[i], sdec[i]))
            ax.add_patch(plt.Circle((xs, ys), rad / PXSCALE, fill=False,
                                    color="#ff5722", lw=1.8, ls="--"))
            ax.annotate(f"mag<20 static, {mem['sepArcsec']:.2f}\" (mag {mem['staticMag']:.1f})\n"
                        f"$\\Rightarrow$ staticVeto FLAG",
                        (xs, ys), xytext=(14, -44), textcoords="offset points",
                        color="#ff5722", fontsize=8.5, fontweight="bold")

        sb = 10.0 / PXSCALE
        ax.plot([x0 + 12, x0 + 12 + sb], [y0 + 12, y0 + 12], color="w", lw=2.5)
        ax.text(x0 + 12 + sb / 2, y0 + 18, '10"', color="w", ha="center", fontsize=9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Epoch {k + 1}   visit {ep['visit']}   det {ep['detector']} "
                     f"({_raft(me.fits_path)})\n"
                     f"MJD {me.mjd:.5f}   RA {me.ra:.5f}  Dec {me.dec:.5f}", fontsize=10)

    dt_s = (rows[-1].mjd - rows[0].mjd) * 86400.0
    dt_lab = f"{dt_s:.0f} s" if dt_s < 120 else f"{dt_s / 60:.1f} min"
    fig.suptitle(f"{alert['alertId']}  ({label})  --  ADCNN detections + forced "
                 f"trail-capsule photometry, night {_night_label(alert['night'])},  "
                 f"$\\Delta$t = {dt_lab},  rate {rate:.2f} $\\pm$ "
                 f"{3 * alert['motion']['rate_sigma_degday']:.2f} deg/day",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, facecolor="w")
    plt.close(fig)


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alerts", required=True, help="alerts.jsonl (schema >= 1.2)")
    p.add_argument("--dets", required=True, help="adcnn dets CSV (masked; must carry fits_path)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--static-catalog", default=None,
                   help="optional static catalog parquet/csv for the staticVeto marker")
    p.add_argument("--static-mag-max", type=float, default=20.0)
    p.add_argument("--static-radius-arcsec", type=float, default=3.0)
    p.add_argument("--half-px", type=int, default=95, help="cutout half-size in px")
    p.add_argument("--dpi", type=int, default=160)
    a = p.parse_args(argv)

    alerts = [json.loads(l) for l in open(a.alerts)]
    dets = pd.read_csv(a.dets)
    os.makedirs(a.out_dir, exist_ok=True)

    statics = None
    if a.static_catalog:
        from scipy.spatial import cKDTree
        sc = (pd.read_parquet(a.static_catalog)
              if str(a.static_catalog).endswith((".parquet", ".pq"))
              else pd.read_csv(a.static_catalog))
        magcol = "mag" if "mag" in sc.columns else "mag_best"
        sc = sc[np.isfinite(sc[magcol]) & (sc[magcol] < a.static_mag_max)]
        statics = (cKDTree(_unit(sc.ra.to_numpy(), sc.dec.to_numpy())),
                   sc.ra.to_numpy(), sc.dec.to_numpy(), sc[magcol].to_numpy(),
                   a.static_radius_arcsec)

    store = PanelStore(max_panels=4)
    n_fail = 0
    for rank, alert in enumerate(alerts):
        out = os.path.join(a.out_dir, f"overlay_rank{rank:02d}_{alert['alertId']}.png")
        # PER-ALERT containment. The caller's try/except wraps the WHOLE report, so before this one
        # failing alert ended the report for the night -- 8 of 9 campaign nights shipped a partial
        # report because a single alert's other-epoch position fell off-panel. One bad alert now
        # costs one overlay, and the failure is named instead of truncating silently.
        try:
            render_alert(alert, dets, store, out, statics=statics,
                         half_px=a.half_px, dpi=a.dpi, rank=rank)
        except Exception as e:  # noqa: BLE001 -- report is best-effort per alert, not per night
            n_fail += 1
            print(f"[trail-overlays] WARNING: rank{rank:02d} {alert.get('alertId')} failed "
                  f"({type(e).__name__}: {e}) -- skipped, continuing", flush=True)
    if n_fail:
        print(f"[trail-overlays] {n_fail} of {len(alerts)} overlays failed and were skipped", flush=True)
        print(f"[trail-overlays] saved {out}", flush=True)
    print(f"[trail-overlays] {len(alerts)} alert figure(s) -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
