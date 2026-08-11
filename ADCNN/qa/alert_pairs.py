#!/usr/bin/env python3
"""One image FILE per alert: each epoch zoomed with the detection outlined, plus a zoomed-out
view showing where it was and where it went, joined by a line.

The contact sheets answer "what does the night look like"; this answers "is THIS alert real",
and is the file to open when something looks interesting.

Layout, left to right:
  [ epoch 0 zoom ]  [ epoch 1 zoom ]  ... [ WIDE: both positions + connecting line ]

  * Each zoom is ~19 arcsec across, centred on that epoch's detection, with an ellipse outlining
    the measured trail (length and position angle as measured, so a wrong-looking outline is
    itself diagnostic).
  * The wide panel is cut from the first epoch's image and sized to contain every member
    position. That is the only frame where the motion is visible at all: at 1-8 deg/day over a
    ~20 min gap the detections are 250-2000 px (50-400 arcsec) apart, hundreds of times the zoom
    field. Positions are outlined and joined by an arrow so the displacement reads at a glance,
    with the separation and a scale bar annotated.

Because the wide panel comes from ONE epoch's image, a real mover shows a source at its own
position there and nothing at the other -- a static residual shows one at both.

Files are named by rank so they sort in ranking order:
  alert_00000_p0.87_2v_61221_000123_CLEAN.png

Usage:
  python -m ADCNN.qa.alert_pairs --alerts stream/alerts.jsonl --cutouts stream/cutouts.npz \
      --out-dir stream/pairs [--top-n 2000]
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

VMIN, VMAX = -2.0, 6.0


def _num(d, key, default=float("nan")):
    """dict.get(k, default) returns None when the key EXISTS and is null -- which some alert
    fields are -- and None then blows up an f-string format spec. Coerce to a float always."""
    v = d.get(key, default) if isinstance(d, dict) else default
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

PIXSCALE = 0.2
EPCOL = ["#ff3b30", "#38d6ff", "#ffd23b", "#8cff6b"]      # per-epoch colour


def _trail_outline(ax, ends, colour, lw=1.2, pad=3.0):
    """Ellipse around the MEASURED trail, drawn from its two projected endpoints so the drawn
    elongation and angle are the measurement itself, in whatever frame `ends` was projected into
    (no re-derivation from beta, which is only valid in its own epoch's panel frame)."""
    from matplotlib.patches import Ellipse
    (x0, y0), (x1, y1) = ends
    if not np.isfinite([x0, y0, x1, y1]).all():
        return None
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    L = float(np.hypot(x1 - x0, y1 - y0))
    ang = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    ax.add_patch(Ellipse((cx, cy), width=L + 2 * pad, height=2 * pad, angle=ang,
                         fill=False, edgecolor=colour, lw=lw))
    return (cx, cy, L, ang)


def _orientation_ray(ax, cx, cy, ang_deg, half_len, colour, lw=1.0):
    """Dashed ray through (cx,cy) at the trail's angle. In the wide view the true trail is only a
    couple of pixels long, so its ORIENTATION -- the thing to compare against the motion arrow --
    would be invisible at true scale. This shows direction only; the solid ellipse shows size."""
    dx = half_len * np.cos(np.radians(ang_deg))
    dy = half_len * np.sin(np.radians(ang_deg))
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=colour, lw=lw, ls=(0, (3, 2)), alpha=0.9)


def _assert_cache_matches(alerts_path, cutouts_npz, n_alerts):
    """Delegates to the sheets implementation -- ONE guard, so the two renderers cannot drift.

    The previous local copy compared counts only, which cannot see a reordering (see alert_sheets).
    """
    from ADCNN.qa.alert_sheets import _assert_cache_matches as _shared
    return _shared(alerts_path, cutouts_npz, n_alerts)


def render(alerts_path, cutouts_npz, out_dir, top_n=None, dpi=120, workers=1, _slice=None):
    """Render per-alert figures. `workers`>1 splits the alert list across processes; each loads the
    cutout cache itself (~1.3 GB resident per worker -- an npz decompresses whole arrays on first
    access, so there is nothing cheaper to share) and writes its own files, which is worth it at
    10k alerts: ~0.4 s per figure is over an hour serially."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ADCNN.qa.alert_report import classify

    alerts = [json.loads(l) for l in open(alerts_path)] if os.path.getsize(alerts_path) else []
    if top_n:
        alerts = alerts[:top_n]
    if not alerts:
        print("[pairs] 0 alerts", flush=True)
        return
    # Clear stale renders once, in the OWNING call, before any work. alertIds are assigned
    # sequentially per link run and are NOT stable across runs (cross-run identity is member
    # position only), so a file left by a previous link names an alert that no longer exists and
    # its name COLLIDES with a different object now. This MUST run before the parallel fan-out:
    # the parent returns right after spawning workers, so a clear placed after that (as it was)
    # never ran in --workers mode -- 20260713 kept 15,615 files for 8,273 alerts that way.
    if _slice is None:
        import glob as _glob
        os.makedirs(out_dir, exist_ok=True)
        stale = _glob.glob(os.path.join(out_dir, "alert_*.png"))
        for f in stale:
            os.remove(f)
        if stale:
            print(f"[pairs] cleared {len(stale)} stale render(s) from {out_dir}", flush=True)
    if workers > 1 and _slice is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        n = len(alerts)
        step = (n + workers - 1) // workers
        chunks = [(i, min(i + step, n)) for i in range(0, n, step)]
        print(f"[pairs] {n} alerts over {len(chunks)} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(render, alerts_path, cutouts_npz, out_dir, top_n, dpi, 1, c)
                    for c in chunks]
            done = sum(f.result() or 0 for f in futs)
        print(f"[pairs] wrote {done} per-alert images -> {out_dir}", flush=True)
        return done
    lo, hi = _slice if _slice else (0, len(alerts))
    _assert_cache_matches(alerts_path, cutouts_npz, len(alerts))
    z = np.load(cutouts_npz)
    stamps, a_ix, e_ix = z["stamps"], z["alert"], z["epoch"]
    zends = z["zoom_ends"] if "zoom_ends" in z.files else None
    wends = z["wide_ends"] if "wide_ends" in z.files else None
    zi = {(int(a_ix[i]), int(e_ix[i])): i for i in range(len(a_ix))}
    wide, w_al = z["wide"], z["wide_alert"]
    w_pos, w_apx = z["wide_pos"], z["wide_apx"]
    wi = {int(w_al[i]): i for i in range(len(w_al))}

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for rank in range(lo, hi):
        al = alerts[rank]
        eps = al["epochs"]
        n = len(eps)
        cls = classify(al)[0]
        pr = (al.get("ranking") or {}).get("pReal")
        fig, axs = plt.subplots(1, n + 1, figsize=(2.55 * (n + 1) + 0.5, 3.1), squeeze=False)
        axs = axs[0]

        for i, ep in enumerate(eps):
            ax = axs[i]
            ax.set_xticks([]); ax.set_yticks([])
            k = zi.get((rank, i))
            if k is None:
                ax.text(.5, .5, "no pixels", ha="center", va="center", fontsize=7); ax.axis("off")
                continue
            st = stamps[k].astype(np.float32)
            ax.imshow(st, origin="lower", cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="nearest")
            if zends is not None and k < len(zends):
                _trail_outline(ax, zends[k], EPCOL[i % len(EPCOL)], lw=1.3, pad=3.5)
            for sp in ax.spines.values():
                sp.set_color(EPCOL[i % len(EPCOL)]); sp.set_linewidth(1.6)
            ax.set_title(f"epoch {i}  v{ep['visit']}\nsnr {_num(ep, 'snr'):.1f}  "
                         f"score {_num(ep, 'score'):.2f}  "
                         f"len {_num(ep, 'trail_len_px', 0.0):.0f}px",
                         fontsize=6.8, color=EPCOL[i % len(EPCOL)], pad=2.5)

        # ---- wide view: both positions, outlined, joined -------------------------------------
        axw = axs[n]
        axw.set_xticks([]); axw.set_yticks([])
        wk = wi.get(rank)
        if wk is None:
            axw.text(.5, .5, "no wide view", ha="center", va="center", fontsize=7); axw.axis("off")
        else:
            W = wide[wk].astype(np.float32)
            axw.imshow(W, origin="lower", cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="nearest")
            apx = float(w_apx[wk])                     # arcsec per wide pixel
            sc = PIXSCALE / apx if apx > 0 else 1.0    # source px -> wide px
            pts, angs = [], []
            ray = 0.11 * W.shape[0]
            for i, ep in enumerate(eps):
                p = w_pos[wk][i]
                if not np.isfinite(p).all():
                    continue
                pts.append((float(p[0]), float(p[1])))
                col = EPCOL[i % len(EPCOL)]
                got = None
                if wends is not None and wk < len(wends):
                    got = _trail_outline(axw, wends[wk][i], col, lw=1.3, pad=2.2)
                if got is not None:
                    _orientation_ray(axw, got[0], got[1], got[3], ray, col, lw=1.0)
                    angs.append(got[3])
                axw.annotate(f"{i}", (p[0], p[1]), textcoords="offset points", xytext=(7, 6),
                             fontsize=7, color=col, weight="bold")
            if len(pts) >= 2:
                for j in range(len(pts) - 1):
                    axw.annotate("", xy=pts[j + 1], xytext=pts[j],
                                 arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1.1,
                                                 alpha=0.85, shrinkA=8, shrinkB=8))
                sep = np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) * apx
                mang = float(np.degrees(np.arctan2(pts[-1][1] - pts[0][1],
                                                   pts[-1][0] - pts[0][0])))
                lbl = f"separation {sep:.0f}\"  ({sep/60:.1f}')"
                if angs:
                    # trail vs motion, modulo 180 deg (a trail has no head/tail)
                    dmax = max(abs(((an - mang + 90.0) % 180.0) - 90.0) for an in angs)
                    lbl += f"   |   trail vs motion {dmax:.0f}°"
                    lbl += "  consistent" if dmax <= 20 else "  MISMATCH"
                axw.set_xlabel(lbl, fontsize=6.2, labelpad=1.5,
                               color=("#1a7f37" if (angs and dmax <= 20) else "#b00020") if angs else "k")
            if apx > 0:                                 # 30" scale bar
                L = 30.0 / apx
                y0 = 0.06 * W.shape[0]
                axw.plot([0.06 * W.shape[1], 0.06 * W.shape[1] + L], [y0, y0], color="w", lw=2)
                axw.text(0.06 * W.shape[1], y0 + 0.03 * W.shape[0], '30"', color="w", fontsize=6)
            axw.set_title(f"wide view (multi-detector mosaic)\nboth positions + motion",
                          fontsize=6.8, pad=2.5)
            for sp in axw.spines.values():
                sp.set_color("#888888")

        m = al.get("motion") or {}
        o = al.get("orbit") or {}
        pr_t = f"P(real) {pr:.3f}   " if isinstance(pr, float) else ""
        fig.suptitle(f"rank {rank}   {al['alertId']}   [{cls}]   {pr_t}"
                     f"{_num(m, 'rate_degday'):.2f}°/d   PA {_num(m, 'pa_deg'):.0f}°   "
                     f"chi2 {_num(o, 'chi2'):.1f}   arc {_num(al, 'arcMin'):.0f} min",
                     fontsize=8, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        pstr = f"p{pr:.2f}" if isinstance(pr, float) else "pNA"
        png = os.path.join(out_dir, f"alert_{rank:05d}_{pstr}_{al['alertId']}_{cls.split('-')[0]}.png")
        fig.savefig(png, dpi=dpi); plt.close(fig)
        written += 1
        if _slice is None and written % 200 == 0:
            print(f"[pairs] {written}/{len(alerts)}", flush=True)
    if _slice is None:
        print(f"[pairs] wrote {written} per-alert images -> {out_dir}", flush=True)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--cutouts", required=True, help="npz from alert_cutouts (with wide views)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=2000, help="how many top-ranked alerts get a file")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12,
                    help="processes rendering figures in parallel (~1.3 GB resident each)")
    a = ap.parse_args(argv)
    render(a.alerts, a.cutouts, a.out_dir, top_n=a.top_n, dpi=a.dpi, workers=a.workers)


if __name__ == "__main__":
    sys.exit(main())
