#!/usr/bin/env python3
"""Ranked alert report for a same-night alert stream: rank stamps + ALERT_REPORT.md + alert_report.csv.

For every alert (any nEpochs -- 2-visit pairs and 3+visit tracks) render one PNG of
18"x18" diffim cutouts (one panel per member epoch) with:

  * a red gapped crosshair on the detection,
  * a blue arrow along the fitted motion PA,
  * for ``staticVeto``-flagged members, the vetoing mag<20 static from the catalog
    (dashed orange circle at the veto radius),
  * a 2" scale bar; per-epoch SNR / score / trail length in the panel title.

Alongside the stamps it writes, next to the alerts.jsonl:

  * ``ALERT_REPORT.md`` -- the human-readable ranked table (class = CLEAN /
    STATIC-FLAGGED / TRAIN-FLAGGED / STATIONARY-FLAGGED, strongest veto wins), and
  * ``alert_report.csv`` -- the same table machine-readable (one row per alert,
    numeric columns unformatted) for automated inspection.

Pixels are read through ``pixel_vet.PanelStore`` (Butler-datastore URIs, local or S3
in-memory via diffim_io), so this runs in the torch env -- no LSST stack needed.
QA-only: reads the alert stream and dets catalog, never modifies pipeline outputs.
Zero alerts is not an error: the stub report + header-only CSV are still written.

Usage (from the repo root):
    PYTHONPATH=. python -m ADCNN.qa.alert_report \
        --alerts .../prod/alerts.jsonl \
        --dets   .../adcnn_dets_masked.csv \
        --out-dir .../prod/report \
        [--static-catalog .../static_catalog.parquet] [--static-mag-max 20] \
        [--night-label 2026-06-28] [--dpi 130]

Or opt in from the linker: ``link_2visit ... --report`` renders overlays + this report
automatically after the alert stream is written.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ADCNN.linking.pixel_vet import PanelStore
from ADCNN.qa.trail_overlays import _night_label, _unit

PXSCALE = 0.2   # arcsec/px
HALF = 45       # cutout half-size px; 0.2"/px -> 9" half-width

CSV_COLS = ["rank", "alertId", "class", "tier", "status", "nEpochs", "priorityScore",
            "rate_degday", "rate_sigma_degday", "pa_deg", "arcMin", "snr_min", "snr_max",
            "score_min", "score_max", "visits", "ra", "dec", "mjd_first", "mjd_last",
            "perAlertShare", "nStaticMembers", "vetoTrain", "trainAligned", "trainRepeats",
            "vetoStationary", "match_obj"]


def classify(alert):
    """(class, filename suffix) -- strongest veto wins; CLEAN means no veto flagged."""
    if (alert.get("staticVeto") or {}).get("nStaticMembers", 0):
        return "STATIC-FLAGGED", "static"
    if (alert.get("trainVeto") or {}).get("vetoTrain", False):
        return "TRAIN-FLAGGED", "train"
    if (alert.get("stationarity") or {}).get("vetoStationary", False):
        return "STATIONARY-FLAGGED", "stationary"
    return "CLEAN", "clean"


def draw_epoch(ax, ep, alert, store, panel_of, statics):
    import matplotlib.pyplot as plt
    v, det = int(ep["visit"]), int(ep["detector"])
    img, _, _, wcs, _ = store.get(panel_of[(v, det)])
    x, y = wcs.world_to_pixel_values(ep["ra"], ep["dec"])
    x, y = float(x), float(y)
    x0, x1 = int(round(x)) - HALF, int(round(x)) + HALF + 1
    y0, y1 = int(round(y)) - HALF, int(round(y)) + HALF + 1
    px0, py0 = max(x0, 0), max(y0, 0)
    cut = img[py0:min(y1, img.shape[0]), px0:min(x1, img.shape[1])]
    lo, hi = np.nanpercentile(cut, [1, 99.7])
    ax.imshow(cut, origin="lower", cmap="gray_r", vmin=lo, vmax=hi,
              extent=[px0 - x, px0 - x + cut.shape[1], py0 - y, py0 - y + cut.shape[0]])
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:   # crosshair with a gap
        ax.plot([dx * 6, dx * 14], [dy * 6, dy * 14], color="tab:red", lw=1.2)
    # motion direction arrow (sky PA east-of-north -> pixel via WCS)
    pa = np.radians(alert["motion"]["pa_deg"]); L = 5.0 / 3600.0   # 5"
    ra2 = ep["ra"] + L * np.sin(pa) / np.cos(np.radians(ep["dec"])); dec2 = ep["dec"] + L * np.cos(pa)
    xa, ya = wcs.world_to_pixel_values(ra2, dec2)
    ax.annotate("", xy=(float(xa) - x, float(ya) - y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.4, alpha=0.9))
    # vetoing static marker if this epoch is flagged
    mem = next((m for m in (alert.get("staticVeto") or {}).get("members", [])
                if m["visit"] == v), None)
    tag = ""
    if mem and mem.get("isStatic") and statics is not None:
        stree, sra, sdec = statics
        _, i = stree.query(_unit(np.array([ep["ra"]]), np.array([ep["dec"]]))[0], k=1)
        xs, ys = wcs.world_to_pixel_values(sra[i], sdec[i])
        ax.add_patch(plt.Circle((float(xs) - x, float(ys) - y), 3.0 / PXSCALE, fill=False,
                                color="orange", lw=1.6, ls="--"))
        ax.plot(float(xs) - x, float(ys) - y, "o", color="orange", ms=5)
        tag = f"  STATIC {mem['sepArcsec']:.2f}\" mag {mem['staticMag']:.1f}"
    ax.set_title(f"v{v} det{det}  snr {ep['snr']:.1f}  score {ep['score']:.2f}\n"
                 f"trail {ep['trail_len_px']:.0f}px{tag}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot([-HALF + 4, -HALF + 4 + 2 / PXSCALE], [-HALF + 5, -HALF + 5], color="k", lw=2)
    ax.text(-HALF + 4 + 1 / PXSCALE, -HALF + 8, '2"', ha="center", fontsize=7)
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alerts", required=True, help="alerts.jsonl (schema >= 1.2)")
    ap.add_argument("--dets", required=True, help="adcnn dets CSV (masked; must carry fits_path)")
    ap.add_argument("--out-dir", required=True, help="directory for the rankNN stamp PNGs")
    ap.add_argument("--static-catalog", default=None,
                    help="static catalog (parquet/csv) to mark the vetoing static on flagged epochs")
    ap.add_argument("--static-mag-max", type=float, default=20.0)
    ap.add_argument("--night-label", default=None,
                    help="date string for the report title (default: derived from the alert night MJD)")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alerts = [json.loads(l) for l in open(a.alerts)] if os.path.getsize(a.alerts) else []
    prod_dir = os.path.dirname(os.path.abspath(a.alerts))
    md_path = os.path.join(prod_dir, "ALERT_REPORT.md")
    csv_path = os.path.join(prod_dir, "alert_report.csv")
    label = a.night_label or (_night_label(alerts[0]["night"]) if alerts else "?")

    if not alerts:
        pd.DataFrame(columns=CSV_COLS).to_csv(csv_path, index=False)
        with open(md_path, "w") as f:
            f.write(f"# Alert report — night {label} ({os.path.basename(prod_dir)})\n\n"
                    "**0 alerts** in this alert stream.\n")
        print(f"[alert-report] 0 alerts -> stub {md_path}", flush=True)
        return

    os.makedirs(a.out_dir, exist_ok=True)
    # panel path per (visit, detector) from the dets catalog -- no manifest/Butler needed
    pmap = pd.read_csv(a.dets, usecols=["visit", "detector", "fits_path"]).drop_duplicates(
        ["visit", "detector"])
    panel_of = {(int(v), int(d)): p for v, d, p in
                zip(pmap.visit, pmap.detector, pmap.fits_path)}

    statics = None
    if a.static_catalog:
        from scipy.spatial import cKDTree
        sc = (pd.read_parquet(a.static_catalog) if a.static_catalog.endswith(".parquet")
              else pd.read_csv(a.static_catalog))
        magcol = "mag" if "mag" in sc.columns else "mag_best"
        sc = sc[np.isfinite(sc[magcol]) & (sc[magcol] < a.static_mag_max)]
        if len(sc):
            statics = (cKDTree(_unit(sc.ra.to_numpy(), sc.dec.to_numpy())),
                       sc.ra.to_numpy(), sc.dec.to_numpy())

    store = PanelStore(max_panels=4)
    rows = []
    for rank, al in enumerate(alerts):
        cls, suffix = classify(al)
        eps = al["epochs"]
        fig, axs = plt.subplots(1, len(eps), figsize=(3.7 * len(eps), 4.1))
        for ax, ep in zip(np.atleast_1d(axs), eps):
            draw_epoch(ax, ep, al, store, panel_of, statics)
        m = al["motion"]
        share = (al.get("fpp") or {}).get("perAlertShare")
        sh_t = f"  fpp {share:.1e}" if isinstance(share, (int, float)) else ""
        fig.suptitle(f"rank {rank}  {al['alertId']}  [{cls}]  tier {al['confidenceTier']}  "
                     f"rate {m['rate_degday']:.2f}±{3*m['rate_sigma_degday']:.2f} °/d  "
                     f"PA {m['pa_deg']:.0f}°{sh_t}", fontsize=10, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        png = os.path.join(a.out_dir, f"rank{rank:02d}_{al['alertId']}_{suffix}.png")
        fig.savefig(png, dpi=a.dpi); plt.close(fig)
        tv = al.get("trainVeto") or {}
        snrs = [e["snr"] for e in eps]
        rows.append({
            "rank": rank, "alertId": al["alertId"], "class": cls,
            "tier": al["confidenceTier"], "status": al["status"], "nEpochs": len(eps),
            "priorityScore": al["priorityScore"], "rate_degday": m["rate_degday"],
            "rate_sigma_degday": m["rate_sigma_degday"], "pa_deg": m["pa_deg"],
            "arcMin": al.get("arcMin"), "snr_min": min(snrs), "snr_max": max(snrs),
            "score_min": al["vetting"]["score_min"], "score_max": al["vetting"]["score_max"],
            "visits": ";".join(str(e["visit"]) for e in eps),
            "ra": eps[0]["ra"], "dec": eps[0]["dec"],
            "mjd_first": eps[0]["mjd"], "mjd_last": eps[-1]["mjd"],
            "perAlertShare": share,
            "nStaticMembers": (al.get("staticVeto") or {}).get("nStaticMembers", 0),
            "vetoTrain": bool(tv.get("vetoTrain", False)),
            "trainAligned": tv.get("nAligned"), "trainRepeats": tv.get("nRepeats"),
            "vetoStationary": bool((al.get("stationarity") or {}).get("vetoStationary", False)),
            "match_obj": (al.get("match") or {}).get("obj"),
        })
        print(f"[alert-report] wrote {png}", flush=True)

    df = pd.DataFrame(rows, columns=CSV_COLS)
    df.to_csv(csv_path, index=False)
    counts = df["class"].value_counts()
    with open(md_path, "w") as f:
        f.write(f"# Alert report — night {label} ({os.path.basename(prod_dir)})\n\n"
                f"**{len(df)} alerts** = {counts.get('CLEAN', 0)} clean + "
                f"{counts.get('TRAIN-FLAGGED', 0)} train-flagged + "
                f"{counts.get('STATIC-FLAGGED', 0)} static-flagged + "
                f"{counts.get('STATIONARY-FLAGGED', 0)} stationary-flagged (FLAG-not-drop).\n\n")
        f.write("| rank | alertId | class | tier | prio | rate °/d | snr | score | visits | "
                "perAlertShare | train | stationary | known |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            sh = f"{r['perAlertShare']:.1e}" if isinstance(r["perAlertShare"], (int, float)) else "—"
            tr = (f"{r['trainAligned']} aligned/{r['trainRepeats']} rep" if r["vetoTrain"] else "—")
            f.write(f"| {r['rank']} | {r['alertId']} | {r['class']} | {r['tier']} | "
                    f"{r['priorityScore']:.2f} | {r['rate_degday']:.2f} | "
                    f"{r['snr_min']:.1f}/{r['snr_max']:.1f} | "
                    f"{r['score_min']:.2f}/{r['score_max']:.2f} | "
                    f"{'/'.join(v[-3:] for v in r['visits'].split(';'))} | {sh} | {tr} | "
                    f"{'yes' if r['vetoStationary'] else '—'} | {r['match_obj'] or '—'} |\n")
        f.write("\nImages: `report/rankNN_<alertId>_<class>.png` — 18\"x18\" diffim cutouts per "
                "epoch, red crosshair = detection, blue arrow = motion PA, dashed orange = vetoing "
                "mag<20 static (3\" radius). `report/overlay_rankNN_<alertId>.png` — full trail "
                "overlays (ADCNN.qa.trail_overlays) with counterpart-epoch forced capsules. "
                "Machine-readable table: `alert_report.csv`.\n")
    print(f"[alert-report] {len(df)} alert(s) -> {md_path} + {csv_path}", flush=True)


if __name__ == "__main__":
    main()
