#!/usr/bin/env python3
"""Contact sheets + browsable index for a large nightly alert stream (~10k alerts).

Reads the panel-ordered cutout cache written by :mod:`ADCNN.qa.alert_cutouts` (NOT the FITS), so
rendering is pure CPU on small arrays: a 10k-alert night becomes ~100 sheet PNGs in a couple of
minutes, and a re-sort after a threshold change costs nothing but the render.

Each sheet holds `--per-sheet` alerts in rank order, one ROW per alert (its epochs side by side,
so a real mover reads as a shifting streak across the row and a static residual as an unmoving
blob). Cell captions carry the discriminating numbers (rank, rate, SNR, score, class).

Outputs under --out-dir:
  sheet_0000.png ...      contact sheets, rank-ordered (best first)
  index.html              scrollable index: every sheet inline + the ranked table
  sheets_manifest.json    {sheet: [alertIds]} for programmatic lookup

Usage:
  python -m ADCNN.qa.alert_sheets --alerts alerts.jsonl --cutouts report/cutouts.npz \
      --out-dir report/sheets [--per-sheet 50] [--limit 10000]
"""
from __future__ import annotations
import argparse, html, json, os, sys

import numpy as np

VMIN, VMAX = -2.0, 6.0          # sigma stretch: saturates at 6 sigma so sub-5sigma trails show


def _classify(al):
    """Short caption class -- same precedence as alert_report.classify (strongest veto wins)."""
    from ADCNN.qa.alert_report import classify
    return classify(al)[0].replace("-FLAGGED", "")


def render(alerts_path, cutouts_npz, out_dir, per_sheet=50, limit=None, dpi=110, cols_max=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alerts = [json.loads(l) for l in open(alerts_path)] if os.path.getsize(alerts_path) else []
    if limit:
        alerts = alerts[:limit]
    if not alerts:
        print("[sheets] 0 alerts -- nothing to render", flush=True)
        return

    z = np.load(cutouts_npz)
    stamps, a_ix, e_ix = z["stamps"], z["alert"], z["epoch"]
    by_alert = {}
    for i in range(len(a_ix)):
        by_alert.setdefault(int(a_ix[i]), {})[int(e_ix[i])] = i

    os.makedirs(out_dir, exist_ok=True)
    n_sheets = (len(alerts) + per_sheet - 1) // per_sheet
    manifest = {}
    for s in range(n_sheets):
        lo, hi = s * per_sheet, min((s + 1) * per_sheet, len(alerts))
        block = alerts[lo:hi]
        ncol = min(cols_max, max(len(al["epochs"]) for al in block))
        fig, axs = plt.subplots(len(block), ncol,
                                figsize=(2.05 * ncol, 2.25 * len(block)), squeeze=False)
        for r, al in enumerate(block):
            rank = lo + r
            m = al.get("motion") or {}
            for c in range(ncol):
                ax = axs[r][c]
                ax.set_xticks([]); ax.set_yticks([])
                if c >= len(al["epochs"]):
                    ax.axis("off"); continue
                idx = by_alert.get(rank, {}).get(c)
                if idx is None:
                    ax.text(0.5, 0.5, "no pixels", ha="center", va="center", fontsize=6)
                    continue
                st = stamps[idx].astype(np.float32)
                ax.imshow(st, origin="lower", cmap="gray",
                          vmin=VMIN, vmax=VMAX, interpolation="nearest")
                # open crosshair at the detection (stamp centre): the eye needs to know WHERE to
                # look -- at sub-5sigma the source is invisible without it. Gap keeps pixels clear.
                k = st.shape[0]; ctr = (k - 1) / 2.0; g, arm = 0.09 * k, 0.20 * k
                ax.plot([ctr - g - arm, ctr - g], [ctr, ctr], color="#ff4040", lw=0.7)
                ax.plot([ctr + g, ctr + g + arm], [ctr, ctr], color="#ff4040", lw=0.7)
                ax.plot([ctr, ctr], [ctr - g - arm, ctr - g], color="#ff4040", lw=0.7)
                ax.plot([ctr, ctr], [ctr + g, ctr + g + arm], color="#ff4040", lw=0.7)
                ax.set_xlim(-0.5, k - 0.5); ax.set_ylim(-0.5, k - 0.5)
                ep = al["epochs"][c]
                if c == 0:
                    ax.set_ylabel(f"#{rank}", fontsize=7, rotation=0, labelpad=13, va="center")
                ax.set_title(f"v{ep['visit']} snr{ep.get('snr', float('nan')):.1f} "
                             f"s{ep.get('score', float('nan')):.2f}", fontsize=5.5, pad=1.5)
            # per-row summary in the last used cell's right margin
            axs[r][min(len(al["epochs"]), ncol) - 1].text(
                1.03, 0.5, f"{_classify(al)}\n{m.get('rate_degday', float('nan')):.2f}°/d\n"
                           f"PA {m.get('pa_deg', float('nan')):.0f}°",
                transform=axs[r][min(len(al["epochs"]), ncol) - 1].transAxes,
                fontsize=5.5, va="center", ha="left")
        fig.suptitle(f"alerts {lo}–{hi - 1} of {len(alerts)}  (rank order, best first)", fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.95, 0.985])
        png = os.path.join(out_dir, f"sheet_{s:04d}.png")
        fig.savefig(png, dpi=dpi); plt.close(fig)
        manifest[os.path.basename(png)] = [a["alertId"] for a in block]
        if (s + 1) % 10 == 0 or s + 1 == n_sheets:
            print(f"[sheets] {s + 1}/{n_sheets} sheets", flush=True)

    with open(os.path.join(out_dir, "sheets_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    night = alerts[0].get("night", "?")
    rows = []
    for rank, al in enumerate(alerts[:2000]):        # table capped; full data in alert_report.csv
        m = al.get("motion") or {}
        rows.append(f"<tr><td>{rank}</td><td>{html.escape(str(al['alertId']))}</td>"
                    f"<td>{_classify(al)}</td><td>{al.get('confidenceTier', '')}</td>"
                    f"<td>{al.get('priorityScore', float('nan')):.3f}</td>"
                    f"<td>{m.get('rate_degday', float('nan')):.2f}</td>"
                    f"<td>{m.get('pa_deg', float('nan')):.0f}</td></tr>")
    imgs = "\n".join(f'<h3 id="s{i}">{n}</h3><img src="{n}" loading="lazy" style="max-width:100%">'
                     for i, n in enumerate(sorted(manifest)))
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(f"""<!doctype html><meta charset=utf-8>
<title>ADCNN alert stream — night {night}</title>
<style>body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#111;color:#eee}}
table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #444;padding:2px 6px}}
img{{border:1px solid #333;margin:.4rem 0}}h3{{font-size:13px;color:#8cf}}</style>
<h1>ADCNN alert stream — night {night}</h1>
<p><b>{len(alerts)}</b> alerts, rank-ordered (best first). Sheets: {n_sheets} ×
{per_sheet} alerts. Each row = one alert, columns = its epochs; a real mover shifts
across the row, a residual does not.</p>
<h2>Ranked table (first {min(len(alerts), 2000)})</h2>
<table><tr><th>rank<th>alertId<th>class<th>tier<th>prio<th>°/d<th>PA</tr>
{''.join(rows)}</table>
<h2>Contact sheets</h2>
{imgs}
""")
    print(f"[sheets] {n_sheets} sheets + index.html -> {out_dir}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--cutouts", required=True, help="npz from ADCNN.qa.alert_cutouts")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-sheet", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args(argv)
    render(a.alerts, a.cutouts, a.out_dir, per_sheet=a.per_sheet, limit=a.limit, dpi=a.dpi)


if __name__ == "__main__":
    sys.exit(main())
