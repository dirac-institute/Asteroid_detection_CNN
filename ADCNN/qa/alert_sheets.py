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

VMIN, VMAX = -2.0, 6.0    # sigma stretch: saturates at 6 sigma so sub-5sigma trails show


def _num(d, key, default=float("nan")):
    """dict.get(k, default) returns None when the key EXISTS and is null -- which some alert
    fields are -- and None then blows up an f-string format spec. Coerce to a float always."""
    v = d.get(key, default) if isinstance(d, dict) else default
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


EP_PER_BLOCK = 2                # epochs drawn per alert block (the 2-visit product; 3v shows first 2)


def _classify(al):
    """Short caption class -- same precedence as alert_report.classify (strongest veto wins)."""
    from ADCNN.qa.alert_report import classify
    return classify(al)[0].replace("-FLAGGED", "")


def _assert_cache_matches(alerts_path, cutouts_npz, n_alerts):
    """Refuse to render from a cutout cache built for a DIFFERENT alerts.jsonl.

    The cache is keyed by alert INDEX (position in alerts.jsonl), so if the night is re-linked or
    re-ranked and the cache is not rebuilt, index i addresses a different physical alert than it
    did -- every image would be captioned with another object's numbers, silently. Night 20260629
    sat in exactly that state: a 6,821-alert cache against an 11,293-alert file.
    """
    meta = os.path.splitext(cutouts_npz)[0] + "_meta.json"
    if not os.path.exists(meta):
        # Warning + return meant a cache with NO sidecar skipped the identity check entirely -- a
        # deliberately mis-indexed cache rendered clean behind one WARN line. The sidecar is the only
        # thing that makes the cache verifiable, so its absence is a refusal, not a note.
        raise SystemExit(
            f"cutout cache {cutouts_npz} has no {os.path.basename(meta)} sidecar, so it CANNOT be "
            f"verified against {os.path.basename(alerts_path)}. The cache is indexed by alert "
            f"position; rendering an unverifiable cache is how every image gets another object's "
            f"caption. Rebuild it with ADCNN.qa.alert_cutouts (which writes the sidecar).")
    m = json.load(open(meta))
    # COUNT is compared against the UNTRUNCATED file, not the post---limit list: run_night passes the
    # same --limit to the cut and the render, so the two counts matched by construction and the guard
    # could never fire; and a legitimate "render the top 11 of 12" aborted, blaming a stale cache.
    try:
        n_file = sum(1 for _ in open(alerts_path))
    except OSError:
        n_file = n_alerts
    # A cache built for the top 20,000 of a 26,253-alert file is the SHIPPED shape, and asking it for
    # the top 100 is legitimate -- an equality test refused that, so nobody could re-render a night's
    # top-N from its own cache. What is actually required is that the cache covers the request and was
    # built from a file no larger than this one.
    m_n = m.get("n_alerts")
    if not (isinstance(m_n, int) and n_alerts <= m_n <= n_file):
        raise SystemExit(
            f"cutout cache is STALE: {cutouts_npz} was built for {m.get('n_alerts')} alerts but "
            f"{alerts_path} has {n_file} and {n_alerts} were requested. The cache is indexed by "
            f"alert position, so rendering would caption every image with the wrong alert. Rebuild:\n"
            f"  python -m ADCNN.qa.alert_cutouts --alerts {alerts_path} --dets <masked dets> "
            f"--out {cutouts_npz}")
    # IDENTITY. A count check cannot see a PERMUTATION, and run_night rewrites alerts.jsonl in place
    # (rerank_alerts) between the cut and the render -- counterfactually only 9 of 20,000 index
    # positions would then have addressed the same alert. The cache already stores visit/detector per
    # zoom row and no renderer ever checked them.
    try:
        import numpy as _np
        z = _np.load(cutouts_npz, allow_pickle=True)
        if {"alert", "visit", "detector"} <= set(z.files):
            ai = _np.asarray(z["alert"]); vv = _np.asarray(z["visit"]); dd = _np.asarray(z["detector"])
            alerts = [json.loads(l) for l in open(alerts_path)]
            # EVERY row, not a 400-row sample. The cache is ordered by rank, so a sample of the first
            # 400 zoom rows covers ~200 alerts of 20,000 -- permuting only ranks >=500 mis-addressed
            # 536 of 1,037 alerts and the guard PASSED, writing 22 sheets in which every caption past
            # rank 500 was wrong. The comparison is a dict lookup per row; the full pass is cheap.
            eps_of = [{(int(e.get("visit", -1)), int(e.get("detector", -1)))
                       for e in (al.get("epochs") or [])} for al in alerts]
            bad = 0
            for k in range(len(ai)):
                j = int(ai[k])
                if j >= len(alerts) or (int(vv[k]), int(dd[k])) not in eps_of[j]:
                    bad += 1
            if bad:
                raise SystemExit(
                    f"cutout cache does NOT MATCH {alerts_path}: {bad} of {len(ai)} "
                    f"cached stamps name a (visit,detector) absent from the alert "
                    f"they are indexed to. The counts agree, so this is a REORDERING -- every image "
                    f"would be captioned with the wrong alert. Rebuild the cache.")
    except SystemExit:
        raise
    except Exception as _e:
        print(f"[cache-check] identity check skipped ({type(_e).__name__}: {_e})", flush=True)


def render(alerts_path, cutouts_npz, out_dir, per_sheet=48, limit=None, dpi=110, grid_cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alerts = [json.loads(l) for l in open(alerts_path)] if os.path.getsize(alerts_path) else []
    if limit:
        alerts = alerts[:limit]
    if not alerts:
        print("[sheets] 0 alerts -- nothing to render", flush=True)
        return

    _assert_cache_matches(alerts_path, cutouts_npz, len(alerts))
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
        # GRID of per-alert blocks (each block = that alert's 2 epochs side by side). One alert per
        # ROW would make a 50-alert sheet ~12000 px tall -- unscannable; a grid keeps every sheet a
        # single viewable page.
        nbx = grid_cols
        nby = int(np.ceil(len(block) / nbx))
        fig, axs = plt.subplots(nby, nbx * EP_PER_BLOCK,
                                figsize=(1.32 * nbx * EP_PER_BLOCK, 1.55 * nby), squeeze=False)
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        for i, al in enumerate(block):
            rank = lo + i
            by, bx = divmod(i, nbx)
            m = al.get("motion") or {}
            for c in range(EP_PER_BLOCK):
                ax = axs[by][bx * EP_PER_BLOCK + c]
                if c >= len(al["epochs"]):
                    continue
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#666"); sp.set_linewidth(0.4)
                idx = by_alert.get(rank, {}).get(c)
                if idx is None:
                    ax.text(0.5, 0.5, "no pixels", ha="center", va="center", fontsize=5)
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
                ax.set_title(f"snr{_num(ep, 'snr'):.1f} "
                             f"s{_num(ep, 'score'):.2f}", fontsize=4.6, pad=1.0)
            # one caption per block, under its left stamp: rank, class, rate. Everything the eye
            # needs to triage without cross-referencing the table.
            cl = _classify(al)
            axs[by][bx * EP_PER_BLOCK].set_xlabel(
                f"#{rank} {cl if cl != 'CLEAN' else ''} {_num(m, 'rate_degday'):.1f}°/d",
                fontsize=5.0, labelpad=1.2,
                color="#202020" if cl == "CLEAN" else "#c05000")
        fig.suptitle(f"alerts {lo}–{hi - 1} of {len(alerts)}   (rank order, best first; "
                     f"each pair = one alert's two epochs)", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.55, w_pad=0.12)
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
                    f"<td>{_num(al, 'priorityScore'):.3f}</td>"
                    f"<td>{_num(m, 'rate_degday'):.2f}</td>"
                    f"<td>{_num(m, 'pa_deg'):.0f}</td></tr>")
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
    ap.add_argument("--per-sheet", type=int, default=48, help="alerts per contact sheet")
    ap.add_argument("--grid-cols", type=int, default=6, help="alert BLOCKS across a sheet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args(argv)
    render(a.alerts, a.cutouts, a.out_dir, per_sheet=a.per_sheet, limit=a.limit, dpi=a.dpi,
           grid_cols=a.grid_cols)


if __name__ == "__main__":
    sys.exit(main())
