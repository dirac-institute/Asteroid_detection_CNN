#!/usr/bin/env python3
"""Decision table for the 2v alert op: shipped (mfsnr>=5) vs candidate (mfsnr=0 + per-field top-N),
ranked by the REAL alert priority logic (alert_stream.priority_score), from the v2 per-pair table
(measure_nomfsnr smin=0.8, uncapped -- exact FP statistics; rows carry chi2 + geometry components).

Gate (do NOT flip op_2v_alert.json unless): top-10/20 keeps most of the mfsnr=0 truth gain; alert load
operationally small; truth ranks ahead of the FP bulk; NY2 regression separately confirmed."""
import json, glob, os
import numpy as np

from ADCNN.pipelines.heliolinc.alert_stream import priority_score

RUN = "ADCNN/pipelines/heliolinc/run_lambda"


def load():
    rows = []; allrec = {}
    for cp in glob.glob(f"{RUN}/_nomfsnr_cache/*_smin0.8_v2.json"):
        k = os.path.basename(cp).split("_smin")[0]
        c = json.load(open(cp))
        for r in c["rows"]:
            rows.append((k, *r))
        for o, s in c["rec"].items():
            allrec[f"{k}_{o}"] = float(s)
    return rows, allrec


def main():
    rows, allrec = load()
    nf = len({r[0] for r in rows}) or 1
    ff_tot = sum(1 for s in allrec.values() if 2 <= s < 10)
    print(f"fields {nf}, faint-fast recoverable {ff_tot}, passing pairs {len(rows)}")
    # per-field ranked lists under mfsnr=0 (floor 0.80 enforced upstream by smin)
    byfield = {}
    for (k, mn, mf, rate, label, nfp, obj, mx, ln, chi2, dpa, dsp, perp) in rows:
        if not (1.0 <= rate <= 8.0):
            continue
        ps = priority_score("NEW", "2visit", chi2, mn, mf)
        byfield.setdefault(k, []).append((ps, label, f"{k}_{obj}" if obj else None, mf, mn))
    for k in byfield:
        byfield[k].sort(key=lambda t: -t[0])

    def ffobj(o):
        return o is not None and 2 <= allrec.get(o, -1) < 10

    def stats(name, mf_min, top_n):
        objs = set(); n_alerts = 0; tp_pairs_in = 0; tot_in = 0; ranks = []
        for k, prs in byfield.items():
            sel = [p for p in prs if p[3] >= mf_min] if mf_min > 0 else prs
            emit = sel[:top_n] if top_n else sel
            n_alerts += len(emit)
            seen = set()
            for rank, (ps, label, obj, mf, mn) in enumerate(sel):
                if label == "tp" and ffobj(obj) and obj not in seen:
                    seen.add(obj); ranks.append(rank + 1)
                    if (not top_n) or rank < top_n:
                        objs.add(obj)
            for (ps, label, obj, mf, mn) in emit:
                tot_in += 1
                if label == "tp":
                    tp_pairs_in += 1
        C = 100 * len(objs) / ff_tot
        pur = 100 * tp_pairs_in / max(tot_in, 1)
        medrank = np.median(ranks) if ranks else float("nan")
        print(f"{name:26s} ffC={C:5.2f}% ({len(objs):3d} objs)  alerts/field={n_alerts/nf:6.2f}  "
              f"topN-purity={pur:5.1f}%  med truth rank={medrank:.0f}")

    print(f"\n{'config':26s} (ranked by real priority_score; floor 0.80)")
    stats("shipped mfsnr>=5 (all)", 5.0, None)
    stats("mfsnr=0 (all pairs)",    0.0, None)
    stats("mfsnr=0 top-5/field",    0.0, 5)
    stats("mfsnr=0 top-10/field",   0.0, 10)
    stats("mfsnr=0 top-20/field",   0.0, 20)


if __name__ == "__main__":
    main()
