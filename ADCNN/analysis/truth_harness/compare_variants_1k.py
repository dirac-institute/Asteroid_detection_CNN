#!/usr/bin/env python3
"""Compare relink variants at the BINDING 1k budget -- the only framing that decides anything here.

WHY NOT ALERT-STAGE COUNTS. The product ships ~1,000 alerts out of ~34,000 linked, so an alert-stage
metric counts ~99.9% of things the op discards. Two verdicts have already flipped on this: the matched
filter went p=0.31 ("null") -> p=0.0028 in the flagship cell, and `gate=any` went "REFUTED, loses 131
fast movers" -> +10 movers / 0 lost, because those 131 never survived the op. At a FIXED budget a cut
that removes a population you do not want is PROTECTIVE: it prevents displacement. So the failure mode
to watch is ALL rising while FLAGSHIP falls.

PAIRED TEST. The arms link the SAME injected objects, so McNemar is the correct test; a two-proportion
z discards the pairing and understates significance (measured: z=4.43 unpaired vs p=1.15e-15 paired).

MEASURED 2026-08-11 on inj_dets_v3.csv (5,315 injected movers, post-audit detection). Ceiling:
2,053 of 5,315 (38.63%) detected in BOTH epochs -- no linker can exceed that.

    cell        base     pregate   paired McNemar
    ALL        11.12%     11.06%   gained 4, lost 7, p=0.55
    FLAGSHIP    2.02%      1.95%   gained 0, lost 1, p=1.0

pregate (ADCNN_PRE_DPA_TT=25, ADCNN_PRE_DPA_TM=30) is REFUTED: it admitted 1,570 MORE alerts at the
link stage (49,690 -> 51,260) and delivered 147 FEWER through the op, for no gain in either cell.
NOTE the flagship arm carries only ~28 delivered movers, so this excludes LARGE effects only -- it
needs ~6 discordant pairs to reach p<0.05 and got 1. "No evidence of benefit", not "equivalent".

THE OP IS PROTECTIVE, and that is the headline (same arm, one variable):

    top-1000 of the RAW linked stream   ALL 10.07%   FLAGSHIP 0.43%
    top-1000 after the shipped 1k op    ALL 11.12%   FLAGSHIP 2.02%

4.7x in the priority cell. At a fixed budget the op does not trade purity against completeness -- it
decides WHICH 1000 ship, and its gates stop faint-fast movers being displaced by alerts that rank
higher on priorityScore. Removing the bright-star proximity veto alone leaves 8,523 survivors instead
of 5,028 and changes FLAGSHIP not at all (2.02% both), so that veto's purity benefit is free here.

These absolutes are NOT comparable to the pre-audit 13.04%/4.18%. Two explanations for the gap were
tested and both REFUTED -- it is not the bright-star veto (removing it made ALL slightly worse) and
not the op (removing it made both cells much worse). Detection changed; the remaining candidates are
the link configuration and detection itself.

Usage: python -m ADCNN.analysis.truth_harness.compare_variants_1k base pregate
"""
import json
import subprocess
import sys
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import binomtest

V = "outputs/runs/pa_validate"
OP = "ADCNN/pipelines/heliolinc/op_2v_stream_1k.json"
REFCAT = "outputs/runs/10k_cadence/run_night_20260706/bright_refcat.parquet"
DETS = f"{V}/inj_dets_v3.csv"
TRUTH = f"{V}/truth_v3.csv"
BUDGET = 1000
FF = lambda T: (T.rate > 2.0) & (T.rate <= 8.0) & (T.snr_t < 6.0)      # the flagship cell


def radec_to_unit(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def alert_oids(alerts, T, tol_arcsec=3.0):
    """oids whose BOTH epochs land on the SAME injected object.

    LABEL HYGIENE: requiring one shared oid across both epochs is what keeps a chance pairing of two
    different objects from being scored as a recovery.
    """
    tol = 2 * np.sin(np.radians(tol_arcsec / 3600.0) / 2)
    trees = {}
    for s in "AB":
        for v, g in T.groupby(f"visit{s}"):
            trees[(int(v), s)] = (cKDTree(radec_to_unit(g[f"ra{s}"], g[f"dec{s}"])), g["oid"].to_numpy())
    out = set()
    for a in alerts:
        eps = a.get("epochs") or []
        if len(eps) < 2:
            continue
        oids = []
        for e in eps:
            hit = -1
            for s in "AB":
                tr = trees.get((int(e["visit"]), s))
                if tr is None:
                    continue
                d, i = tr[0].query(radec_to_unit([e["ra"]], [e["dec"]]), k=1)
                if d[0] < tol:
                    hit = int(tr[1][i[0]]); break
            oids.append(hit)
        if len(set(oids)) == 1 and oids[0] >= 0:
            out.add(oids[0])
    return out


def delivered(tag):
    """Apply the FULL 1k op (incl. the bright-star veto), then take the top-BUDGET by priorityScore."""
    src, surv = f"{V}/a_v3_{tag}.jsonl", f"{V}/surv_v3_{tag}.jsonl"
    r = subprocess.run([sys.executable, "-m", "ADCNN.qa.filter_op", "--alerts", src, "--dets", DETS,
                        "--op", OP, "--out", surv, "--refcat", REFCAT, "--allow-unranked"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    print("   " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
    if r.returncode != 0:
        raise SystemExit(f"filter_op failed for {tag}:\n{r.stdout}\n{r.stderr}")
    al = [json.loads(l) for l in open(surv)]
    # Rank by priorityScore, the delivered ordering. Ties broken by chi2 so the cut is deterministic
    # rather than dependent on the order the linker happened to emit.
    al.sort(key=lambda a: (-(a.get("priorityScore") or 0.0),
                           float(((a.get("orbit") or {}).get("chi2")) or 1e9)))
    return al[:BUDGET], len(al)


def mcnemar(a_set, b_set, pop):
    """Exact paired test. b=gained by arm B, c=lost by arm B."""
    b = sum(1 for o in pop if o in b_set and o not in a_set)
    c = sum(1 for o in pop if o in a_set and o not in b_set)
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, p


def main(tags):
    T = pd.read_csv(TRUTH)
    T["detA_ok"] = T.detA_ok.fillna(False).astype(bool)
    T["detB_ok"] = T.detB_ok.fillna(False).astype(bool)
    both = T[T.detA_ok & T.detB_ok]
    print(f"truth: {len(T):,} injected, {len(both):,} detected in BOTH epochs "
          f"({100*len(both)/len(T):.2f}%) -- the ceiling any linker can reach\n")

    res = {}
    for t in tags:
        print(f"[{t}] applying the full 1k op ...")
        al, n_surv = delivered(t)
        res[t] = dict(oids=alert_oids(al, T), n_deliv=len(al), n_surv=n_surv)
        print(f"   survivors {n_surv:,} -> delivered {len(al):,}\n")

    cells = [("ALL", np.ones(len(T), bool)), ("FLAGSHIP (rate 2-8, SNR<6)", FF(T).to_numpy())]
    print(f"{'cell':<28}" + "".join(f"{t:>14}" for t in tags))
    for name, mask in cells:
        g = T[mask]
        print(f"{name:<28}", end="")
        for t in tags:
            print(f"{100*g.oid.isin(res[t]['oids']).mean():>13.2f}%", end="")
        print(f"   (n={len(g):,})")

    if len(tags) == 2:
        a, b = tags
        print(f"\nPAIRED McNemar, {b} vs {a}:")
        for name, mask in cells:
            pop = T[mask].oid.to_numpy()
            nb, nc, p = mcnemar(res[a]["oids"], res[b]["oids"], pop)
            verdict = ("no significant difference" if p > 0.05 else
                       f"{b} WINS" if nb > nc else f"{b} REGRESSES")
            print(f"  {name:<28} gained {nb:>4}  lost {nc:>4}  p={p:.4g}   {verdict}")
        print("\nA rise in ALL with a fall in FLAGSHIP is the failure mode: at a fixed budget the")
        print("extra alerts DISPLACE faint-fast movers. Judge on FLAGSHIP first.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "pregate"])
