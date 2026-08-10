#!/usr/bin/env python3
"""Flagship-cell comparison: MF vs reconstructed control, on ONE detection pass.

Reports BOTH tests, because the choice matters and the weaker one was used earlier:
  two-proportion z -- assumes INDEPENDENT samples. These arms share the same objects, so discarding
                      the pairing throws away information and UNDERSTATES significance. Conservative.
  McNemar          -- the correct test for paired binary outcomes: it looks only at objects whose
                      state CHANGED (recovered by one arm, not the other) and ignores agreements.
"""
import json, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
sys.path.insert(0, "outputs/runs/pa_validate")
from build_rank_table import radec_to_unit
V = "outputs/runs/pa_validate"

def alert_oids(path, T, tol_arcsec=3.0):
    tol = 2*np.sin(np.radians(tol_arcsec/3600.0)/2); trees={}
    for s in "AB":
        for v,g in T.groupby(f"visit{s}"):
            trees[(int(v),s)]=(cKDTree(radec_to_unit(g[f"ra{s}"],g[f"dec{s}"])),g["oid"].to_numpy())
    out=set()
    for l in open(path):
        a=json.loads(l); eps=a["epochs"]
        if len(eps)<2: continue
        oids=[]
        for e in eps:
            h=-1
            for s in "AB":
                tr=trees.get((int(e["visit"]),s))
                if tr is None: continue
                d,i=tr[0].query(radec_to_unit([e["ra"]],[e["dec"]]),k=1)
                if d[0]<tol: h=int(tr[1][i[0]]); break
            oids.append(h)
        if len(set(oids))==1 and oids[0]>=0: out.add(oids[0])
    return out

T=pd.read_csv(f"{V}/truth_ff.csv")
T["detA_ok"]=T.detA_ok.fillna(False); T["detB_ok"]=T.detB_ok.fillna(False)
aM=alert_oids(f"{V}/a_ff.jsonl",T); aC=alert_oids(f"{V}/a_ffctrl.jsonl",T)
T["mf"]=T.oid.isin(aM); T["ct"]=T.oid.isin(aC)
nM=sum(1 for _ in open(f"{V}/a_ff.jsonl")); nC=sum(1 for _ in open(f"{V}/a_ffctrl.jsonl"))
print(f"injected {len(T):,} (ALL inside the flagship cell: trail 14-40px, SNR 2-6)")
print(f"alert volume: control {nC:,}  MF {nM:,}   ({100*(nM/nC-1):+.0f}%)")
print(f"purity: control {100*T.ct.sum()/nC:.3f}%   MF {100*T.mf.sum()/nM:.3f}%\n")
print(f"{'population':<26}{'n':>6}{'control':>10}{'MF':>9}{'delta':>8}{'objects':>11}{'z':>7}{'McNemar p':>12}")
for lab,m in (("ALL (flagship cell)",np.ones(len(T),bool)),
              ("  SNR 2-4",(T.snr_t<4).to_numpy()),
              ("  SNR 4-6",(T.snr_t>=4).to_numpy()),
              ("  rate 2.2-3.2 (14-20px)",(T.L_target<=20).to_numpy()),
              ("  rate 4.5-6.4 (28-40px)",(T.L_target>=28).to_numpy())):
    g=T[m]; k1=int(g.ct.sum()); k2=int(g.mf.sum()); n=len(g)
    p=(k1+k2)/(2*n); se=np.sqrt(p*(1-p)*(2/n)); z=(k2/n-k1/n)/se if se>0 else 0
    b=int((g.mf&~g.ct).sum()); c=int((g.ct&~g.mf).sum())   # discordant pairs
    pm=stats.binomtest(b,b+c,0.5).pvalue if (b+c)>0 else 1.0
    print(f"{lab:<26}{n:>6}{100*k1/n:>9.2f}%{100*k2/n:>8.2f}%{100*(k2-k1)/n:>+8.2f}"
          f"{f'{k1}->{k2}':>11}{z:>7.2f}{pm:>12.2e}   (MF-only {b}, ctrl-only {c})")
print("\nMcNemar counts: 'MF-only' = recovered by MF but NOT control; 'ctrl-only' = the reverse.")
print("Only these discordant pairs carry information; objects both arms agree on are uninformative.")
