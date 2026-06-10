# Length-split hybrid: threshold ladder on real DP2 (run_band box RA[345,348], 9 nights, MB grid)

Architecture: A = len_db<12px -> classic make_tracklets chain; B = len_db>=6px -> trail-as-tracklet
chain (--trail-as-tracklet, endpoint pseudo-obs, 36s pairing window); 6-12px overlap band feeds BOTH
(len_db noise straddles the split; union dedups). Linkage-level union; cull = art_frac<0.3 + len<=50.

| floor | A dets | B dets | monolithic baseline      | hybrid rec | purity | false | wall    |
|-------|--------|--------|--------------------------|-----------|--------|-------|---------|
| 0.80  | 28,073 | 13,106 | 629 / 86.5% / ~3 min     | 614       | 86.6%  | 95    | ~2 min  |
| 0.70  | 44,378 | 43,111 | 644 / 85.4% / 6.5 min    | **661**   | 86.2%  | 106   | ~3 min  |
| 0.60  | 81,073 | 118,619| **EXPLODED** (>9.5 min CPU, no output) | **644** | 84.8% | 115 | ~4.5 min |

FINDINGS
1. GATE PASSED at 0.80: 614/629 = 97.6% of baseline recovery, purity identical (86.6 vs 86.5), fewer
   false (95 vs 98). Residual 15 objects = len-noise tails >12px (wider overlap closes it if needed).
2. S>=0.70: hybrid EXCEEDS the monolithic chain (661 vs 644) at equal purity, half the runtime.
3. S>=0.60 REOPENED: where make_tracklets exploded, the hybrid completes in ~4.5 min at 84.8% purity --
   no purity collapse, no tractability wall. The reservoir architecture works.
4. FP-immunity of the trail branch: 119k trailed detections (mostly streak FP) at S0.60 produced ZERO
   false multi-night linkages (random streaks cannot align across 3 nights) -- the FP bulk rides the
   cheap branch and dies by geometry.
5. Honest caveats: (a) MB substrate -- chain B contributes 0 recoveries here because the box has ~no
   multi-night-observable fast movers; B's positive value (linking 1-trail-per-night FAST objects) is
   untested until a NEO-rich/injection substrate. (b) Absolute completeness gain 0.80->0.60 is modest
   here (+5%; detection recall is the ceiling on this bright-known population); the faint-fast NEO gain
   is expected larger because faint REAL movers live at low score. (c) S=0.5 rung needs a GPU
   re-detection pass (catalog floor is 0.60).

Reproduce: dets_in_hyA12_<tag>.csv / dets_in_hyB_<tag>.csv -> run_heliolinx [--trail-as-tracklet]
-> score_mn_linkage --dets-file ... --out-json. 2026-06-10.
