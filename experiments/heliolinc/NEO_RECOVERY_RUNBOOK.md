# NEO Recovery Runbook — ADCNN + HelioLinC on real Rubin DP2 data

**Headline result (2026-05-26):** the ADCNN trail detector + HelioLinC linking pipeline
**recovered 4 real near-Earth asteroids** from real Rubin/LSSTCam DP2 difference images,
end-to-end, with every detection in each track correctly belonging to one NEO (`match_frac = 1.0`):

| track | ADCNN dets | nights | NEO (MPC designation) |
|-------|-----------|--------|------------------------|
| 1 | 7  | 3 | **2025 MX91** |
| 2 | 10 | 3 | **2025 MP99** |
| 3 | 7  | 3 | **2025 ND73** |
| 4 | 14 | 3 | **2025 MJ1**  |

Run: `run_neo_wide`. Field: RA 295–320, Dec −25..−15 (DP2 NEO ecliptic-opposition strip),
day_obs 20250620–20250720. Output files in `run_neo_wide/`: `recovered_neo.csv`, `classified.csv`,
`lr.csv`/`lr_rms.csv`, `adcnn_dets.csv`, `manifest.csv`, `neo_truth.csv`.

DP2 is **real data** (not a simulation); `ObjID`s are real MPC designations, so a "recovery" means a
linked track lands on an MPC object's predicted track within 3″.

---

## One-command reproduction

```bash
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
./neo_recovery.sh                      # full chain into run_neo_wide/
# or a fresh run dir:
RUN_NAME=run_neo_test ./neo_recovery.sh
```

All parameters live in `neo_recovery_config.sh`. The chain is 5 SLURM jobs wired by `afterok`
dependencies: **prep → detect → tracklets → link → finalize**. Resume/部分:

```bash
./neo_recovery.sh --from tracklets     # detect already done, continue
./neo_recovery.sh --only finalize      # re-run one stage
```

### Stages (neo_stages/)
| stage | partition | what it does | key script(s) |
|-------|-----------|--------------|---------------|
| `prep` | roma (lsst env) | NEO grid (r<RMAX) + targeted manifest (panels on known-NEO arcs) + known.csv + neo_truth.csv | `make_neo_grid.py`, `targeted_neo_manifest.py`, `build_known_catalog.py` |
| `detect` | ampere 4×A100 | stream diffims → ADCNN+RF → `adcnn_dets.csv` (trail endpoints) | `discover_stream.py` |
| `tracklets` | roma | ADCNN-only slim catalog → `make_tracklets` → `pairdets.csv`/`pairs.txt` | `heliolinc2/src/make_tracklets` |
| `link` | roma (4-task array) | grid-parallel HelioLinC over NEO grid → `clusters_mn/` | `link.slurm` + `node_worker_neo.sh` |
| `finalize` | roma | `link_refine` + crossmatch vs neo_truth + classify | `crossmatch.py`, `classify_tracks.py` |

---

## Why the parameters are what they are (hard-won — do not "simplify" without re-checking)

1. **Wide, multi-tract field is mandatory for NEOs.** At ≥1°/day a NEO crosses a single ~1.5° tract
   in <1 night → it appears on only one night there → HelioLinC (needs ≥2 nights) can never link it.
   A single-tract run (`run_neo_field`) recovered **0** NEOs for this reason. The strip RA 295–320 /
   Dec −25..−15 is where DP2's 88 NEO-rate known objects cluster (found via `scout_ss.py` on
   `ss_source`). The **targeted manifest** keeps only the ~1,300 panels the NEOs' ephemeris arcs
   actually cross (vs ~230k for the whole strip × 2 weeks) — 175× cheaper.

2. **`CLUSTRAD=100000` km is essential** (`neo_recovery_config.sh`). NEOs are close (topo ~0.1–0.5 AU),
   so a real object's tracklets project to state vectors that spread > 16,000 km across nights.
   `clustrad=16000` (the main-belt value) clustered **ZERO** NEOs; only same-visit FP blends clustered
   and were then killed by link_refine's duplicate-MJD check. At `100000` the clusters capture 34/86
   NEOs and 4 come out cleanly. The **NEO-only grid (r<1.6 AU)** keeps the looser radius from
   exploding the cluster count (the main belt is where FP chance-clusters blow up).

3. **`MAXVEL=5.0` deg/day** for make_tracklets — the belt default 2.0 rejects fast NEOs
   ("tracklet rejected: angvel 2.0036 not in range").

4. **Use a job ARRAY, not multi-node `srun`.** Multi-node srun jobs here die instantly with
   `RaisedSignal:53` (intermittent cluster flake, no log). The 4-task array (`link.slurm`,
   `--array=0-3`) gives the same ~4× speedup robustly. Individual array tasks *also* occasionally hit
   signal-53 at launch — just resubmit that element: `sbatch --array=<k> neo_stages/link.slurm`,
   then `./neo_recovery.sh --only finalize`.

5. **`clustrad` controls scratch volume.** Cluster files go to node-local `/lscratch`; with the NEO
   grid + clustrad they stay a few MB. Never run the link on the login node (it spikes load to ~40).

---

## Expected output / verification

`finalize` prints, and writes to `run_neo_wide/`:
```
refined tracks: ~287
CONFIRMED (known) : 4 tracks -> 4 distinct known asteroids re-discovered   # recovered_neo.csv
classify: KNOWN 4 / SPURIOUS ~283 / NEW-CANDIDATE 0                         # classified.csv
```
`recovered_neo.csv` should list 2025 MX91, MP99, ND73, MJ1 with `match_frac` 1.0.

Diagnostic helpers (already run; rerun to re-derive):
- `scout_ss.py` — where DP2's NEO-rate objects are (88 in RA 295–320 / Dec −20).
- `neo_in_field.py` / NEO crossmatch — ADCNN detected 61/86 NEOs, 49 on ≥2 nights, 40 form tracklets.

---

## Current limits and the next levers (recovery is 4 of 49 linkable)

- **Detection is NOT the bottleneck:** ADCNN detected 61/86 NEOs; 40 form multi-night tracklets.
- **Linking/FP is the bottleneck:** clusters capture 34/86 NEOs but only 4 are extracted clean; the
  rest are blended with the ~283 spurious FP tracks (loose posRMS 9k–30k km, MIXED rating).
- **0 genuinely-new candidates** — a new object must beat the FP background; not achievable yet.
- Untried levers: `CLUSTRAD` sweep 16k–100k for the precision/recall optimum; FP suppression before
  linking; cluster refinement to pull the 34 captured NEOs out of MIXED blends; trail-tracklets for
  the ~11 one-detection-per-night NEOs (`adcnn_dets.csv` already carries ra0/dec0/ra1/dec1 endpoints).

See memory: `neo-recovery-result`, `missed-object-recovery`, `neo-discovery-field`,
`adcnn-heliolinc-fp-blocker`.
```
