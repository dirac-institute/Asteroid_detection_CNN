# Same-night NEO discovery pipeline (ADCNN → trail-state linking)

Detect fast-moving near-Earth objects in a single night of Rubin/LSST difference images, by the next
morning, from **two same-night visits** (the WFD pair the survey delivers ~56% of the time) — with a
3-σ-clean 3-visit confirmed tier and a purified 2-visit candidate/alert tier.

## Run it end-to-end (one command)

```bash
# morning run: build the manifest for a night + tract(s), detect, link
DAY=20250706 TRACTS=8489 RUN=/sdf/.../out  sbatch --export=ALL,DAY,TRACTS,RUN sn_run.slurm
# or, with a manifest already in $RUN/manifest.csv:
RUN=/sdf/.../out  sbatch --export=ALL,RUN sn_run.slurm
```

`sn_run.slurm` (CPU orchestrator) runs these stages; resumable/idempotent (skips a stage whose output exists):

| # | stage | script | env | in → out |
|---|---|---|---|---|
| 0 | manifest | `build_manifest.py` | LSST stack | tract(s)+day → `manifest.csv` (visit,detector,band,fits_path) |
| 1 | detect | `sn_detect.slurm` → `discover_stream.py` | asteroid_cnn (GPU) | diffims → `adcnn_dets.csv` (sky pos + trail endpoints + score) |
| 2 | known | `build_known_catalog.py` | LSST stack | manifest → `known.csv` (ObjID,mjd,ra,dec for crossmatch) |
| 3 | mask | `mask_flags.py` | asteroid_cnn | dets → `adcnn_dets_masked.csv` (+`art_frac`, TP-safe artifact cut) |
| 4 | link | `trail_state_link.py` | asteroid_cnn | masked dets → `tracks.csv` (CONFIRMED + NEW, tagged by tier) |

Stage 1 runs on the preemptible `ampere` GPU partition; `discover_stream` is resumable (per-shard `.done`),
and `sn_run` resubmits it through preemptions. Envs: Butler/Sorcha stages use the cvmfs LSST stack
(`loadLSST.sh; setup lsst_distrib`); detection/linking use the `asteroid_cnn` conda env.

## The linker (`trail_state_link.py`) — two tiers, one pass

The trail measures on-sky velocity **directly** (endpoints over the exposure), so there is NO heliocentric
hypothesis grid and NO candidate explosion — O(N log N). `physical_check` rejects false links.

- **3+visit tier** — trail-velocity clustering (`--pos-tol-3v 0.05`) + linear-fit + bound-orbit check.
  At `--score-min 0.80` this is the **3-σ discovery** op-point (λ_FP ≲ 1.35×10⁻³/field-night; recovers 2025 NY2).
- **2-visit tier** — the survey's native same-night product. Shipped op-point:
  - **chord seeding** (`--seed-2v chord`): pair detections by the *precise position chord* (k-d tree over the
    plausible rate band) and verify with the trail — vs the old trail-velocity clustering that scattered ~80%
    of real pairs. **~4× recall, ~10× lower FP** (real DP2 off-ecliptic FP).
  - **combined orbit-fit χ² gate** (`--chi2-2v-max 3.0`): one weighted Mahalanobis goodness-of-fit
    (`pair_chi2`: collinearity, bound-orbit rate-residual, brightness, trail-vs-motion PA & speed, each / its
    real-pair scatter `CHI2_SIG_2V`) instead of independent AND-thresholds. **+2.5× completeness at the same
    false rate** (λ≈0.0023/pair; 0 false / 439 real pairs). No ML, no training.
  - optional **recurrence veto** (`--recur-max 2`): TP-safe stationarity cut (residuals recur at a fixed sky
    position across visits; a ≥1°/day mover never does). Needs many visits/night (dense fields); ~no-op for a
    WFD pair. Operationally use a persistent-residual catalog from survey history.
  - optional **Δt window** (`--max-arc-2v-min 40`) and per-member score floor (`--score-2v-min`).

**`tracks.csv`** columns: `night, ndet, nvisit, n_epochs, tier(2visit|3+visit), arc_hr, rms_arcsec,
speed_degday, chi2, a_au, ecc, ra, dec, check, match_obj, match_frac, status(CONFIRMED|NEW)`. The 2-visit
**NEW** rows are a follow-up candidate stream — rank by ascending `chi2`.

**Interpretation:** a single night's 2 detections cannot self-confirm a NEO (no survey confirms from 2 same-
night points) — 3+visit tracks are 3-σ discoveries; 2-visit are candidates that a 3rd epoch (same-night
triplet, 17%; or next night → tracklet→track) confirms. See `SAME_NIGHT_2v_3sigma.md` for the full
completeness/purity analysis and the cadence/detection ceilings.

## Analysis & calibration tools (not in the runtime path)

| script | purpose |
|---|---|
| `count_realfp.py` | **direct** 2-visit FP rate on real off-ecliptic fields (no injection, no MC). Canonical purity tool: `--seed chord --chi2-max 3.0`. |
| `calibrate_link_fpp.py` | null-MC FPP→score calibration (the 3-visit 3-σ op-point). NOTE: uses AND-cuts; for the 2-visit chi² tier use `count_realfp` (the null-MC overestimates the 2v rate — see SAME_NIGHT doc). |
| `recovery_metrics.py` | completeness/purity 2-vs-3 on the injected test2 set. |
| `sim_orbits.py`, `sample_granvik.py`, `build_pointing_db.py` | Sorcha/Granvik injection: realistic NEO orbits × cadence → injected test set. |
| `build_realfp_manifests.py` | pick dense off-ecliptic (clean-FP) field-nights → manifests for `count_realfp`. |
| `scan_cadence.py` | rank processable dense (tract,night) fields. |
| `recurrence.py` | `add_recurrence` (stationarity flag), used by the linker veto and `count_realfp`. |
| `orbit_check.py` | Herget/Lambert bound-orbit fit (`orbit_ok`), used by `physical_check`/`pair_chi2`. |
| `inject_trails.py` | `add_trails` — synthetic trails into diffims (used by `discover_stream --inject`). |
| `validate_candidate.py` | re-link + null-test a candidate night (post-discovery vetting). |

`archive/` — one-off exploratory scripts kept for provenance (field_fastmovers*, butler_diasource_catalog,
trail_quality, recall_at_threshold); not part of the pipeline or active tooling.
