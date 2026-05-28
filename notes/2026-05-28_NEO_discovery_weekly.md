|   May 28, 2026 LSST Streak Detection Weekly

This week I ran the first end-to-end asteroid **discovery** pass with the full pipeline on real
data — ADCNN (the v7 segmentation network + a new focal-loss cutout-CNN false-positive filter)
feeding HelioLinC — over a wide DP2 difference-image field, and it produced **one strong new
near-Earth fast-mover candidate**. Getting there meant fixing a measurement-stage bug that was
silently throwing away every fast mover, cleaning up redundant filtering, and (importantly)
understanding what the field could and could not yield. I also closed out the HelioLinC
false-positive-budget study at an extreme density. Details below.

**The data.** Real Rubin/LSST DP2 difference images (`difference_image`, collection
`LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4`), streamed straight from the Butler datastore (no
pixels copied). The field is a ~9.6° × 2.2° strip just north of the ecliptic, covered by six
contiguous skymap tracts:

| Property | Value |
|---|---|
| Sky footprint | RA 339.6°–349.2°, Dec −4.8° to −2.6° (≈21 deg²) |
| Skymap tracts | 8971–8976 (`lsst_cells_v1`) |
| Epoch | day_obs 2025-07-17 → 2025-07-23 (7 nights, MJD ≈ 60873–60880) |
| Size | 78 visits, 180 detectors, **3,807 panels**, bands r/g/i/u (1493/1020/823/471) |

Selection was a single Butler registry query, leakage-guarded against the training set:
`queryDatasets("difference_image", where="instrument='LSSTCam' AND skymap='lsst_cells_v1' AND tract IN (8971..8976) AND visit.day_obs>=20250717 AND visit.day_obs<20250724")`, dropping any `(visit,detector)` the network trained on.

**The pipeline and its parameters.** Four stages — detect (GPU), measure (Veres trailed fit),
clean, link — chained by SLURM dependencies. The full parameter set:

| Stage | Parameter | Value |
|---|---|---|
| Detect | v7 model | `v7_diffim_scripted.pt` (reg2; UNetResSE + orientation, half-width) |
| | stage-2 FP filter | focal-loss cutout CNN `cnn_postproc.pt` (48×48×3 [diffim/σ, v7_prob, v7_agg], width-40) |
| | CNN score cut | **0.63** (the single FP cut) |
| | cheap gate `gate_pmax` | 0.10 |
| | stride / tile_batch / n_gpus | 64 / 64 / 4×A100 |
| | trail de-bias | length = (mf_length − 33.4)/0.887 |
| Measure | Veres model | `VeresModel` forward fit, L-BFGS-B, length ∈ [1,300] px |
| | ADCNN-length pre-gate | **6 px** (speed only — which trails are worth a fit) |
| | workers | 60 |
| Clean | Veres-length cut | **6 px** (≈ 1°/day — the real fast-mover cut) |
| | diaSource reliability | 0.5 (+ !isNegative; stack stream only) |
| Link | tracklets | trail mode (one trail = one tracklet), exposure 30 s |
| | hypothesis grid | `heliohypo_all.txt` (r 1.05–6.5 AU, 109,983 pts) |
| | clustrad / npt / minobsnights / mintimespan | 100,000 km / 3 / 2 / 0.05 d |
| | mjd reference | median of detections (≈ 60876.3) |
| | link_refine maxrms | 100,000 km (loose — real fast NEOs sit at posRMS 9–54k) |
| Crossmatch | tolerance | 3.0″ / 0.02 d, against the DP2 known catalog |

**Finding 1 — the measure-stage cut was killing all fast movers.** The first attempt linked
**zero** tracks. The cause was the Veres pre-gate, set at length ≥ 40 px. That demands ~7°/day, but
real >1°/day movers land at a trail length of ~10 px median (95th percentile ~29 px), so fewer than
0.1% survived 40. Lowering it to 6 px (= 1°/day, the actual target) keeps the whole fast population
(52k → 224k detections measured) and linking starts working.

**Finding 2 — one credible candidate.** With the fix the pipeline went

> 324,077 detections (CNN ≥ 0.63) → 217,442 Veres-measured → 424,956 trail-tracklets → **5 refined tracks**.

I vetted the five with two tests: orbit-fit quality, and whether each member trail's orientation
agrees with the track's night-to-night sky motion (a genuine mover's 30-second trail points along
its motion; unrelated trails that merely chance-align in position do not).

| Track | posRMS (km) | velRMS | trail-vs-motion | verdict |
|---|---|---|---|---|
| **1** | 37,459 | 0.46 | **8°** | **looks real** |
| 2 | 35,992 | 2.1 | 42° | false link |
| 3 | 57,645 | 1.9 | 73° | false link |
| 4 | 25,526 | 3.3 | 61° | false link |
| 5 | 43,830 | 4.4 | 38° | false link / marginal |

Four of the five are false links — their trails point 38–73° off the motion, the classic signature
of unrelated streaks chance-aligned on an orbit hypothesis. **Track 1** is the standout: its trails
align with the motion to 8°, it has the lowest velRMS, and it sits at a near-Earth heliocentric
distance of 1.09 AU moving ~2.3°/day. Its three epochs over two nights, for the record:

```
2025-07-21 04:46 UT   RA 23 11 26.1   Dec −04 02 27   (r)
2025-07-21 05:20 UT   RA 23 11 16.0   Dec −04 02 50   (i)
2025-07-22 05:30 UT   RA 23 02 01.0   Dec −03 44 59   (g)
```
A side note that turned into a fix: the pipeline had been stamping `mag = 21` on every detection — a
hardcoded placeholder — and the Veres fit was discarding the flux it computes. I rewired the
measurement step to actually measure photometry: the analytic optimal trail flux
(`VeresModel.computeFluxWithGradient`), a matched-filter SNR from the unit-flux model against the
diffim variance, and the calibrated AB magnitude via the `difference_image` PhotoCalib. Track 1 then
measures **mag 22.7 / 21.1 / 22.8 at SNR 5.4 / 3.7 / 5.6** (r/i/g) — moderately faint, low-SNR. (My
first instinct that it was much fainter, ~25–27, was wrong: optimal extraction over the trail
recovers the flux that a point-source SNR scaling underestimates.) The ~1.5-mag brighter i-band point
is a brightness inconsistency across epochs — a vetting flag. The HelioLinC heliocentric elements
from this 2-night arc are not a real determination: they are self-consistent (a ≈ 2.89 AU, e ≈ 0.97,
q ≈ 0.10 AU, i ≈ 175°, epoch MJD 60876.26; a ≠ r is fine, the object is near perihelion), but a is
undetermined over so short an arc — a ±5% change in the poorly-constrained radial velocity swings a
from 1.7 to 4.8 AU (and +10% → 15 AU, +20% → hyperbolic). Only the perihelion q ≈ 0.1 AU is stable.
So the orbit should not be reported beyond "a fast near-Earth object consistent with the arc"; real
elements need a third night + proper OD (Find_Orb/Gauss). It is unmatched in the DP2 known catalog. So: a faint,
low-SNR, two-night candidate, not a confirmation — needs an MPChecker submission and follow-up.

**Finding 3 — why no *known* objects came back, and it's the field, not the pipeline.** This field
turns out to be a poor place to hunt fast movers: of its ~13,900 catalogued objects the median
on-sky rate is 0.27°/day (essentially all slow main-belt), and only **4** known objects exceed
1°/day. The network detected 1 of those 4 and none had the coverage to link. So zero known fast
recoveries is roughly the ceiling here. (Earlier runs that "recovered dozens" were all-speed
recoveries through the stack-diaSource path, which catches slow point sources — not comparable to
this fast trailed path.) The genuinely NEO-rich field is the targeted one (`NEO_large`, 35 known
>1°/day plus an 86-object NEO truth set); that is where a meaningful recovery rate lives.

**Two pipeline corrections that came out of this.** First, the false-positive cut was being applied
*three times* — the CNN score at detect (0.63), then again at measure (0.5) and clean (0.3). The
latter two were redundant no-ops; I removed them so the FP cut happens exactly once, at detect.
Second, the fast-mover length cut should use the **Veres-measured** length (it already did, but was
mislabeled "de-biased" everywhere) — the noisy ADCNN length now appears only as the measure-stage
speed pre-gate, and the real cut lives once, on the accurate Veres length, at clean.

**A note on the hypothesis grid.** I ran with the broad grid (`heliohypo_all`, 1.05–6.5 AU), which
covers the near-Earth band but samples it coarsely and spends most of its 110k points on main-belt
distances. There is a dedicated NEO grid (`heliohypo_neo`, 1.05–1.58 AU) that is ~3× finer in
distance and denser in velocity across the NEO band — that is the correct grid for a fast-mover hunt
and what I will use next. Both grids floor at 1.05 AU, so neither reaches sub-AU close approachers
(Atens/Atiras near perihelion); a complete NEO survey will want the inner edge extended.

**Supporting result — the HelioLinC false-positive budget, at the extreme.** To justify running the
detector at high recall (keeping faint trails and letting linking reject the noise), I pushed the
FP-budget Monte Carlo to **2,800 false detections per panel** — roughly 30× the realistic ADCNN
rate (~88/panel) — split across three nodes. Injecting fast NEOs into that background, completeness
held at **0.98 with purity 1.00 and zero false links**: noise FP do not threaten real-NEO recovery
even at absurd density. The complementary field-wide pure-trash run, under the operational loose
gate, produced **734** chance-aligned false links at 2,800/panel (versus ~0 up to 150/panel) — so
the loose gate is fine in the realistic regime but is not a magic bullet at extreme over-density;
crossmatch-to-known is what rejects those. The realistic operating point is comfortably safe.

**Next.** Re-run the (now-fixed) pipeline on the NEO-targeted field with the NEO grid
(`MANIFEST=NEO_large`, `HELIODIST=heliohypo_neo.txt`), where the fast-mover population actually
exists; submit Track 1 to MPChecker; and look at extending the grid below 1.05 AU for close
approachers.
