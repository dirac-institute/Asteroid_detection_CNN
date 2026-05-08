"""Build a manifest of (visit, detector) pairs suitable for stage-1 diffim
generation.

A pair is "suitable" iff, in the configured collection chain, it has ALL of:
  - preliminary_visit_image          (PVI itself)
  - single_visit_star_footprints     (AlardLupton kernel-candidate sources)
  - >= 1 overlapping template_coadd patch in the SAME band

Pairs are sampled to be DETECTOR-DIVERSE: we shuffle the PVI candidate pool
and require at most --max-per-detector pairs from any single detector. The
on-stack `bind={'band': ...}` filter is also unreliable for some queries on
this stack, so the actual band is read from each PVI's ref dataId and used
for filtering after the fact.

Usage:
    python build_manifest.py --band g --n-pairs 20 --max-per-detector 2 \
        --out manifests/pilot_g_20.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from lsst.daf.butler import Butler

REPO_DEFAULT = "dp2_prep"
STAGE3_DEFAULT = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2_DEFAULT = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP_DEFAULT = "lsst_cells_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--stage3", default=STAGE3_DEFAULT)
    p.add_argument("--stage2", default=STAGE2_DEFAULT)
    p.add_argument("--skymap", default=SKYMAP_DEFAULT)
    p.add_argument("--band", default=None,
                   help="Optional band filter; if omitted, accept any band.")
    p.add_argument("--n-pairs", type=int, required=True)
    p.add_argument("--max-per-detector", type=int, default=2,
                   help="Cap on accepted pairs per detector.")
    p.add_argument("--candidate-pool", type=int, default=2000,
                   help="Number of PVI refs to draw into the candidate pool"
                   " before shuffling and filtering.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    b = Butler(args.repo, collections=[args.stage3, args.stage2])
    reg = b.registry

    # The previous PVI-iteration approach didn't work because the registry
    # returns all-detector-1 PVIs first; to get detector diversity we'd have
    # to iterate hundreds of thousands of refs. Instead: enumerate VISITS
    # (visit dim DOES support a real band filter), then for each visit probe
    # a small set of candidate detectors and accept the first that has all
    # three inputs (PVI, footprints, same-band overlapping template).
    band_for_query = args.band or "g"

    # Spread of detector IDs across the LSSTCam focal plane (1..189 in the
    # registry). The order is the probe order; the first detector that
    # passes wins for that visit.
    candidate_detectors = [1, 50, 100, 150]

    visit_iter = reg.queryDimensionRecords(
        "visit",
        where="instrument='LSSTCam' AND band = :band",
        bind={"band": band_for_query},
    )
    # Materialize an upper-bound pool, then shuffle for stratification.
    visit_records: list[Any] = []
    for rec in visit_iter:
        visit_records.append(rec)
        if len(visit_records) >= args.candidate_pool:
            break
    rng.shuffle(visit_records)
    print(f"[info] visit pool: {len(visit_records)} (shuffled, band={band_for_query})")

    kept: list[dict] = []
    per_det: Counter[int] = Counter()
    scanned = 0

    for vrec in visit_records:
        scanned += 1
        v = vrec.id
        for d in candidate_detectors:
            if per_det[d] >= args.max_per_detector:
                continue
            did_full = {"instrument": "LSSTCam", "visit": int(v), "detector": int(d)}

            # 1. PVI exists?
            pvi_refs = list(reg.queryDatasets(
                "preliminary_visit_image",
                dataId=did_full,
                collections=[args.stage2],
                findFirst=True,
            ))
            if not pvi_refs:
                continue
            ref = pvi_refs[0]

            # 2. single_visit_star_footprints?
            srefs = list(reg.queryDatasets(
                "single_visit_star_footprints",
                dataId=did_full,
                collections=[args.stage2],
                findFirst=True,
            ))
            if not srefs:
                continue

            # 3. same-band overlapping template_coadd?
            try:
                expanded = reg.expandDataId(ref.dataId)
                region = expanded.region
                band = expanded.get("band")
            except Exception:
                continue
            if region is None or band is None:
                continue
            all_t = list(reg.queryDatasets(
                "template_coadd",
                where="skymap = :skymap AND patch.region OVERLAPS :region",
                bind={"skymap": args.skymap, "region": region},
                collections=[args.stage3],
                findFirst=True,
            ))
            gt = [r for r in all_t if r.dataId.get("band") == band]
            if not gt:
                continue

            kept.append({
                "visit": int(v),
                "detector": int(d),
                "band": band,
                "n_template_patches": int(len(gt)),
            })
            per_det[d] += 1
            break  # accept the first passing detector for this visit
        if len(kept) >= args.n_pairs:
            break

    manifest = {
        "repo": args.repo,
        "collections": [args.stage3, args.stage2],
        "skymap": args.skymap,
        "band_requested": args.band,
        "max_per_detector": args.max_per_detector,
        "seed": args.seed,
        "pairs": kept,
        "scanned": scanned,
        "n_unique_detectors": int(len(per_det)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    bands_seen = Counter(p["band"] for p in kept)
    print(f"wrote {out_path} with {len(kept)} pairs "
          f"(scanned {scanned}, unique detectors={len(per_det)}, bands={dict(bands_seen)})")
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
