"""Produce a large pool of VALIDATED (visit,detector) pairs for the streaming
producer, using the same select_good_refs_random_check (template-overlap + ref
checks) that the validated datasets used -- so the producer doesn't waste time on
pairs that fail PSF-matched subtraction. Excludes test_5sigma + test_real + the
already-generated realistic/realistic_big panels (-> new backgrounds, leakage-safe).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO / "ADCNN/data/dataset_creation"))
from simulate_inject_diffim import select_good_refs_random_check, _key_from_dataId

STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
WHERE = ("instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 "
         "AND band in ('u','g','r','i','z','y')")


def main():
    ex = set()
    for t in ["DATA_DIFFIM/test_5sigma/test.csv", "DATA_DIFFIM/test_real/test.csv",
              "DATA_DIFFIM_realistic/train.csv", "DATA_DIFFIM_realistic_big/train.csv"]:
        p = REPO / t
        if p.exists():
            d = pd.read_csv(p); ex |= {(int(v), int(dd)) for v, dd in zip(d.visit, d.detector)}
    print(f"excluding {len(ex)} (visit,detector) pairs (test + existing)", flush=True)
    refs = select_good_refs_random_check(
        repo="dp2_prep", collections=[STAGE3, STAGE2], where=WHERE,
        skymap="lsst_cells_v2", stage3_collection=STAGE3, instrument="LSSTCam",
        k=4000, seed=777, pool_size=8000, max_checks=400000,
        exclude_keys=ex, verbose=True)
    rows = [_key_from_dataId(r.dataId) for r in refs]
    out = REPO / "experiments/explore_simreal_gap/validated_pairs.csv"
    pd.DataFrame(rows, columns=["visit", "detector"]).to_csv(out, index=False)
    print(f"[done] {len(rows)} validated pairs -> {out}", flush=True)


if __name__ == "__main__":
    main()
