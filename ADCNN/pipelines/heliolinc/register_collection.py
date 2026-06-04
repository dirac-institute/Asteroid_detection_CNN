"""Register the private Butler RUN `u/mrakovci/ADCNN/samenight_link_lambda` and its dataset types, and
provide put/get helpers used to publish the campaign's deliverables (the permanent record).

Granularity: per-panel injected diffims are ~13 TB so they are NOT materialised (injection is deterministic
inline -- regenerable from inject_*.csv + seed). What we PUBLISH into the collection is the meaningful,
reproducible record as a few EMPTY-DIMENSION table datasets (one per type per run):
  - samenight_lambda_detections : master per-detection catalog (ADCNN score, stack-5sigma flag, injected
                                  objID/SNR truth, re-timed mjd, trail geometry) over all fields
  - samenight_lambda_curve      : lambda(S), completeness(S) by SNR bin
  - samenight_lambda_result     : the headline S* (lambda=1.35e-3) + metadata
Fallback: if writeable dp2_prep is denied, parquet under <run>/butler_fallback/ with the same names.
"""
from __future__ import annotations
import argparse
from pathlib import Path

RUN = "u/mrakovci/ADCNN/samenight_link_lambda"
TYPES = ("samenight_lambda_detections", "samenight_lambda_curve", "samenight_lambda_result")
FALLBACK = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/"
                "ADCNN/pipelines/heliolinc/run_lambda/butler_fallback")


def get_butler():
    """Return (butler, ok) -- ok False => use disk fallback."""
    try:
        from lsst.daf.butler import Butler
        return Butler("dp2_prep", writeable=True), True
    except Exception as e:
        print(f"[register] writeable Butler unavailable ({type(e).__name__}: {str(e)[:120]}) -> disk fallback", flush=True)
        return None, False


def register(butler):
    from lsst.daf.butler import DatasetType
    butler.registry.registerRun(RUN)
    for name in TYPES:
        dt = DatasetType(name, dimensions=[], storageClass="ArrowAstropy", universe=butler.dimensions)
        try:
            butler.registry.registerDatasetType(dt)
        except Exception as e:
            print(f"[register] {name}: {type(e).__name__} {str(e)[:80]}", flush=True)
    print(f"[register] RUN {RUN} + {len(TYPES)} dataset types ready", flush=True)


def publish(name, df):
    """Put a pandas DataFrame as `name` into the RUN (or parquet fallback). Idempotent (prunes existing)."""
    from astropy.table import Table
    t = Table.from_pandas(df)
    butler, ok = get_butler()
    if ok:
        register(butler)
        old = list(butler.registry.queryDatasets(name, collections=RUN)) if butler.registry.queryDatasetTypes(name) else []
        if old:
            butler.pruneDatasets(old, purge=True, unstore=True)
        butler.put(t, name, dataId={}, run=RUN)
        print(f"[register] published {name} ({len(df)} rows) -> {RUN}", flush=True)
    else:
        FALLBACK.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FALLBACK / f"{name}.parquet")
        print(f"[register] wrote fallback {FALLBACK/(name+'.parquet')} ({len(df)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="just register the RUN + dataset types")
    ap.parse_args()
    butler, ok = get_butler()
    if ok:
        register(butler)
    else:
        FALLBACK.mkdir(parents=True, exist_ok=True)
        print(f"[register] fallback dir ready: {FALLBACK}", flush=True)


if __name__ == "__main__":
    main()
