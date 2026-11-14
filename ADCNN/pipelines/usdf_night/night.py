"""Resolve one Rubin day_obs to its prompt-processing difference images and write the panel manifest.

The night is the chain LSSTCam/runs/prompt-<day>. Only its RUN children under /runs/prompt/ are
searched (find_first over that ordered list), so a daytimeAP rerun sharing the chain can neither
shadow a prompt panel nor leak into the manifest. Repo `embargo` is tried first, then `main`; the
dataset type is future_difference_image where that holds the pixels, else difference_image.

  python -m ADCNN.pipelines.usdf_night.night check <YYYYMMDD|YYYY-MM-DD>
  python -m ADCNN.pipelines.usdf_night.night manifest <day> --out N/work/manifest.csv [--limit K]

check prints {repo, collection, dataset_type, runs, n_panels, n_visits} and exits 0 (also for a
night without diffims or without a chain); exit 2 = credentials / Butler access problem.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

REPOS = ("embargo", "main")
DATASET_TYPES = ("future_difference_image", "difference_image")
PROMPT = "/runs/prompt/"
MANIFEST_COLUMNS = ["visit", "detector", "band", "day_obs", "mjd_tai", "fits_path", "repo",
                    "dataset_type", "run"]


class CredentialError(RuntimeError):
    """Butler or S3 access is not set up for this user."""


def normalize_day(s) -> str:
    """'20261113' or '2026-11-13' -> '20261113'; anything else (or a non-date) -> ValueError."""
    s = str(s).strip()
    m = re.fullmatch(r"(\d{4})-?(\d{2})-?(\d{2})", s)
    if not m or ("-" in s and s.count("-") != 2):
        raise ValueError(f"day must be YYYYMMDD or YYYY-MM-DD, got {s!r}")
    day = "".join(m.groups())
    dt.datetime.strptime(day, "%Y%m%d")
    return day


def chain_name(day: str) -> str:
    return f"LSSTCam/runs/prompt-{day}"


def day_where(day: str) -> str:
    return f"instrument='LSSTCam' AND visit.day_obs={day}"


def prompt_runs(collections) -> list:
    """The chain's children that are prompt-processing runs, in chain (find_first) order."""
    return [c for c in collections if PROMPT in c]


def credentials_problem(env=None):
    """One-line reason why Butler/S3 access cannot work, or None. Never reads credential values."""
    env = os.environ if env is None else env
    if not env.get("DAF_BUTLER_REPOSITORY_INDEX"):
        return "DAF_BUTLER_REPOSITORY_INDEX is not set: load the LSST stack (loadLSST.sh; setup lsst_distrib)"
    path = Path(env.get("AWS_SHARED_CREDENTIALS_FILE")
                or Path(env.get("HOME", "~")).expanduser() / ".lsst" / "aws-credentials.ini")
    if not path.is_file():
        return f"S3 credentials file {path} not found: embargo access needs your own USDF embargo credentials"
    with open(path, errors="replace") as f:
        if not any(line.strip() == "[embargo]" for line in f):
            return f"S3 credentials file {path} has no [embargo] profile"
    return None


def open_butler(repo: str):
    """Butler for `repo` after a registry round trip; any failure -> CredentialError (one line)."""
    from lsst.daf.butler import Butler
    log = logging.getLogger("lsst.daf.butler")
    level = log.level
    log.setLevel(logging.CRITICAL)      # the failure is reported below, in one line
    try:
        b = Butler(repo)
        b.query_dimension_records("instrument", limit=None, explain=False)
    except Exception as e:
        msg = (str(e).strip().splitlines() or [""])[0][:200]
        raise CredentialError(f"cannot open Butler repo '{repo}' ({type(e).__name__}: {msg}); "
                              "check your USDF Butler database credentials") from e
    finally:
        log.setLevel(level)
    return b


def find_panels(butler, day: str):
    """(dataset_type, refs) of the night's prompt diffims in one repo; (None, []) when there are none."""
    from lsst.daf.butler import MissingCollectionError, MissingDatasetTypeError
    try:
        runs = prompt_runs(butler.collections.query(chain_name(day), flatten_chains=True))
    except MissingCollectionError:
        return None, []
    if not runs:
        return None, []
    for dataset_type in DATASET_TYPES:
        try:
            refs = butler.query_datasets(dataset_type, collections=runs, where=day_where(day),
                                         find_first=True, limit=None, explain=False)
        except MissingDatasetTypeError:
            continue
        if refs:
            return dataset_type, refs
    return None, []


def resolve(day: str, open_fn=open_butler) -> dict:
    """First repo in REPOS holding prompt diffims for `day`."""
    for repo in REPOS:
        b = open_fn(repo)
        dataset_type, refs = find_panels(b, day)
        if refs:
            return {"repo": repo, "butler": b, "dataset_type": dataset_type, "refs": refs}
    return {"repo": None, "butler": None, "dataset_type": None, "refs": []}


def summary(day: str, res: dict) -> dict:
    refs = res["refs"]
    return {"repo": res["repo"], "collection": chain_name(day), "dataset_type": res["dataset_type"],
            "runs": sorted({r.run for r in refs}), "n_panels": len(refs),
            "n_visits": len({int(r.dataId["visit"]) for r in refs})}


def visit_midpoints(butler, day: str) -> dict:
    """visit -> (TAI MJD of the visit-record timespan midpoint, day_obs)."""
    out = {}
    for r in butler.query_dimension_records("visit", where=day_where(day), limit=None, explain=False):
        ts = r.timespan
        out[int(r.id)] = ((ts.begin.tai.mjd + ts.end.tai.mjd) / 2.0, int(r.day_obs))
    return out


def uri_string(uri) -> str:
    """Manifest fits_path: POSIX path for a local datastore, full URI (s3://...) otherwise."""
    return uri.ospath if uri.scheme in ("", "file") else uri.geturl()


def sorted_refs(refs, limit=None):
    refs = sorted(refs, key=lambda r: (int(r.dataId["visit"]), int(r.dataId["detector"])))
    return refs[:limit] if limit else refs


def manifest_frame(butler, repo: str, refs, day: str):
    import pandas as pd
    mids = visit_midpoints(butler, day)
    uris = butler.get_many_uris(refs)
    rows = []
    for r in refs:
        v, d = int(r.dataId["visit"]), int(r.dataId["detector"])
        mjd, day_obs = mids[v]
        rows.append((v, d, str(r.dataId["band"]), day_obs, mjd, uri_string(uris[r].primaryURI), repo,
                     r.datasetType.name, r.run))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(["visit", "detector"])
    if df.duplicated(["visit", "detector"]).any():
        raise RuntimeError("duplicate (visit, detector) panels in the manifest")
    return df.reset_index(drop=True)


def lsst_weekly():
    m = re.search(r"w_\d{4}_\d{2}", os.environ.get("LSST_DISTRIB_DIR", ""))
    return m.group(0) if m else None


def update_info(path, section: str, payload: dict) -> None:
    """Merge one stage's record into prep_info.json (atomic rewrite)."""
    path = Path(path)
    info = json.loads(path.read_text()) if path.exists() else {}
    info[section] = payload
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=1) + "\n")
    os.replace(tmp, path)


def write_atomic(df, out) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    if out.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="night", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").add_argument("day")
    pm = sub.add_parser("manifest")
    pm.add_argument("day")
    pm.add_argument("--out", required=True)
    pm.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    try:
        day = normalize_day(a.day)
    except ValueError as e:
        print(f"night: {e}", file=sys.stderr)
        return 1
    t0 = time.time()
    try:
        problem = credentials_problem()
        if problem:
            raise CredentialError(problem)
        res = resolve(day)
    except CredentialError as e:
        print(f"night {day}: {e}", file=sys.stderr)
        return 2
    if a.cmd == "check":
        print(json.dumps(summary(day, res)))
        return 0
    refs = sorted_refs(res["refs"], a.limit)
    res["refs"] = refs
    if refs:
        df = manifest_frame(res["butler"], res["repo"], refs, day)
    else:
        import pandas as pd
        df = pd.DataFrame(columns=MANIFEST_COLUMNS)
    write_atomic(df, a.out)
    info = summary(day, res)
    info.update(night=day, limit=a.limit, lsst_weekly=lsst_weekly(),
                lsst_distrib_dir=os.environ.get("LSST_DISTRIB_DIR"), seconds=round(time.time() - t0, 1))
    update_info(Path(a.out).parent / "prep_info.json", "manifest", info)
    print(f"[manifest] {len(df)} panels, {df.visit.nunique()} visits, {res['repo']} "
          f"{res['dataset_type']} -> {a.out} ({info['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
