#!/usr/bin/env python3
"""Classify a processed night by ARTIFACT CONSISTENCY, so re-entry can tell done from half-done.

The campaign kept skipping incomplete nights because every stage guarded on `output file exists`,
and a stage that died mid-write leaves a file that exists but is wrong -- stream tracks written but
alerts.jsonl never emitted (20260705/06), a pairs/ dir holding a superseded link's images
(20260629/30), a cutout cache built for a different alert count. `exists` cannot see any of that.

This module answers one question per night -- what is the FIRST stage whose output is missing or
inconsistent -- from the artifacts alone. A campaign driver resumes there; a human reads the same
verdict. A night is COMPLETE only when every downstream artifact is present AND mutually
consistent, at which point a `.complete` sentinel is written so the common case is a cheap stat.

Stages, in order, with the consistency check each must pass:
  detect   adcnn_dets_masked.csv covers >= (1 - miss_tol) of the manifest panels
  link     stream/alerts.jsonl exists and is non-empty
  images   one stream/pairs/*.png per alertId, no duplicate ids, no orphan files
  sheets   stream/sheets/index.html + at least one sheet PNG
  summary  stream/stream_summary.json parses and its n_alerts matches alerts.jsonl

Usage:
  python -m ADCNN.pipelines.night_status outputs/runs/run_night_20260705      # one night
  python -m ADCNN.pipelines.night_status --all                                 # every run_night_*
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path

MISS_TOL = 0.05
STAGES = ["detect", "link", "images", "sheets", "summary"]
_IMG_RE = re.compile(r"^alert_\d+_p[\d.NA]+_(.+)_[A-Z']+\.png$")


def _panels(csv_path, cols=("visit", "detector")):
    import pandas as pd
    d = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    return set(zip(d[cols[0]].astype(int), d[cols[1]].astype(int)))


def status(run_dir):
    """-> dict(night, complete, first_missing, detail). first_missing is the earliest STAGE to
    (re)run, or None when COMPLETE."""
    R = Path(run_dir)
    sd = R / "stream"
    st = {"night": R.name.replace("run_night_", ""), "complete": False,
          "first_missing": None, "detail": {}}
    if (R / ".complete").exists():
        st["complete"] = True
        st["detail"]["sentinel"] = True
        return st

    # detect
    masked = R / "adcnn_dets_masked.csv"
    man = R / "manifest.csv"
    if not (masked.exists() and masked.stat().st_size > 0):
        st["first_missing"] = "detect"; st["detail"]["detect"] = "no masked dets"; return st
    if man.exists():
        want, got = _panels(man), _panels(masked)
        miss = len(want - got) / max(len(want), 1)
        st["detail"]["detect"] = f"{len(got)}/{len(want)} panels ({100*miss:.1f}% missing)"
        if miss > MISS_TOL:
            st["first_missing"] = "detect"; return st

    # link
    ap = sd / "alerts.jsonl"
    if not (ap.exists() and ap.stat().st_size > 0):
        st["first_missing"] = "link"; st["detail"]["link"] = "no stream alerts"; return st
    ids = [json.loads(l)["alertId"] for l in open(ap)]
    n = len(ids)
    st["detail"]["link"] = f"{n} alerts"

    # images: one file per alertId, no dup, no orphan
    imgs = glob.glob(str(sd / "pairs" / "alert_*.png"))
    fids = [m.group(1) for m in (_IMG_RE.match(os.path.basename(p)) for p in imgs) if m]
    idset, fidset = set(ids), set(fids)
    dup = len(fids) - len(fidset)
    missing_imgs = idset - fidset
    orphan = fidset - idset
    st["detail"]["images"] = (f"{len(imgs)} files, {dup} dup, {len(missing_imgs)} missing, "
                              f"{len(orphan)} orphan")
    if dup or missing_imgs or orphan:
        st["first_missing"] = "images"; return st

    # sheets
    if not ((sd / "sheets" / "index.html").exists() and glob.glob(str(sd / "sheets" / "*.png"))):
        st["first_missing"] = "sheets"; st["detail"]["sheets"] = "missing"; return st

    # summary
    sp = sd / "stream_summary.json"
    ok = False
    if sp.exists():
        try:
            ok = json.load(open(sp)).get("n_alerts") == n
        except Exception:
            ok = False
    if not ok:
        st["first_missing"] = "summary"; st["detail"]["summary"] = "missing/mismatch"; return st

    st["complete"] = True
    return st


def mark_complete(run_dir):
    (Path(run_dir) / ".complete").write_text("")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every outputs/runs/run_night_*")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    dirs = sorted(glob.glob("outputs/runs/run_night_*")) if a.all else a.run_dirs
    dirs = [d for d in dirs if os.path.isdir(d) and "_fill" not in d]
    rows = [status(d) for d in dirs]
    if a.json:
        print(json.dumps(rows, indent=2)); return
    for s in rows:
        v = "COMPLETE" if s["complete"] else f"-> {s['first_missing']}"
        det = "  ".join(f"{k}:{v2}" for k, v2 in s["detail"].items() if k != "sentinel")
        print(f"{s['night']:<10} {v:<12} {det}")


if __name__ == "__main__":
    sys.exit(main())
