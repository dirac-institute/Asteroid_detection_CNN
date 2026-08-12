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
IMAGE_CAP = 20000            # run_night renders the top --stream-pairs-top-n alerts (default 20000);
                            # a night with MORE alerts images only the top IMAGE_CAP, so the
                            # verifier must require images for min(n_alerts, IMAGE_CAP), not all n.
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
    # The sentinel is a CACHE, not proof. It was a one-way latch: `touch .complete` on an empty
    # directory reported COMPLETE, and all nine real sentinels PREDATED the artifacts they certify
    # (0629: sentinel 08-01, catalogue 08-11). Trust it only if it is newer than the artifacts.
    _sent = R / ".complete"
    if _sent.exists():
        # The sentinel certifies these artifacts. It is invalid if any REQUIRED one is missing (a
        # bare `touch .complete` on an empty directory reported COMPLETE) or if any is NEWER than the
        # sentinel (all nine real sentinels predated the artifacts they certified).
        # The sentinel must cover EVERYTHING status() would otherwise verify, or it becomes a way to
        # bypass the very checks this module exists for: with a valid sentinel, deleting the entire
        # stream/pairs image product still reported COMPLETE. Include the frozen science product --
        # run_night_20260629 carries a sentinel with NO alerts.jsonl, tracks.csv or report/, and can
        # never self-heal because run_night sees COMPLETE and does nothing.
        _required = (R / "adcnn_dets_masked.csv", R / "stream" / "alerts.jsonl",
                     R / "stream" / "stream_summary.json")
        _required_dirs = (R / "stream" / "pairs", R / "stream" / "sheets")
        _missing = [f for f in _required if not (f.exists() and f.stat().st_size > 0)]
        _missing += [d for d in _required_dirs if not (d.is_dir() and any(d.iterdir()))]
        _newer = [f for f in (R / "adcnn_dets_masked.csv", R / "dets_merged.csv",
                              R / "stream" / "alerts.jsonl")
                  if f.exists() and f.stat().st_mtime > _sent.stat().st_mtime]
        if _missing:
            print(f"[night_status] .complete is INVALID (missing/empty "
                  f"{', '.join(f.name for f in _missing)}) -- re-verifying", flush=True)
            _sent.unlink()
        elif _newer:
            print(f"[night_status] .complete is STALE (older than {', '.join(f.name for f in _newer)}) "
                  f"-- re-verifying", flush=True)
            _sent.unlink()
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
    # A TRUNCATED alerts.jsonl must degrade to "rerun the link stage", never raise. run_night calls
    # status() on ENTRY, before preflight, so an unguarded json.loads wedged the night permanently:
    # all three campaign retries burned on an identical instant crash. This is exactly the half-write
    # class this module exists to catch -- it handled empty files but not partial lines.
    try:
        ids = [json.loads(l)["alertId"] for l in open(ap)]
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
        print(f"[night_status] {ap} is unreadable ({type(e).__name__}) -- treating the link stage as "
              f"incomplete so it is rebuilt", flush=True)
        # MUST return the status DICT. Returning a bare string here reproduced the very wedge this
        # guard was added to remove: every caller indexes the result (`_s["complete"]` in run_night,
        # `s["first_missing"]` in main), so a string raised TypeError on entry -- before preflight,
        # so all retries crashed identically. Same wedge, different exception.
        st["first_missing"] = "link"; st["detail"]["link"] = f"unreadable ({type(e).__name__})"
        return st
    n = len(ids)
    st["detail"]["link"] = f"{n} alerts"

    # images: one file per alertId for the TOP min(n, IMAGE_CAP) ranked alerts (alerts.jsonl is
    # rank-ordered), no dup, no orphan. Beyond the cap is intentionally un-rendered, not missing.
    want_img = set(ids[:min(n, IMAGE_CAP)])
    imgs = glob.glob(str(sd / "pairs" / "alert_*.png"))
    fids = [m.group(1) for m in (_IMG_RE.match(os.path.basename(p)) for p in imgs) if m]
    fidset = set(fids)
    dup = len(fids) - len(fidset)
    missing_imgs = want_img - fidset
    orphan = fidset - set(ids)                        # a file naming an alert not in the stream
    st["detail"]["images"] = (f"{len(imgs)} files, top {len(want_img)} expected, {dup} dup, "
                              f"{len(missing_imgs)} missing, {len(orphan)} orphan")
    # PRESENCE IS NOT CORRECTNESS. alertId and count cannot see a PERMUTATION, and that is exactly
    # how six nights certified COMPLETE while every rendered image carried another alert's caption
    # (the cache is keyed by alert POSITION and rerank_alerts rewrote alerts.jsonl after the cut).
    # alert_cutouts now records a sha256 of the full (visit,detector) sequence; if the cache is still
    # around, check the pixels were cut from THIS file. A cache is deleted after a successful night,
    # so its absence is normal and not a failure -- what must never happen is certifying a MISMATCH.
    mp = sd / "cutouts_meta.json"
    if imgs and mp.exists():
        try:
            fp = json.load(open(mp)).get("alerts_fingerprint")
        except (json.JSONDecodeError, OSError):
            fp = None
        if fp:
            import hashlib
            h = hashlib.sha256()
            with open(ap) as f:
                for line in f:
                    for e in (json.loads(line).get("epochs") or []):
                        h.update(f"{e.get('visit',-1)}:{e.get('detector',-1)};".encode())
                    h.update(b"|")
            if h.hexdigest() != fp:
                st["detail"]["images"] += "; CACHE MISMATCH -- images show the WRONG alerts"
                st["first_missing"] = "images"
                return st
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
    # Re-write the sentinel after a successful re-verification, so the cheap stat path is restored
    # for the next caller. Verification invalidates a stale sentinel (see above); without this, every
    # subsequent call would redo the full artifact scan on a night that is genuinely finished.
    try:
        (R / ".complete").write_text("")
    except OSError:
        pass
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
