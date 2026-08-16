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
  summary  stream/stream_summary.json parses and its n_alerts matches alerts.jsonl
  deliver  stream_1k/ (the ~1k clean product), WHEN PRESENT: alerts.jsonl parses, one pairs image
           per alert, summary count matches. An absent stream_1k is NOT a failure --
           run_night alone does not build it (regen_campaign does) -- but a half-built one must
           never certify: the campaign wrote .regen_complete off this module's verdict while the
           1k build's rc was logged and IGNORED, so a failed 1k stage was skipped forever.

Usage:
  python -m ADCNN.pipelines.night_status outputs/runs/run_night_20260705      # one night
  python -m ADCNN.pipelines.night_status --all                                 # every night, both layouts
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path

MISS_TOL = 0.05
IMAGE_CAP = 20000            # run_night renders the top --stream-pairs-top-n alerts (default 20000);
                            # a night with MORE alerts images only the top IMAGE_CAP, so the
                            # verifier must require images for min(n_alerts, IMAGE_CAP), not all n.
STAGES = ["detect", "link", "images", "summary", "deliver"]
_IMG_RE = re.compile(r"^alert_\d+_p[\d.NA]+_(.+)_[A-Z']+\.png$")


def _panels(csv_path, cols=("visit", "detector")):
    import pandas as pd
    d = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    return set(zip(d[cols[0]].astype(int), d[cols[1]].astype(int)))


def _cache_fingerprint_ok(alerts_path, meta_path):
    """-> (checked, ok). Does the cutout cache beside `meta_path` belong to THIS alerts file?

    alert_cutouts records a sha256 over the (visit,detector) sequence of `alerts[:limit]`. A cache
    is deleted after a successful night, so absence is normal -- `checked` is False then and the
    caller must not treat that as a failure. What must never happen is certifying a MISMATCH:
    the cache is keyed by alert POSITION, so a permutation of alerts.jsonl silently re-points every
    rendered image at another alert's caption. That shipped on six nights.

    ALSO hashes the sky position, not just (visit,detector): two alerts on the SAME panel pair are
    interchangeable under a detector-only signature, and 8.8-25.2% of delivered alerts share an
    epoch signature with at least one other. Rounding to ~0.4 arcsec keeps float noise out.
    """
    from ADCNN.qa.cache_identity import verify
    if not meta_path.exists():
        return False, True
    try:
        meta = json.load(open(meta_path))
    except (json.JSONDecodeError, OSError):
        return False, True
    return verify(alerts_path, meta)


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
        # The sentinel must cover the DELIVERED product too. With stream_1k absent from this list,
        # deleting 999 of 1000 stream_1k/pairs PNGs still reported COMPLETE -- and run_night's
        # re-entry returns immediately on COMPLETE, so such a night could never self-heal. Presence
        # of stream_1k stays optional (run_night alone does not build it), but ONCE IT EXISTS its
        # artifacts are required, or the deliver stage below is bypassable by a valid sentinel.
        _required = [R / "adcnn_dets_masked.csv", R / "stream" / "alerts.jsonl",
                     R / "stream" / "stream_summary.json"]
        _required_dirs = [R / "stream" / "pairs"]
        if (R / "stream_1k").is_dir():
            _required += [R / "stream_1k" / "alerts.jsonl", R / "stream_1k" / "stream_summary.json"]
            _required_dirs += [R / "stream_1k" / "pairs"]
        _missing = [f for f in _required if not (f.exists() and f.stat().st_size > 0)]
        _missing += [d for d in _required_dirs if not (d.is_dir() and any(d.iterdir()))]
        # ...and a 1:1 COUNT check: "directory is non-empty" cannot see 999 of 1000 images deleted.
        if not _missing:
            for _adir, _apath in ((R / "stream" / "pairs", R / "stream" / "alerts.jsonl"),
                                  (R / "stream_1k" / "pairs", R / "stream_1k" / "alerts.jsonl")):
                if _adir.is_dir() and _apath.exists():
                    _nal = min(sum(1 for _ in open(_apath)), IMAGE_CAP)
                    if len(glob.glob(str(_adir / "alert_*.png"))) < _nal:
                        _missing.append(_adir)
                        break
        # stream_1k is in the NEWER list but not the REQUIRED list: its absence is a legitimate
        # state (run_night alone does not build it), but a 1k product rebuilt after the sentinel
        # must force re-verification or the deliver stage below is bypassable by a stale sentinel.
        _newer = [f for f in (R / "adcnn_dets_masked.csv", R / "dets_merged.csv",
                              R / "stream" / "alerts.jsonl", R / "stream_1k" / "alerts.jsonl")
                  if f.exists() and f.stat().st_mtime > _sent.stat().st_mtime]
        if _missing:
            # STDERR, not stdout. `--json` writes the report to STDOUT, and regen_campaign
            # captures that into regen_status.json -- a diagnostic on stdout lands INSIDE the JSON
            # and the driver's json.load() dies, discarding a verdict that says complete=true. The
            # night is then never marked done and the next pass rm -rf's and rebuilds the delivered
            # product: an unbounded loop. Diagnostics are diagnostics; they belong on stderr.
            print(f"[night_status] .complete is INVALID (missing/empty "
                  f"{', '.join(f.name for f in _missing)}) -- re-verifying", file=sys.stderr, flush=True)
            _sent.unlink()
        elif _newer:
            print(f"[night_status] .complete is STALE (older than {', '.join(f.name for f in _newer)}) "
                  f"-- re-verifying", file=sys.stderr, flush=True)
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
              f"incomplete so it is rebuilt", file=sys.stderr, flush=True)
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
    # Identity is ADCNN.qa.cache_identity's business now -- one implementation for the writer and
    # both verifiers, prefix-aware and version-gated. A cache is deleted after a successful night,
    # so its absence is normal; what must never happen is certifying a MISMATCH.
    _chk, _ok = _cache_fingerprint_ok(ap, sd / "cutouts_meta.json")
    if imgs and _chk and not _ok:
        st["detail"]["images"] += "; CACHE MISMATCH -- images show the WRONG alerts"
        st["first_missing"] = "images"
        return st
    if dup or missing_imgs or orphan:
        st["first_missing"] = "images"; return st

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

    # deliver: the ~1k clean product, verified WHEN PRESENT. run_night alone does not build
    # stream_1k (regen_campaign does), so absence is a legitimate state -- but a half-built one must
    # never certify. Before this stage the campaign wrote .regen_complete off this module's verdict
    # while the 1k chain's rc was logged and IGNORED: a night whose 1k build died mid-way was marked
    # VERIFIED COMPLETE and skipped by every later re-entry.
    kd = R / "stream_1k"
    if kd.is_dir():
        kap = kd / "alerts.jsonl"
        if not (kap.exists() and kap.stat().st_size > 0):
            st["first_missing"] = "deliver"; st["detail"]["deliver"] = "stream_1k present but no alerts"
            return st
        try:
            kids = [json.loads(l)["alertId"] for l in open(kap)]
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            st["first_missing"] = "deliver"
            st["detail"]["deliver"] = f"unreadable ({type(e).__name__})"
            return st
        kimgs = glob.glob(str(kd / "pairs" / "alert_*.png"))
        kfids = {m.group(1) for m in (_IMG_RE.match(os.path.basename(p_)) for p_ in kimgs) if m}
        k_missing = set(kids) - kfids
        k_orphan = kfids - set(kids)
        k_sum = None
        try:
            k_sum = json.load(open(kd / "stream_summary.json")).get("n_alerts")
        except Exception:
            pass
        st["detail"]["deliver"] = (f"{len(kids)} delivered, {len(kimgs)} pair files, "
                                   f"{len(k_missing)} missing, {len(k_orphan)} orphan, "
                                   f"summary={'ok' if k_sum == len(kids) else k_sum}")
        if k_missing or k_orphan or k_sum != len(kids):
            st["first_missing"] = "deliver"
            return st
        # THE PERMUTATION CLASS, on the product that is actually delivered. It was guarded on the
        # QA stream and unguarded here: reversing stream_1k/alerts.jsonl reported COMPLETE while
        # the identical reversal of stream/ correctly reported CACHE MISMATCH. Counts and ids
        # cannot see a permutation; only the fingerprint can.
        _chk, _ok = _cache_fingerprint_ok(kap, kd / "cutouts_meta.json")
        if _chk and not _ok:
            st["detail"]["deliver"] += "; CACHE MISMATCH -- delivered images show the WRONG alerts"
            st["first_missing"] = "deliver"
            return st

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
    ap.add_argument("--all", action="store_true",
                    help="every outputs/runs/run_night_* AND outputs/runs/*/run_night_*")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    # BOTH layouts. The outputs reorg moved every real product to runs/<campaign>/run_night_*, so
    # globbing only the flat path made `--all` report on nothing but leftover dry-run stubs -- it
    # printed one bogus row and rc=0 while nine delivered nights were invisible.
    dirs = (sorted(set(glob.glob("outputs/runs/run_night_*")
                       + glob.glob("outputs/runs/*/run_night_*")))
            if a.all else a.run_dirs)
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
