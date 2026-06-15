"""ADCNN operational nightly pipeline -- APPLIES a frozen release to one night of real data.

Second of the two top-level pipelines (the first is ``train_and_validate``, which DECIDES + freezes
the operating point). ``run_night`` loads a frozen release, VERIFIES its integrity, runs nightly
inference + same-night 2-visit linking under the FROZEN alert cuts, and emits ranked NEO alert
candidates plus a runtime report. It is a faithful Python orchestrator of the canonical morning
chain (``ADCNN/pipelines/heliolinc/sn_run.slurm``) with three additions:

  * integrity preflight  -- verify model md5s + MF_LEN + the alert cuts equal the frozen values
                            (fail loud; a frozen release must not silently drift);
  * alert-op default     -- default the link op-point to the pipeline's ``alert_op_point``
                            (``op_2v_alert.json``, mfsnr>=5), NOT the discovery op
                            (``link_op_point.json``, mfsnr>=10). The discovery op is opt-in via
                            ``--discovery``/``--op-point`` (the alert product is the default here);
  * runtime telemetry    -- wall time per stage, per visit, per detector-pass, per night
                            (``runtime_report.json`` + a QA plot).

Stages (mirroring sn_run.slurm): build-manifest (LSST stack env) -> detect (GPU sbatch) ->
build-known (LSST stack env) -> mask-flags -> link-2visit (-> tracks.csv + ranked alerts.jsonl).

Speed contract: the linker's cheap prefilters (chord seeding + partial-chi2 pre-gates) only PRUNE
candidates; the final orbit-fit chi2 and physical_check gates stay exact -- final measurements and
gates are reproducible, approximation is for pruning only.

    # dry-run one night (prints the exact chain, runs the integrity preflight):
    python -m ADCNN.pipelines.run_night --pipeline models/current/pipeline.json \
        --butler-repo dp2_prep --collection LSSTCam/runs/.../DM-XXXXX --night 20250718 \
        --tracts 8489 --out run_night_20250718 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from ADCNN.config import load_pipeline, REPO

HL = REPO / "ADCNN" / "pipelines" / "heliolinc"
DEFAULT_ALERT_OP = HL / "op_2v_alert.json"
DISCOVERY_OP = HL / "link_op_point.json"
FROZEN_ALERT_GOLDEN = {"score_min": 0.80, "chi2_2v_max": 5.0, "mfsnr_min_2v": 5.0,
                       "rate_lo_2v": 1.0, "rate_hi_2v": 8.0, "alerts_top_n": 50}
# the discovery product is a SEPARATE op (mfsnr>=10, uncapped) -- validated only when --discovery.
FROZEN_DISCOVERY_GOLDEN = {"score_min": 0.80, "chi2_2v_max": 5.0, "mfsnr_min_2v": 10.0,
                           "rate_lo_2v": 1.0, "rate_hi_2v": 8.0}


class IntegrityError(RuntimeError):
    """Raised when a frozen release fails its preflight integrity checks."""


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- preflight
def preflight(pipe, op_point_path, discovery=False):
    """Verify the frozen release is intact before spending a night of compute. Fail loud.

    Checks: (1) stage-1/stage-2 md5s match the recorded identity (the freeze dir's md5s.json if
    present, else models/v2_D/v2_D_release.json); (2) the op-point we will apply matches the frozen
    cuts for the SELECTED product -- the release thresholds.json or alert golden for the default
    alert product, the discovery golden for ``--discovery``; (3) MF_LEN is internally consistent.
    """
    print("  preflight:")
    ok = True
    src_dir = pipe.source.parent

    # (1) model md5s vs the recorded identity
    md5s_file = src_dir / "md5s.json"
    if md5s_file.exists():
        recorded = json.loads(md5s_file.read_text())
        identity = {"stage1.pt": recorded.get("stage1.pt"), "stage2.pt": recorded.get("stage2.pt")}
        got = {"stage1.pt": _md5(pipe.seg_model), "stage2.pt": _md5(pipe.cnn_model)}
        src = "md5s.json"
    else:
        rel = json.loads((REPO / "models/v2_D/v2_D_release.json").read_text())
        identity = {"stage1.pt": rel["models"]["stage1_segmentation_scripted_md5"],
                    "stage2.pt": rel["models"]["stage2_cnn_postproc_md5"]}
        got = {"stage1.pt": _md5(pipe.seg_model), "stage2.pt": _md5(pipe.cnn_model)}
        src = "v2_D_release.json"
    for name in ("stage1.pt", "stage2.pt"):
        m_ok = identity[name] == got[name]
        ok &= m_ok
        print(f"    [{'OK' if m_ok else 'FAIL'}] md5 {name} = {str(got[name])[:12]} "
              f"(recorded {str(identity[name])[:12]} in {src})")

    # (2) the op-point we will apply matches the frozen cuts for the SELECTED product
    op = json.loads(Path(op_point_path).read_text())
    thr_file = src_dir / "thresholds.json"
    if discovery:
        want = FROZEN_DISCOVERY_GOLDEN
        why = "frozen discovery golden values"
    elif thr_file.exists():
        thr = json.loads(thr_file.read_text())
        want = {"score_min": thr["score_min"], "chi2_2v_max": thr["chi2_max"],
                "mfsnr_min_2v": thr["mfsnr_min"], "rate_lo_2v": thr["rate_lo"],
                "rate_hi_2v": thr["rate_hi"], "alerts_top_n": thr.get("budget_top_n", 50)}
        why = "release thresholds.json"
    else:
        want = FROZEN_ALERT_GOLDEN
        why = "frozen alert golden values"
    op_ok = all(op.get(k) == v for k, v in want.items())
    ok &= op_ok
    print(f"    [{'OK' if op_ok else 'FAIL'}] link op-point {Path(op_point_path).name} matches {why} {want}")

    # (3) MF_LEN internal consistency (pipeline-resolved values are finite/positive)
    mf_ok = pipe.mf_len_slope > 0 and pipe.mf_len_offset >= 0
    ok &= mf_ok
    print(f"    [{'OK' if mf_ok else 'FAIL'}] MF_LEN offset/slope = {pipe.mf_len_offset}/{pipe.mf_len_slope}")

    if not ok:
        raise IntegrityError("run_night preflight FAILED -- the frozen release drifted (see above). "
                             "Refusing to run a night on a non-intact release.")
    print("    preflight PASSED.")


# --------------------------------------------------------------------------------------------- chain
class _Timer:
    """Accumulate per-stage wall time into a runtime report."""
    def __init__(self):
        self.stages = []
        self.t0 = time.monotonic()

    def stage(self, name, fn):
        t = time.monotonic()
        fn()
        dt = time.monotonic() - t
        self.stages.append({"stage": name, "seconds": round(dt, 3)})
        print(f"    [{name}] {dt:.1f}s")

    def report(self, n_visits, n_detector_passes):
        total = round(time.monotonic() - self.t0, 3)
        det = next((s["seconds"] for s in self.stages if s["stage"] == "detect"), None)
        return {
            "stages": self.stages, "total_seconds": total,
            "n_visits": n_visits, "n_detector_passes": n_detector_passes,
            "per_visit_seconds": round(det / n_visits, 3) if det and n_visits else None,
            "per_detector_pass_seconds": round(det / n_detector_passes, 3)
            if det and n_detector_passes else None,
            "per_night_seconds": total,
        }


def _bash(cmd, dry):
    print(f"      $ {cmd}")
    if not dry:
        subprocess.run(cmd, shell=True, check=True)


def _manifest_counts(manifest):
    """(n_visits, n_detector_passes) from a built manifest, else (None, None)."""
    p = Path(manifest)
    if not p.exists():
        return None, None
    import csv
    visits = set()
    rows = 0
    with open(p) as fh:
        for r in csv.DictReader(fh):
            rows += 1
            if "visit" in r:
                visits.add(r["visit"])
    return (len(visits) or None), (rows or None)


def run(a):
    pipe = load_pipeline(a.pipeline)
    op_point = a.op_point or (str(DISCOVERY_OP) if a.discovery else str(pipe.alert_op_point or DEFAULT_ALERT_OP))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    print(f"[run_night] pipeline={pipe.name} provenance={pipe.provenance}")
    print(f"  night={a.night} tracts={a.tracts} collection={a.collection}")
    print(f"  op-point={op_point} ({'DISCOVERY' if op_point == str(DISCOVERY_OP) else 'ALERT (default)'})")

    preflight(pipe, op_point, discovery=a.discovery)

    lsst = a.lsst_setup
    tm = _Timer()

    def s_manifest():
        if manifest.exists() and manifest.stat().st_size > 0 and not a.force:
            print("      (manifest exists; reuse)"); return
        cmd = (f"bash -c '{lsst}; cd {REPO}; python -m ADCNN.pipelines.heliolinc.build_manifest "
               f"--tracts {a.tracts} --day-start {a.night} --day-end {int(a.night)+1} "
               f"--butler-repo {shlex.quote(a.butler_repo)} --collection {shlex.quote(a.collection)} "
               f"--out {manifest}'")
        _bash(cmd, a.dry_run)

    def s_detect():
        cmd = (f"RUN={out} sbatch --export=ALL,RUN,SEGMODEL={pipe.seg_model},CNNMODEL={pipe.cnn_model} "
               f"--wait {HL/'sn_detect.slurm'}   # GPU; resumable; -> {out}/adcnn_dets.csv")
        _bash(cmd, a.dry_run)

    def s_known():
        kn = out / "known.csv"
        if kn.exists() and not a.force:
            print("      (known.csv exists; reuse)"); return
        cmd = (f"bash -c '{lsst}; cd {REPO}; python -m ADCNN.pipelines.heliolinc.build_known_catalog "
               f"--manifest {manifest} --butler-repo {shlex.quote(a.butler_repo)} "
               f"--collection {shlex.quote(a.collection)} --out {kn}'")
        _bash(cmd, a.dry_run)

    def s_mask():
        cmd = (f"python -m ADCNN.pipelines.heliolinc.mask_flags --dets {out}/adcnn_dets.csv "
               f"--manifest {manifest} --out {out}/adcnn_dets_masked.csv --workers {a.mask_workers}")
        _bash(cmd, a.dry_run)

    def s_link():
        cmd = (f"python -m ADCNN.pipelines.heliolinc.trail_state_link --dets {out}/adcnn_dets_masked.csv "
               f"--known {out}/known.csv --out {out}/tracks.csv --op-point {op_point} "
               f"--npt 2 --min-epochs 2 --seed-2v chord --alerts-out {out}/alerts.jsonl")
        _bash(cmd, a.dry_run)

    tm.stage("build_manifest", s_manifest)
    tm.stage("detect", s_detect)
    tm.stage("build_known", s_known)
    tm.stage("mask_flags", s_mask)
    tm.stage("link_2visit", s_link)

    n_visits, n_passes = _manifest_counts(manifest)
    rep = tm.report(n_visits, n_passes)
    rep.update({"night": a.night, "tracts": a.tracts, "collection": a.collection,
                "pipeline": pipe.name, "op_point": op_point, "dry_run": bool(a.dry_run)})
    (out / "runtime_report.json").write_text(json.dumps(rep, indent=2))
    print(f"  runtime_report.json -> per-visit {rep['per_visit_seconds']}s, "
          f"per-detector-pass {rep['per_detector_pass_seconds']}s, night {rep['per_night_seconds']}s")
    _plot_runtime(out, rep, a.dry_run)
    print(f"[run_night] done -> {out}/ (tracks.csv + alerts.jsonl + runtime_report.json)")
    return rep


def _plot_runtime(out, rep, dry):
    try:
        from ADCNN.qa.plots_runtime import plot_runtime
        plot_runtime(rep, out / "runtime.png")
        print(f"  runtime.png -> {out}/runtime.png")
    except Exception as e:
        # qa.plots_runtime lands with the reorg; until then this is a documented no-op.
        if not dry:
            print(f"  (runtime plot deferred: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", default=None, help="frozen release pipeline.json (default: active/current)")
    ap.add_argument("--butler-repo", default="dp2_prep")
    ap.add_argument("--collection", required=True, help="Butler collection (the diffim DRP run)")
    ap.add_argument("--night", required=True, help="day_obs, e.g. 20250718")
    ap.add_argument("--tracts", required=True, help="tract list/ranges, e.g. 8489 or 8487-8493")
    ap.add_argument("--out", required=True, help="output run dir")
    ap.add_argument("--op-point", default=None, help="override the link op-point JSON")
    ap.add_argument("--discovery", action="store_true",
                    help="use the discovery op (link_op_point.json, mfsnr>=10) instead of the alert op")
    ap.add_argument("--obscode", default="I11")
    ap.add_argument("--mask-workers", type=int, default=64)
    ap.add_argument("--lsst-setup",
                    default="source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; "
                            "setup lsst_distrib")
    ap.add_argument("--force", action="store_true", help="rebuild manifest/known even if present")
    ap.add_argument("--dry-run", action="store_true", help="print the chain + run preflight; no compute")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
