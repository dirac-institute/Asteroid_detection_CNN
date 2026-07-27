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

Stages (the productized night recipe, mirroring sn_run.slurm + the 2026-07 embargo campaign):
build-manifest (LSST stack env; --tracts or --visits) -> detect (GPU sbatch, resumable) ->
build-known (LSST stack env; --no-known writes a header-only catalog for post-hoc SkyBoT labelling)
-> mask-flags -> static-catalog (DRP object tables; NO-coverage night => static veto OFF fail-safe)
-> link-2visit (frozen alert op + candidate floor 0.5 + static/train/stationarity vetoes, FLAG never
drop, --report QA package) -> pixel-vet -> mpc-crossmatch (network; failure WARNs, never fails).

Speed contract: the linker's cheap prefilters (chord seeding + partial-chi2 pre-gates) only PRUNE
candidates; the final orbit-fit chi2 and physical_check gates stay exact -- final measurements and
gates are reproducible, approximation is for pruning only.

    # dry-run one night (prints the exact chain, runs the integrity preflight):
    ./adcnn night --butler-repo embargo --collection LSSTCam/runs/prompt/.../ApPipe/... \
        --night 20260628 --visits 2026062800001-2026062800400 --dry-run
    # DRP re-run of a tract night:
    ./adcnn night --butler-repo main --collection LSSTCam/runs/DRP/.../DM-51933 \
        --night 20250718 --tracts 8489
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

from ADCNN.config import OUTPUTS, load_pipeline, REPO

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
    stream_op = a.stream_op_point or str(REPO / "ADCNN/pipelines/heliolinc/op_2v_stream.json")
    out = Path(a.out) if a.out else OUTPUTS / "runs" / f"run_night_{a.night}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    print(f"[run_night] pipeline={pipe.name} provenance={pipe.provenance}")
    print(f"  night={a.night} tracts={a.tracts} visits={a.visits} collection={a.collection}")
    print(f"  out={out}")
    print(f"  op-point={op_point} ({'DISCOVERY' if op_point == str(DISCOVERY_OP) else 'ALERT (default)'})")

    preflight(pipe, op_point, discovery=a.discovery)

    # the sbatch detect stage resolves models via the ACTIVE pipeline config -- pin it to the
    # release we just preflighted so --export=ALL propagates the same choice to the GPU job.
    os.environ["ADCNN_PIPELINE"] = str(pipe.source)
    os.environ.setdefault("ADCNN_REPO", str(REPO))

    lsst = a.lsst_setup
    tm = _Timer()
    static_catalog = out / "static_catalog.parquet"

    def s_manifest():
        if manifest.exists() and manifest.stat().st_size > 0 and not a.force:
            print("      (manifest exists; reuse)"); return
        sel = (f"--visits {a.visits}" if a.visits else
               f"--tracts {a.tracts} --day-start {a.night} --day-end {int(a.night)+1}")
        cmd = (f"bash -c '{lsst}; cd {REPO}; python -m ADCNN.pipelines.heliolinc.build_manifest "
               f"{sel} --butler-repo {shlex.quote(a.butler_repo)} "
               f"--collection {shlex.quote(a.collection)} --out {manifest}'")
        _bash(cmd, a.dry_run)

    def s_detect():
        dets = out / "adcnn_dets.csv"
        if dets.exists() and dets.stat().st_size > 0 and not a.force:
            print("      (adcnn_dets.csv exists; reuse)"); return
        cmd = (f"cd {REPO} && RUN={out} sbatch --export=ALL,RUN --wait {HL/'sn_detect.slurm'}"
               f"   # GPU; per-panel .done resume; -> {out}/adcnn_dets.csv")
        _bash(cmd, a.dry_run)

    def s_known():
        kn = out / "known.csv"
        if kn.exists() and not a.force:
            print("      (known.csv exists; reuse)"); return
        if a.no_known:
            # embargo/prompt recipe: header-only catalog; label post-hoc (SkyBoT / mpc-crossmatch).
            print("      (--no-known: writing header-only known.csv)")
            if not a.dry_run:
                kn.write_text("ObjID,ra,dec,mjd\n")
            return
        cmd = (f"bash -c '{lsst}; cd {REPO}; python -m ADCNN.pipelines.heliolinc.build_known_catalog "
               f"--manifest {manifest} --butler-repo {shlex.quote(a.butler_repo)} "
               f"--collection {shlex.quote(a.collection)} --out {kn}'")
        _bash(cmd, a.dry_run)

    def s_mask():
        cmd = (f"python -m ADCNN.pipelines.heliolinc.mask_flags --dets {out}/adcnn_dets.csv "
               f"--manifest {manifest} --out {out}/adcnn_dets_masked.csv --workers {a.mask_workers}")
        _bash(cmd, a.dry_run)
        # adcnn_dets_masked.csv is a STRICT superset of adcnn_dets.csv (identical rows, +21 mask
        # columns) and is what every downstream stage reads, so keeping both stores the night's
        # detections twice (~100 MB/night). Drop the raw file only after verifying the row counts
        # match -- and drop its .done marker with it, or a later re-run would resume against a
        # catalog that no longer exists and silently produce nothing.
        raw, msk = out / "adcnn_dets.csv", out / "adcnn_dets_masked.csv"
        if a.keep_raw_dets or a.dry_run or not (raw.exists() and msk.exists()):
            return
        nr = sum(1 for _ in open(raw)); nm = sum(1 for _ in open(msk))
        if nr == nm:
            raw.unlink()
            (out / "adcnn_dets.csv.done").unlink(missing_ok=True)
            print(f"      raw dets removed (superseded by masked, {nm - 1} rows; --keep-raw-dets to retain)")
        else:
            print(f"      WARN raw/masked row mismatch ({nr} vs {nm}) -- keeping raw dets")

    def s_static():
        # bright-static template-footprint catalog (DRP coadd object tables). A night with NO DRP
        # coverage fail-louds in the builder => static veto OFF fail-safe (link without the catalog).
        if a.no_static_veto:
            print("      (--no-static-veto: skipped)"); return
        if static_catalog.exists() and not a.force:
            print("      (static_catalog.parquet exists; reuse)"); return
        # Prune at BUILD time to the magnitudes the veto can ever use. The veto matches statics
        # brighter than --static-mag-max (20.0); on 20260630 that is 6.2% of the 9.5M-row coadd
        # catalog, so writing it whole stores ~210 MB/night of rows nothing will ever read. The
        # default keeps a magnitude of margin so the veto cut can be raised without a rebuild.
        cmd = (f"bash -c '{lsst}; cd {REPO}; python -m ADCNN.linking.build_static_catalog "
               f"--dets {out}/adcnn_dets_masked.csv --out {static_catalog} "
               f"--mag-max {a.static_catalog_mag_max}'")
        try:
            _bash(cmd, a.dry_run)
        except subprocess.CalledProcessError:
            print("      WARN: static-catalog build failed (no DRP coverage for these tracts?) -- "
                  "linking WITHOUT the static veto (documented fail-safe).")

    def s_link():
        floor = f" --score-candidate-min {a.candidate_floor}" if a.candidate_floor else ""
        static = f" --static-catalog {static_catalog}" if static_catalog.exists() and not a.no_static_veto else ""
        report = "" if a.no_report else " --report"
        cmd = (f"python -m ADCNN.linking.link_2visit --dets {out}/adcnn_dets_masked.csv "
               f"--known {out}/known.csv --out {out}/tracks.csv --op-point {op_point} "
               f"--npt 2 --min-epochs 2 --seed-2v chord{floor}{static} --train-veto{report} "
               f"--alerts-out {out}/alerts.jsonl")
        _bash(cmd, a.dry_run)

    def s_vet():
        # pixel stationarity vet (INVESTIGATION_2V_CONFIDENCE.md sections 7/8): annotates pixelVet +
        # `confident`, demotes flagged/killed alerts in the ranking, never drops. Needs the dets
        # catalog's fits_path for panel lookup -- pixel_vet itself no-ops (pass-through) without it.
        cmd = (f"python -m ADCNN.linking.pixel_vet --alerts {out}/alerts.jsonl "
               f"--dets {out}/adcnn_dets_masked.csv --in-place")
        _bash(cmd, a.dry_run)

    def s_stream():
        """Low-threshold ALERT STREAM: a SECOND, additive linking pass at the stream op-point,
        ranked and rendered to browsable contact sheets for nightly visual QA.

        Deliberately separate from the frozen alert product above: that one stays byte-for-byte
        the validated science output, this one trades purity for volume (~10k/night) so the night
        can be eyeballed for systematics and the cut chosen later, downstream, on the ranked list.
        Images go through the panel-ordered cutout cache (ADCNN.qa.alert_cutouts), so pixel I/O
        is O(panels) not O(alerts) -- the only way 10k alerts is affordable."""
        if a.no_stream:
            print("      (--no-stream: skipped)"); return
        sd = out / "stream"
        sd.mkdir(parents=True, exist_ok=True)
        static = f" --static-catalog {static_catalog}" if static_catalog.exists() and not a.no_static_veto else ""
        # per-substage resume: a re-run after an interrupted night must not redo the ~45 min link
        # or the S3 cutout pass. --force redoes everything (same convention as manifest/known).
        if a.force or not (sd / "alerts.jsonl").exists():
            # claim-order=quality + rank-by=chi2 are what make a LOW-THRESHOLD stream trustworthy:
            # at 11k alerts the seeding-order claim loses validated alerts to spurious pairs, and
            # the CNN-score ordering buries the survivors near the bottom (both measured on 20260630).
            _bash(f"python -m ADCNN.linking.link_2visit --dets {out}/adcnn_dets_masked.csv "
                  f"--known {out}/known.csv --out {sd}/tracks.csv --op-point {stream_op} "
                  f"--npt 2 --min-epochs 2 --seed-2v chord{static} --train-veto "
                  f"--claim-order quality --rank-by chi2 "
                  f"--alerts-out {sd}/alerts.jsonl", a.dry_run)
        else:
            print(f"      (stream alerts.jsonl exists -- reusing; --force to relink)")
        if a.force or not (sd / "cutouts.npz").exists():
            _bash(f"python -m ADCNN.qa.alert_cutouts --alerts {sd}/alerts.jsonl "
                  f"--dets {out}/adcnn_dets_masked.csv --out {sd}/cutouts.npz "
                  f"--stamp-px {a.stream_stamp_px} --workers {a.stream_workers} "
                  f"--limit {a.stream_top_n}", a.dry_run)
        else:
            print(f"      (stream cutouts.npz exists -- reusing; --force to re-cut)")
        # rank by the CALIBRATED P(real) before rendering, so the images are produced in the order
        # a human should look at them (post-hoc: no re-link, no re-cut -- see ADCNN/qa/rerank_alerts.py)
        if not a.no_rerank:
            _bash(f"python -m ADCNN.qa.rerank_alerts --alerts {sd}/alerts.jsonl", a.dry_run)
        _bash(f"python -m ADCNN.qa.alert_sheets --alerts {sd}/alerts.jsonl "
              f"--cutouts {sd}/cutouts.npz --out-dir {sd}/sheets "
              f"--per-sheet {a.stream_per_sheet} --limit {a.stream_top_n}", a.dry_run)
        _bash(f"python -m ADCNN.qa.alert_pairs --alerts {sd}/alerts.jsonl "
              f"--cutouts {sd}/cutouts.npz --out-dir {sd}/pairs "
              f"--top-n {a.stream_pairs_top_n}", a.dry_run)
        _bash(f"python -m ADCNN.qa.stream_summary --alerts {sd}/alerts.jsonl "
              f"--out {sd}/stream_summary.json", a.dry_run)
        # The cutout cache is ~1.1 GB/night and is pure intermediate: it exists so re-ranking and
        # re-rendering cost no pixel IO. Once the images are written it is regenerable in ~25 min
        # from the dets catalog, so it is not worth keeping by default.
        cz = sd / "cutouts.npz"
        if not a.keep_cutouts and not a.dry_run and cz.exists():
            mb = cz.stat().st_size / 1e6
            cz.unlink()
            print(f"      cutout cache removed ({mb:.0f} MB, regenerable; --keep-cutouts to retain "
                  f"for fast re-ranking)")

    def s_mpc():
        # MPC conesearch crossmatch of the ranked alerts (network). Best-effort: WARN, never fail.
        if a.no_crossmatch:
            print("      (--no-crossmatch: skipped)"); return
        cmd = (f"python -m ADCNN.pipelines.heliolinc.mpc_crossmatch --alerts {out}/alerts.jsonl "
               f"--out {out}/mpc_matches.csv")
        try:
            _bash(cmd, a.dry_run)
        except subprocess.CalledProcessError:
            print("      WARN: mpc_crossmatch failed (network?) -- alerts stand, label later.")

    tm.stage("build_manifest", s_manifest)
    tm.stage("detect", s_detect)
    tm.stage("build_known", s_known)
    tm.stage("mask_flags", s_mask)
    tm.stage("static_catalog", s_static)
    tm.stage("link_2visit", s_link)
    tm.stage("pixel_vet", s_vet)
    tm.stage("mpc_crossmatch", s_mpc)
    tm.stage("alert_stream", s_stream)

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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", default=None, help="frozen release pipeline.json (default: active/current)")
    ap.add_argument("--butler-repo", default=os.environ.get("BUTLER_REPO", "main"),
                    help="Butler repo: 'main' (DRP diffims) or 'embargo' (prompt processing); "
                         "default $BUTLER_REPO or main")
    ap.add_argument("--collection", required=True, help="diffim collection (DRP run or prompt ApPipe)")
    ap.add_argument("--night", required=True, help="day_obs, e.g. 20250718")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--tracts", help="tract list/ranges, e.g. 8489 or 8487-8493 (DRP night)")
    sel.add_argument("--visits", help="visit id list/ranges (prompt/embargo night; from queryDatasets)")
    ap.add_argument("--out", default=None,
                    help="output run dir (default: <outputs>/runs/run_night_<night>)")
    ap.add_argument("--op-point", default=None, help="override the link op-point JSON")
    ap.add_argument("--discovery", action="store_true",
                    help="use the discovery op (link_op_point.json, mfsnr>=10) instead of the alert op")
    ap.add_argument("--candidate-floor", type=float, default=0.5,
                    help="two-tier CANDIDATE score floor for the linker (--score-candidate-min); "
                         "the shipped night product uses 0.5; 0 = single-floor op only")
    ap.add_argument("--no-known", action="store_true",
                    help="write a header-only known.csv (prompt/embargo recipe: label post-hoc)")
    ap.add_argument("--no-static-veto", action="store_true",
                    help="skip the bright-static template-footprint veto stage")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the in-run QA report package (overlays + stamps + ALERT_REPORT.md)")
    ap.add_argument("--no-crossmatch", action="store_true", help="skip the MPC conesearch crossmatch")
    ap.add_argument("--no-stream", action="store_true",
                    help="skip the low-threshold alert stream (the ~10k/night ranked QA product)")
    ap.add_argument("--stream-op-point", default=None,
                    help="stream linking op-point JSON (default: ADCNN/pipelines/heliolinc/op_2v_stream.json)")
    ap.add_argument("--stream-top-n", type=int, default=10000,
                    help="how many top-ranked stream alerts get cutouts + sheets (linking keeps ALL; "
                         "this only bounds the image render)")
    ap.add_argument("--stream-per-sheet", type=int, default=48)
    ap.add_argument("--stream-pairs-top-n", type=int, default=10000,
                    help="alerts that get their OWN pair+wide-view image file (rank order). The "
                         "default matches the nightly alert budget, i.e. one image per alert -- "
                         "~2 GB/night at ~200 kB each. Lower it only to save disk: the contact "
                         "sheets image every alert regardless, so nothing becomes unviewable")
    ap.add_argument("--keep-raw-dets", action="store_true",
                    help="keep adcnn_dets.csv after masking (default: drop it, the masked file is a "
                         "strict superset)")
    ap.add_argument("--keep-cutouts", action="store_true",
                    help="keep the ~1.1 GB stream cutout cache (default: drop it after rendering; "
                         "keep it if you plan to re-rank and re-render without re-reading pixels)")
    ap.add_argument("--static-catalog-mag-max", type=float, default=21.0,
                    help="magnitude cut applied when BUILDING the static catalog; the veto only uses "
                         "sources brighter than --static-mag-max (20), so this keeps 1 mag of margin")
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip the calibrated P(real) re-ranking of the stream")
    ap.add_argument("--stream-stamp-px", type=int, default=96, help="cutout size in px (96 = 19.2 arcsec)")
    ap.add_argument("--stream-workers", type=int, default=16)
    ap.add_argument("--obscode", default="I11")
    ap.add_argument("--mask-workers", type=int, default=64)
    ap.add_argument("--lsst-setup",
                    default="source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; "
                            "setup lsst_distrib")
    ap.add_argument("--force", action="store_true", help="rebuild manifest/known even if present")
    ap.add_argument("--dry-run", action="store_true", help="print the chain + run preflight; no compute")
    a = ap.parse_args(argv)
    run(a)


if __name__ == "__main__":
    main()
