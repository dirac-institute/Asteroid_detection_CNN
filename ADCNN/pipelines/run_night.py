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

# THE OPERATING POINT IS FIXED AND LIVES IN op_2v_stream_1k.json (score_min 0.70, chi2_2v_max 10).
# It is NOT adapted per night. A predecessor picked the score floor from detection density and then
# relinked when the budget came up short; both are deliberately gone (user decision 2026-08-14) --
# a night thinner than full cadence delivers fewer than 1,000 alerts and that is the accepted
# product, not a fault. See op_2v_stream_1k.json:_op_FIXED for the calibration and its measured cost.
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


def _ok(p, min_bytes=1):
    """Exists AND is non-empty. A bare .exists() lets a half-written or 0-byte file become THE input.

    Demonstrated by the audit: a planted 0-byte dets_merged.csv was passed to link_2visit, pixel_vet
    and alert_cutouts as the night's catalogue, and a 0-byte bright_refcat.parquet silently disables
    the bright-star proximity veto -- the primary ring lever, which flags ~56% of the raw product.
    Every shell driver already used `[ -s ]`; this brings run_night to the same discipline.
    """
    try:
        return p.exists() and p.stat().st_size >= min_bytes
    except OSError:
        return False


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
    # THE STREAM OP DEFAULTS TO THE FULL-CADENCE ONE, FOR EVERY NIGHT.
    #
    # The two stream ops differ in what they control, and only one of them is a safety property:
    #   score_min  -> TRACTABILITY. Seeding cost goes as detection DENSITY SQUARED. The calibration
    #                 (op_2v_stream_fullcadence._calibration) measured 0.70 cutting a pilot pointing
    #                 group from 68,878 to 9,155 linkable dets -- ~57x the seeding cost at 0.50 --
    #                 and the (since-removed) 0.50 op run unchanged on 20260629 DID NOT FINISH ONE
    #                 POINTING GROUP IN 42 MINUTES, extrapolating to several hundred thousand alerts.
    #   chi2       -> VOLUME. It filters AFTER the orbit solve, so loosening it is nearly free. Both
    #                 ops already use chi2 <= 30, so they do not differ on the volume knob at all.
    # And BOTH op files state that the nightly count is finally set by the top --stream-top-n RANK
    # cut, not by these gates. So choosing the tighter score floor costs stream size that the rank
    # cut and the downstream 1k op were going to impose anyway, while buying a link that finishes.
    #
    # It used to be selected per night by PANEL COUNT >= 3000, in a launcher that has since been
    # deleted -- so every caller that forgot the flag silently got the intractable op. That is not a
    # theoretical risk: it happened during the 2026-08-13 regeneration and inflated 20260711 from 837
    # alerts to 25,439 before it was caught. Panel count is also a poor proxy for the quantity that
    # actually drives the cost: MEASURED linkable dets/visit were 6,445 (20260630, 1,869 panels),
    # 11,646 (20260711, 3,731) and 3,310 (20260712, 6,370) -- 20260712 has 3.4x the panels of
    # 20260630 and HALF the density, so the threshold would have mis-ranked them.
    # Default to the op that is tractable on the densest night; --stream-op-point still overrides.
    stream_op = a.stream_op_point or str(REPO / "ADCNN/pipelines/heliolinc/op_2v_stream_fullcadence.json")
    print(f"      stream op-point: {os.path.basename(stream_op)}"
          f"{' (explicit --stream-op-point)' if a.stream_op_point else ' (default: tractable on any cadence)'}")

    out = Path(a.out) if a.out else OUTPUTS / "runs" / f"run_night_{a.night}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"

    # Idempotent re-entry. A night is done only when night_status finds every artifact present AND
    # mutually consistent (not merely "the output file exists", which cannot tell a truncated link
    # or a stale image set from a finished one). Skip a verified-complete night; otherwise proceed
    # and let each stage's own reuse guard skip the parts that are already good.
    from ADCNN.pipelines.night_status import status as _night_status, mark_complete as _mark_complete
    if not a.force:
        _s = _night_status(str(out))
        if _s["complete"]:
            print(f"[run_night] {a.night} already COMPLETE (all artifacts consistent); "
                  f"nothing to do. --force to rebuild."); return
        if _s["first_missing"]:
            print(f"[run_night] {a.night} incomplete -> resuming from '{_s['first_missing']}' "
                  f"({_s['detail']})")

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
    bright_refcat = out / "bright_refcat.parquet"
    stack_dets = out / "stack_dets.csv"
    merged_dets = out / "dets_merged.csv"
    # THE catalogue every downstream stage reads: ADCNN + stack DIA sources, merged (see _dets()).

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
        # THE MASKED CATALOGUE IS ALSO A VALID REUSE POINT. adcnn_dets.csv is DELETED after masking
        # unless --keep-raw-dets, so a completed night keeps only adcnn_dets_masked.csv -- and this
        # guard, looking for the raw file alone, re-submitted the GPU job for a night whose own
        # status line said "detect: 1869/1869 panels (0.0% missing)". That is hours of GPU to
        # reproduce a file we already hold the downstream product of, and on a busy queue it fails
        # outright: measured, the ada partition rejected the submission and took the whole night's
        # regeneration with it (rc=1 before any of the stages that actually needed rebuilding).
        masked = out / "adcnn_dets_masked.csv"
        if masked.exists() and masked.stat().st_size > 0 and not a.force:
            print(f"      (adcnn_dets.csv gone but adcnn_dets_masked.csv is present "
                  f"({masked.stat().st_size/1e6:.0f} MB) -- detection already done; reuse)")
            return
        cmd = (f"cd {REPO} && RUN={out} sbatch --export=ALL,RUN --wait {HL/'sn_detect.slurm'}"
               f"   # GPU; per-panel .done resume; -> {out}/adcnn_dets.csv")
        _bash(cmd, a.dry_run)
        if not a.dry_run:
            _check_detect_coverage()

    def _check_detect_coverage(source_csv=None):
        """Fail loud if detection silently skipped panels.

        A per-GPU shard that dies -- e.g. `CUDA error: uncorrectable ECC error`, which hit THREE
        different ada nodes in one night -- does NOT fail the slurm job. Its panels are simply
        absent, the job reports COMPLETED, and the night proceeds with a quarter of the sky
        missing and nothing in the exit status to say so. Only the absent progress heartbeats
        gave it away. The .done files are the authoritative record of panels actually processed
        (a panel with zero detections never appears in the dets CSV), so compare those against
        the manifest and write the residual manifest so a top-up run is one sbatch away."""
        import csv as _csv
        want = set()
        with open(manifest) as fh:
            for r in _csv.DictReader(fh):
                want.add((int(r["visit"]), int(r["detector"])))
        # Panels actually PROCESSED. The .done markers are authoritative (a panel with zero
        # detections is processed but contributes no rows) -- but detect_night DELETES them when it
        # assembles the shards into adcnn_dets.csv, so on a completed night they are gone and we
        # must fall back to the panels present in the dets catalogue. That fallback under-counts by
        # the genuinely-empty panels, hence the tolerance below rather than demanding 100%.
        got, src = set(), "none"
        dn_files = [] if source_csv else (
            list(out.glob("_shard_adcnn_dets_*.csv.done")) + list(out.glob("adcnn_dets.csv.done")))
        if dn_files:
            for dn in dn_files:
                for line in open(dn):
                    p = line.strip().split(",")
                    if len(p) == 2:
                        got.add((int(p[0]), int(p[1])))
            src = f"{len(dn_files)} .done marker file(s)"
        else:
            dets = Path(source_csv) if source_csv else (out / "adcnn_dets.csv")
            if dets.exists():
                import csv as _c2
                with open(dets) as fh:
                    for r in _c2.DictReader(fh):
                        got.add((int(r["visit"]), int(r["detector"])))
                src = "adcnn_dets.csv (markers already cleaned up)"
        missing = want - got
        frac = len(missing) / max(len(want), 1)
        if frac <= a.detect_miss_tol:
            print(f"      detect coverage OK: {len(got)}/{len(want)} panels via {src} "
                  f"({len(missing)} absent, {100*frac:.1f}% <= {100*a.detect_miss_tol:.0f}% tol)")
            return
        resid = out / "manifest_residual.csv"
        with open(manifest) as fh, open(resid, "w", newline="") as fo:
            rd = _csv.DictReader(fh)
            w = _csv.DictWriter(fo, fieldnames=rd.fieldnames)
            w.writeheader()
            for r in rd:
                if (int(r["visit"]), int(r["detector"])) in missing:
                    w.writerow(r)
        raise IntegrityError(
            f"detection covered {len(got)}/{len(want)} panels via {src} -- {len(missing)} MISSING "
            f"({100*frac:.1f}%). One shard per 4 GPUs is 25%: check the detect log for 'ECC' or a "
            f"[gpuN] with no heartbeats. Residual manifest ({len(missing)} panels) -> {resid}\n"
            f"  top up:  ADCNN_REPO=$PWD RUN=<fresh dir holding that manifest> "
            f"sbatch --exclude=<bad nodes> {HL/'sn_detect.slurm'}\n"
            f"  then merge its adcnn_dets.csv into this night's and re-run.")

    def s_known():
        kn = out / "known.csv"
        if _ok(kn, min_bytes=0) and not a.force:
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
        # Reuse like every other stage. This is REQUIRED now that the raw dets are dropped once
        # masking succeeds: without it a second run of the chain re-invokes mask_flags against a
        # file its own predecessor deleted, so re-running a completed night fails instead of
        # resuming at the linking stage.
        msk = out / "adcnn_dets_masked.csv"
        if msk.exists() and msk.stat().st_size > 0 and not a.force:
            print("      (adcnn_dets_masked.csv exists; reuse)")
            # Re-verify coverage on the REUSE path too. s_detect's guard only runs when detection
            # actually runs; a night whose detection was cut short by a dead shard, then re-entered
            # later, would skip detect AND mask on the stale files and link an incomplete night
            # silently. That happened to 20260705 (25% missing) and 20260706 (50%): both reported
            # rc=0 and produced alerts from half a night.
            _check_detect_coverage(source_csv=msk)
            return
        cmd = (f"python -m ADCNN.pipelines.heliolinc.mask_flags --dets {out}/adcnn_dets.csv "
               f"--manifest {manifest} --out {out}/adcnn_dets_masked.csv --workers {a.mask_workers}")
        _bash(cmd, a.dry_run)
        # adcnn_dets_masked.csv is a STRICT superset of adcnn_dets.csv (identical rows, +21 mask
        # columns) and is what every downstream stage reads, so keeping both stores the night's
        # detections twice (~100 MB/night). Drop the raw file only after verifying the row counts
        # match -- and drop its .done marker with it, or a later re-run would resume against a
        # catalog that no longer exists and silently produce nothing.
        # Best-effort: this is a disk saving, never a reason to fail a night that has already been
        # detected and masked. (It DID fail one: an unguarded unlink() raced a concurrent chain and
        # took down the run after 57 min of GPU.) Everything here is wrapped and missing_ok.
        raw, msk = out / "adcnn_dets.csv", out / "adcnn_dets_masked.csv"
        if a.keep_raw_dets or a.dry_run or not (raw.exists() and msk.exists()):
            return
        try:
            nr = sum(1 for _ in open(raw)); nm = sum(1 for _ in open(msk))
            if nr != nm:
                print(f"      WARN raw/masked row mismatch ({nr} vs {nm}) -- keeping raw dets")
                return
            raw.unlink(missing_ok=True)
            (out / "adcnn_dets.csv.done").unlink(missing_ok=True)
            print(f"      raw dets removed (superseded by masked, {nm - 1} rows; --keep-raw-dets to retain)")
        except OSError as e:
            print(f"      WARN could not remove raw dets ({e}) -- harmless, continuing")

    def s_static():
        # bright-static template-footprint catalog (DRP coadd object tables). A night with NO DRP
        # coverage fail-louds in the builder => static veto OFF fail-safe (link without the catalog).
        if a.no_static_veto:
            print("      (--no-static-veto: skipped)"); return
        if _ok(static_catalog) and not a.force:
            print("      (static_catalog.parquet exists; reuse)"); return
        # Prune at BUILD time to the magnitudes the veto can ever use. The veto matches statics
        # brighter than --static-mag-max (20.0); on 20260630 that is 6.2% of the 9.5M-row coadd
        # catalog, so writing it whole stores ~210 MB/night of rows nothing will ever read. The
        # default keeps a magnitude of margin so the veto cut can be raised without a rebuild.
        cmd = (f"bash -c '{lsst}; cd {REPO}; python -m ADCNN.linking.build_static_catalog "
               f"--dets {_dets()} --out {static_catalog} "
               f"--mag-max {a.static_catalog_mag_max}'")
        try:
            _bash(cmd, a.dry_run)
        except subprocess.CalledProcessError:
            print("      WARN: static-catalog build failed (no DRP coverage for these tracts?) -- "
                  "linking WITHOUT the static veto (documented fail-safe).")

    def s_refcat():
        """ALL-SKY bright-star refcat -> the product's bright-star PROXIMITY veto (the primary ring
        lever). Unlike the coadd static catalog this is all-sky, so it works on nights with no DRP
        coverage -- which is most of them. DEPTH IS THE POINT: the residual dipole/RINGS in the
        delivered product sit on mag 19-21 stars, so a shallow (mag<19) refcat catches 0% of them.
        Measured 20260706 with an offset null: mag<21 @2.5" removes 55.8% of the product at 2.7%
        cost to real movers (~20:1). Best-effort: a failure just leaves the veto off (fail-safe)."""
        if a.no_refcat:
            print("      (--no-refcat: skipped)"); return
        if _ok(bright_refcat) and not a.force:
            print("      (bright_refcat.parquet exists; reuse)"); return
        cmd = (f"bash -c '{lsst}; cd {REPO}; BUTLER_REPO=embargo python -m ADCNN.linking.build_static_refcat "
               f"--dets {_dets()} --out {bright_refcat} "
               f"--refcat the_monster_20250219 --mag-max {a.refcat_mag_max}'")
        try:
            _bash(cmd, a.dry_run)
        except subprocess.CalledProcessError:
            print("      WARN: refcat build failed -- product runs WITHOUT the bright-star proximity "
                  "veto (rings will survive; documented fail-safe).")

    def _dets():
        """The detection catalogue downstream stages consume: the ADCNN+stack merge when it
        exists, else ADCNN alone (fail-safe)."""
        return merged_dets if _ok(merged_dets) else (out / 'adcnn_dets_masked.csv')

    def s_stack_merge():
        """ALWAYS-ON union of the stack's DIA sources with ADCNN's detections, BOTH sides ring-cleaned.

        Numbers from the 2026-08-12/13 one-harness measurement (3,857 injected movers, trail-segment
        matching), which SUPERSEDE the older figures that stood here. DETECTION both-epoch: ADCNN
        49.8%, stack 37.0%, union 53.8% -- a tie on short trails (44.5 vs 44.2 at 0-8px), diverging
        with length to 49.0 vs 23.1 at 44-60px; the stack contributes 154 movers ADCNN misses.
        END-TO-END at the 1k budget the merge is NEUTRAL (9.26% vs 9.28%; flagship 2.06% vs 2.16%):
        the stack's unique detections carry no usable trail geometry -- DPDD trailLength is NaN on
        ~31% and near-PSF elsewhere on that population -- so the trail-based linker cannot pair them,
        and re-measuring them with OUR estimator provably cannot fix that (the only gate that stops it
        saturating is our own seg+stage-2, and what passes that gate IS what ADCNN already found).
        The merge stays on because it is measured harmless at the budget, keeps the short-trail/bright
        complementarity, and preserves the ceiling for a future geometry fix (running
        lsst.meas.extensions.trailedSources ourselves, which the DRP does not).
        merge_dets cleans BOTH sides with the deep refcat before the union -- the stack side measured
        61.2% ring-positioned vs 10.4% chance and its own dipole columns are inert -- and dedups
        against the FULL pre-cleaning catalogue so deleted rings' stack copies cannot re-enter.
        Every row keeps `src`; unmeasured fields stay NaN, never defaulted to clean-looking values.
        Best-effort: if the stack ingest fails the merge falls back to ADCNN-only and says so."""
        if a.no_stack_merge:
            print("      (--no-stack-merge: ADCNN-only, forfeits the stack-only movers)"); return
        # STALENESS, not just existence. dets_merged.csv is derived from adcnn_dets_masked.csv; the
        # Aug-11 mf_snr repair reached the ADCNN catalogue and NOT the merge, and nothing noticed
        # for four days because this guard reused on existence alone.
        if _ok(merged_dets) and not a.force:
            _src_m = (out / "adcnn_dets_masked.csv")
            if _src_m.exists() and _src_m.stat().st_mtime > merged_dets.stat().st_mtime:
                print("      dets_merged.csv is OLDER than adcnn_dets_masked.csv -- the ADCNN "
                      "catalogue changed after the merge (e.g. a post-hoc repair). REBUILDING.")
            else:
                print("      (dets_merged.csv exists and is newer than its inputs; reuse)"); return
        if not a.collection:
            print("      WARN: no --collection, cannot ingest DIA sources -- ADCNN-only"); return
        # THE PROMPT COLLECTIONS LIVE IN `embargo`, NOT IN THE DEFAULT SCIENCE REPO. Passing
        # a.butler_repo (default dp2_prep) raised MissingCollectionError, the fail-safe below caught
        # it, and the night silently linked ADCNN-only -- so the entire stack merge, and the measured
        # 9.26% -> 9.75% delivered-completeness gain that comes with ring-cleaning both sides, was
        # absent from the product with only a WARN line to show for it. build_static_refcat two
        # stages earlier already hardcodes BUTLER_REPO=embargo for exactly this reason; this is the
        # same repo split, applied consistently. Override with --diasrc-butler-repo.
        _dia_repo = getattr(a, "diasrc_butler_repo", None) or (
            "embargo" if "/runs/prompt/" in (a.collection or "") else a.butler_repo)
        try:
            _bash(f"bash -c '{lsst}; cd {REPO}; BUTLER_REPO={_dia_repo} "
                  f"python -m ADCNN.linking.ingest_diasource --butler-repo {_dia_repo} "
                  f"--collection {shlex.quote(a.collection)} --out {stack_dets}'", a.dry_run)
        except subprocess.CalledProcessError:
            # NOT silent. An ADCNN-only night is a materially different product; say what was lost.
            print(f"      WARN: DIA-source ingest FAILED against repo '{_dia_repo}' -- this night "
                  f"links ADCNN-ONLY, forfeiting the stack merge (measured worth +0.5 pts delivered "
                  f"completeness at the 1k budget). Documented fail-safe, but check the repo/"
                  f"collection pair before accepting the product.")
            return
        # pass the deep refcat so the ring-drop still happens on catalogues predating is_dipole
        _bash(f"python -m ADCNN.linking.merge_dets --adcnn {out}/adcnn_dets_masked.csv "
              f"--stack {stack_dets} --out {merged_dets} --refcat {bright_refcat} "
              f"--refcat-mag-max {a.refcat_mag_max}", a.dry_run)

    def s_link():
        floor = f" --score-candidate-min {a.candidate_floor}" if a.candidate_floor else ""
        static = f" --static-catalog {static_catalog}" if _ok(static_catalog) and not a.no_static_veto else ""
        report = "" if a.no_report else " --report"
        cmd = (f"python -m ADCNN.linking.link_2visit --dets {_dets()} "
               f"--known {out}/known.csv --out {out}/tracks.csv --op-point {op_point} "
               f"--npt 2 --min-epochs 2 --seed-2v chord{floor}{static} --train-veto{report} "
               f"--alerts-out {out}/alerts.jsonl")
        _bash(cmd, a.dry_run)

    def s_vet():
        # pixel stationarity vet (INVESTIGATION_2V_CONFIDENCE.md sections 7/8): annotates pixelVet +
        # `confident`, demotes flagged/killed alerts in the ranking, never drops. Needs the dets
        # catalog's fits_path for panel lookup -- pixel_vet itself no-ops (pass-through) without it.
        cmd = (f"python -m ADCNN.linking.pixel_vet --alerts {out}/alerts.jsonl "
               f"--dets {_dets()} --in-place")
        _bash(cmd, a.dry_run)

    def s_stream():
        """Low-threshold ALERT STREAM: a SECOND, additive linking pass at the stream op-point,
        ranked and rendered to per-alert pair images for nightly visual QA.

        Deliberately separate from the frozen alert product above: that one stays byte-for-byte
        the validated science output, this one trades purity for volume (~10k/night) so the night
        can be eyeballed for systematics and the cut chosen later, downstream, on the ranked list.
        Images go through the panel-ordered cutout cache (ADCNN.qa.alert_cutouts), so pixel I/O
        is O(panels) not O(alerts) -- the only way 10k alerts is affordable."""
        if a.no_stream:
            print("      (--no-stream: skipped)"); return
        sd = out / "stream"
        sd.mkdir(parents=True, exist_ok=True)
        static = f" --static-catalog {static_catalog}" if _ok(static_catalog) and not a.no_static_veto else ""
        # per-substage resume: a re-run after an interrupted night must not redo the ~45 min link
        # or the S3 cutout pass. --force redoes everything (same convention as manifest/known).
        ap = sd / "alerts.jsonl"
        # Re-link when the alerts file is MISSING OR EMPTY, not merely absent. A link that died
        # after writing tracks.csv but before alerts.jsonl (20260705/06) left a tracks file and no
        # alerts, and an `exists`-only guard on a later stage would then render from nothing. The
        # cutout cache and sheets/pairs are downstream of this, so a truncated link must re-run.
        from ADCNN.qa.filter_op import survivors_at as _surv, CHI2_GRID as _CG
        _op1k = json.load(open(REPO / "ADCNN/pipelines/heliolinc/op_2v_stream_1k.json"))
        _budget = _op1k.get("budget", 1000)

        if a.force or not (ap.exists() and ap.stat().st_size > 0):
            # claim-order + rank-by are what make a LOW-THRESHOLD stream trustworthy: at 11k alerts
            # the seeding-order claim loses real pairs to spurious ones. Which PRIORITY to claim by
            # was settled against INJECTED TRUTH on the calibration night, not against the 12 frozen
            # science alerts -- that proxy preferred chi2 (11/12 vs 9/12 preserved) but truth prefers
            # P(real): 987 vs 936 real pairs recovered of 5,226 pairable, at higher purity too
            # (8.39% vs 8.02%). The proxy is not truth; none of those 12 has an MPC match.
            #
            # ONE FIXED OPERATING POINT (user decision 2026-08-14). No --score-min override: the
            # link uses the op file's own score_min (0.70) on EVERY night, dense or thin. The floor
            # is not predicted from density and not relinked -- a night that cannot fill the budget
            # simply delivers fewer than 1,000, which is the accepted behaviour, not a fault.
            # Rationale and the measured cost live in op_2v_stream_1k.json:_op_FIXED.
            _bash(f"python -m ADCNN.linking.link_2visit --dets {_dets()} "
                  f"--known {out}/known.csv --out {sd}/tracks.csv --op-point {stream_op} "
                  f"--npt 2 --min-epochs 2 --seed-2v chord{static} --train-veto "
                  f"--claim-order preal --rank-by chi2 "
                  f"--alerts-out {sd}/alerts.jsonl", a.dry_run)
        else:
            print(f"      (stream alerts.jsonl exists -- reusing; --force to relink)")
        # REPORT the fill, never act on it. Under a fixed op the number cannot change what we do,
        # but it is the one line that says whether this night made budget and why -- and a night
        # that is short is short because of DETECTIONS, not because a knob is mistuned.
        if not a.dry_run and ap.exists():
            _have = _surv(str(ap), _op1k, float(_op1k["chi2_2v_max"]))
            print(f"      stream fill: {_have:,} alerts pass the fixed op (chi2<="
                  f"{_op1k['chi2_2v_max']:g}, score_min {_op1k['score_min']:g}) against a budget of "
                  f"{_budget:,}"
                  + ("" if _have >= _budget else
                     f" -- UNDER BUDGET; this night is detection-limited. Expected on a thin or "
                     f"field-sparse night and accepted by the fixed operating point."), flush=True)
        # RANK BEFORE CUTTING. rerank_alerts rewrites alerts.jsonl IN PLACE, permuting it, and the
        # cutout cache is keyed by alert POSITION -- so cutting first and re-ranking second addresses
        # every cached stamp to a different alert than the one whose caption it is rendered under.
        # This shipped: on 20260710 only 13 of 18,009 positions survived the permutation, and the
        # delivered sheet_0000.png is reproduced to 0.13% of pixels by re-rendering with a
        # linker-order cache while differing from the correct render in 76.27% of pixels. Six of the
        # nine delivered nights ran in that order. Ranking first costs nothing (it is a pure sort of
        # a JSONL file, no pixels) and makes the cache correct by construction; the identity guard in
        # alert_sheets then has nothing left to catch.
        if not a.no_rerank:
            _bash(f"python -m ADCNN.qa.rerank_alerts --alerts {sd}/alerts.jsonl", a.dry_run)
        if a.force or not (sd / "cutouts.npz").exists():
            _bash(f"python -m ADCNN.qa.alert_cutouts --alerts {sd}/alerts.jsonl "
                  f"--dets {_dets()} --out {sd}/cutouts.npz "
                  f"--stamp-px {a.stream_stamp_px} --workers {a.stream_workers} "
                  f"--limit {a.stream_top_n}", a.dry_run)
        else:
            print(f"      (stream cutouts.npz exists -- reusing; --force to re-cut)")
        # CONTACT SHEETS ARE NOT PRODUCED (user decision 2026-08-16). The per-alert pair images
        # below are the reviewed artifact; the sheets were a grid view of the same pixels costing
        # 2.4 GB across nine nights. ADCNN.qa.alert_sheets still exists and can be run by hand
        # against a rebuilt cutout cache if a bulk view is ever wanted.
        _bash(f"python -m ADCNN.qa.alert_pairs --alerts {sd}/alerts.jsonl "
              f"--cutouts {sd}/cutouts.npz --out-dir {sd}/pairs "
              f"--top-n {a.stream_pairs_top_n}", a.dry_run)
        _bash(f"python -m ADCNN.qa.stream_summary --alerts {sd}/alerts.jsonl "
              f"--out {sd}/stream_summary.json"
              + (f" --static-catalog {static_catalog}" if _ok(static_catalog) else ""), a.dry_run)
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
    tm.stage("bright_refcat", s_refcat)
    tm.stage("stack_merge", s_stack_merge)
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

    # Sentinel: write .complete ONLY if every artifact now verifies, so re-entry is a cheap stat
    # and a half-finished night is never mistaken for done. If it does not verify, say what is
    # still missing rather than claiming success -- the campaign driver reads this to decide.
    if not a.dry_run and not a.no_stream:
        fin = _night_status(str(out))
        if fin["complete"]:
            _mark_complete(str(out))
            print(f"[run_night] {a.night} COMPLETE -> {out}/ (.complete written)")
        else:
            print(f"[run_night] {a.night} finished stages but NOT complete: "
                  f"still needs '{fin['first_missing']}' ({fin['detail']}). "
                  f"Re-run to resume; not marking complete.")
    else:
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
    ap.add_argument("--diasrc-butler-repo", default=None,
                    help="Butler repo holding the DIA-source collection (default: `embargo` for a "
                         "/runs/prompt/ collection, else --butler-repo). The prompt products are not "
                         "in the science repo.")
    ap.add_argument("--no-stack-merge", action="store_true",
                    help="do NOT merge the stack DIA sources; ADCNN-only. Measured cost: forfeits "
                         "the ~3.4%% of real movers only the stack finds (~18%% relative recall)")
    ap.add_argument("--no-refcat", action="store_true",
                    help="skip building the all-sky bright-star refcat; the product then runs WITHOUT "
                         "the bright-star proximity veto (the primary ring lever) -- rings will survive")
    ap.add_argument("--refcat-mag-max", type=float, default=21.0,
                    help="depth of the all-sky refcat for the proximity veto. 21 (the_monster is "
                         "complete to G~21) is REQUIRED: the residual product rings sit on mag 19-21 "
                         "stars, so the old mag<19 catalog caught 0%% of them (measured 20260706)")
    ap.add_argument("--no-static-veto", action="store_true",
                    help="skip the bright-static template-footprint veto stage")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the in-run QA report package (overlays + stamps + ALERT_REPORT.md)")
    ap.add_argument("--no-crossmatch", action="store_true", help="skip the MPC conesearch crossmatch")
    ap.add_argument("--no-stream", action="store_true",
                    help="skip the low-threshold alert stream (the ~10k/night ranked QA product)")
    ap.add_argument("--stream-op-point", default=None,
                    help="stream linking op-point JSON. Default op_2v_stream_fullcadence.json, which "
                         "is tractable at any cadence (score_min 0.70; seeding cost goes as density^2 "
                         "and a 0.50 floor does not finish a dense night -- the 2026-08-13 scan also "
                         "measured 0.50/0.60/0.70 completeness-IDENTICAL at the 1k budget, so a lower "
                         "floor buys no recall; the legacy 0.50 op file was removed for exactly that "
                         "reason). The score floor is FIXED at the op file's value on every night and "
                         "is never adapted; see op_2v_stream_1k.json:_op_FIXED.")
    ap.add_argument("--stream-top-n", type=int, default=20000,
                    help="how many top-ranked stream alerts get cutouts + pair images (linking keeps ALL; "
                         "this only bounds the image render)")
    ap.add_argument("--stream-per-sheet", type=int, default=48)
    ap.add_argument("--stream-pairs-top-n", type=int, default=20000,
                    help="alerts that get their OWN pair+wide-view image file (rank order). The "
                         "default matches the nightly alert budget, i.e. one image per alert -- "
                         "~2 GB/night at ~200 kB each. Lower it only to save disk: the contact "
                         "sheets image every alert regardless, so nothing becomes unviewable")
    ap.add_argument("--detect-miss-tol", type=float, default=0.05,
                    help="fraction of manifest panels allowed to be absent after detection before "
                         "failing. Some absence is normal when coverage is inferred from the dets "
                         "catalogue (panels with zero detections leave no rows); a dead GPU shard "
                         "loses 1/n_gpus = 25%%, far above this")
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
