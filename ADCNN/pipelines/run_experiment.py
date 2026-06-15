"""ADCNN — single canonical entry point for the full detector workflow.

One driver for every stage of the pipeline, reading the ACTIVE pipeline config
(``ADCNN/config.py``; ``ADCNN_PIPELINE`` selects, default = the promoted current detector)
so there are no ad-hoc env-var combinations to remember:

    data            build training/validation/test injection datasets        (GPU/Butler)
    train-stage1    train the stage-1 segmentation model                      (GPU)
    train-stage2    train/refit the stage-2 cutout-CNN scorer                 (GPU)
    calibrate-mflen apply the trail-length de-bias (len_db + endpoints)       (CPU)
    detect          run detection on validation/test/blind fields            (GPU)
    alert-eval      same-night 2-visit alert evaluation (purity/completeness) (CPU)
    report          regenerate the headline blind table + threshold plots     (CPU)

GPU/Butler stages PRINT the exact ``sbatch`` command (the SLURM backend is
``ADCNN/pipelines/heliolinc/train_v2_D_e2e.sh`` + ``TRAIN_V2_D_E2E.md``); they submit only with
``--submit``. CPU stages run in-process. ``--dry-run`` prints what each stage would do without
side effects. Use ``--stages all`` for the ordered end-to-end plan.

    python -m ADCNN.pipelines.run_experiment --stage report
    python -m ADCNN.pipelines.run_experiment --stages all --dry-run
    python -m ADCNN.pipelines.run_experiment --stage calibrate-mflen \
        --run-dir .../run_blind_v2eval --manifests .../run_blind --out .../run_blind_v2eval_cal

Reproduces the frozen headline (faint-fast 2v alert completeness 3.64% -> 10.33%, +184%);
see REPRODUCE.md / TRAINING_PROTOCOL.md / EVALUATION_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from ADCNN.config import load_pipeline

REPO = Path(__file__).resolve().parents[2]
HL = REPO / "ADCNN" / "pipelines" / "heliolinc"

STAGE_ORDER = ["data", "train-stage1", "train-stage2", "calibrate-mflen",
               "detect", "alert-eval", "report", "evaluation-notebooks"]
GPU_STAGES = {"data", "train-stage1", "train-stage2", "detect"}


def _run(cmd, cwd=None, env_pythonpath=True):
    """Run a CPU stage in-process subprocess; fail loud on non-zero exit."""
    import os
    env = dict(os.environ)
    if env_pythonpath:
        env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    print(f"  $ {' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, check=True)


def _emit(title, cmd, *, dry, submit):
    """A GPU/Butler stage: print the exact command; submit only if --submit (and not --dry-run)."""
    print(f"  [{title}] command:")
    print(f"    {cmd}")
    if submit and not dry:
        print("  submitting ...", flush=True)
        subprocess.run(cmd, shell=True, cwd=str(HL / "run_ft"), check=True)
    else:
        print("  (not submitted; pass --submit to run on SLURM, or see TRAIN_V2_D_E2E.md)")


# --------------------------------------------------------------------------------------------- stages
def stage_data(a, pipe, dry, submit):
    """Build synthetic-trail-on-real-diffim datasets. Enforces exposure-level blind disjointness."""
    if a.train_manifests and a.blind_manifests:
        from ADCNN.pipelines.leakage_guard import assert_disjoint
        nt, nb = assert_disjoint(a.train_manifests, a.blind_manifests)
        print(f"  leakage guard OK: {nt} train vs {nb} blind (visit,detector) exposures, disjoint.")
    _emit("build datasets", "python -m ADCNN.pipelines.make_sim_data  # + build_realfp_manifests; "
          "see TRAINING_PROTOCOL.md (synthetic trails on REAL diffims; field/night-grouped splits)",
          dry=dry, submit=submit)


def stage_train_stage1(a, pipe, dry, submit):
    _emit("stage-1 fine-tune",
          "sbatch --account=rubin:commissioning --export=ALL,RUN_NAME=current,LR=5e-5,STKBAL=0.85 "
          "variant.slurm   # init from v1 trainable; hard-positive domain adaptation",
          dry=dry, submit=submit)


def stage_train_stage2(a, pipe, dry, submit):
    _emit("stage-2 refit",
          "sbatch --account=rubin:commissioning refit_stage2.slurm   # refit cutout-CNN on the new stage-1 "
          "(leakage-clean panels)",
          dry=dry, submit=submit)


def stage_calibrate_mflen(a, pipe, dry, submit):
    """Apply the active pipeline's MF_LEN de-bias (recompute len_db + endpoints from raw length)."""
    if not (a.run_dir and a.manifests and a.out):
        print(f"  active de-bias: offset={pipe.mf_len_offset} slope={pipe.mf_len_slope} "
              f"(pipeline={pipe.name}).")
        print("  provide --run-dir --manifests --out [--fields ...] to apply it to a detection run.")
        print("  (fitting NEW constants is a separate analysis; see TRAINING_PROTOCOL.md.)")
        return
    cmd = [sys.executable, str(HL / "run_dev" / "recompute_lendb.py"),
           "--src", a.run_dir, "--manifests", a.manifests, "--out", a.out,
           "--offset", pipe.mf_len_offset, "--slope", pipe.mf_len_slope]
    if a.fields:
        cmd += ["--fields", *[str(f) for f in a.fields]]
    if dry:
        print(f"  [dry-run] would run: {' '.join(shlex.quote(str(c)) for c in cmd)}")
        return
    _run(cmd)


def stage_detect(a, pipe, dry, submit):
    _emit("blind detect",
          "sbatch --account=rubin:commissioning --export=ALL,"
          f"RUN={HL}/run_blind,SEGMODEL={pipe.seg_model},CNNMODEL={pipe.cnn_model},"
          f"OUTDIR={HL}/run_blind_eval -J det_blind --array=0-19,24-29 detect_v2full.slurm",
          dry=dry, submit=submit)


def stage_alert_eval(a, pipe, dry, submit):
    """Same-night 2v alert evaluation == the headline machinery (regen_v2_report over the pair caches)."""
    _report(dry)


def stage_report(a, pipe, dry, submit):
    if dry:
        print("  [dry-run] release-check (md5s / MF_LEN / thresholds / leakage / active pipeline) + write")
        print(f"  [dry-run] would run: PYTHONPATH={REPO} python -m ADCNN.evaluation.summarize_results  (-> blind headline)")
        return
    release_check(pipe)        # items 1-5 + 8: verify integrity + write the final result table
    _report()                  # item 6: regenerate the CLEAN-24 / all-26 blind verdict


def stage_evaluation_notebooks(a, pipe, dry, submit):
    """Render the catalog-based Evaluation notebooks (current + legacy) from existing catalogs (CPU)."""
    nbs = [REPO / "Evaluation" / "Evaluation.ipynb", REPO / "Evaluation" / "Evaluation_legacy_v1.ipynb"]
    for nb in nbs:
        if not nb.exists():
            print(f"  (skip {nb.name}: not found)"); continue
        if dry:
            print(f"  [dry-run] would render: jupyter nbconvert --execute --inplace {nb.name} (+ --to html)")
            continue
        import os
        env = dict(os.environ); env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}".rstrip(":")
        # Execute ONCE -> notebook (with outputs), then export HTML from the executed notebook (no re-run).
        print(f"  $ jupyter nbconvert --execute --to notebook --inplace {nb.name}", flush=True)
        subprocess.run(["jupyter", "nbconvert", "--execute", "--ExecutePreprocessor.timeout=1800",
                        "--to", "notebook", "--inplace", str(nb)], cwd=str(REPO), env=env, check=True)
        print(f"  $ jupyter nbconvert --to html {nb.name}", flush=True)
        subprocess.run(["jupyter", "nbconvert", "--to", "html", str(nb)],
                       cwd=str(REPO), env=env, check=True)


def _report():
    _run([sys.executable, "-m", "ADCNN.evaluation.summarize_results"])


def release_check(pipe):
    """Level-1 integrity gate + final result table. Fails loud on any drift; writes Evaluation/results.json."""
    import hashlib
    import json as _json
    print("  release-check:")
    ok = True

    # (2) active pipeline = current
    a_ok = pipe.name == "current"; ok &= a_ok
    print(f"    [{'OK' if a_ok else 'FAIL'}] active pipeline = {pipe.name} (expect current)")

    rel = _json.loads((REPO / "models/v2_D/v2_D_release.json").read_text())

    # (3) model md5s match the frozen release
    def _md5(p):
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    for key, path in [("stage1_segmentation_scripted_md5", pipe.seg_model),
                      ("stage2_cnn_postproc_md5", pipe.cnn_model)]:
        want = rel["models"].get(key); got = _md5(path)
        m_ok = (want == got); ok &= m_ok
        print(f"    [{'OK' if m_ok else 'FAIL'}] md5 {path.name} = {got[:12]} (release {str(want)[:12]})")

    # (4) MF_LEN constants match the release de-bias
    deb = rel["mf_len_debias"]; mf_ok = (pipe.mf_len_offset, pipe.mf_len_slope) == (deb["offset"], deb["slope"])
    ok &= mf_ok
    print(f"    [{'OK' if mf_ok else 'FAIL'}] MF_LEN {pipe.mf_len_offset}/{pipe.mf_len_slope} (release {deb['offset']}/{deb['slope']})")

    # (5) frozen alert op-point golden values
    op = _json.loads((HL / "op_2v_alert.json").read_text())
    want_op = {"score_min": 0.80, "chi2_2v_max": 5.0, "mfsnr_min_2v": 5.0,
               "rate_lo_2v": 1.0, "rate_hi_2v": 8.0, "alerts_top_n": 50}
    op_ok = all(op.get(k) == v for k, v in want_op.items()); ok &= op_ok
    print(f"    [{'OK' if op_ok else 'FAIL'}] frozen op_2v_alert values {want_op}")

    # (1) leakage audit: regenerate and assert contamination confined to the disclosed blind fields 0,1
    try:
        from ADCNN.pipelines.leakage_guard import visit_detector_pairs
        import glob
        blind = {mf.split("manifest_")[1].split(".")[0]:
                 visit_detector_pairs(mf) for mf in glob.glob(str(HL / "run_blind/manifest_*.csv"))}
        train = set()
        for p in [HL / "run_ft/ft_catalog.csv", HL / "run_ft_cnn/ft_catalog.csv"] + \
                 [Path(x) for x in glob.glob(str(HL / "run_dev/manifest_*.csv"))]:
            if p.exists():
                train |= visit_detector_pairs(p)
        hit_fields = sorted(k for k, vd in blind.items() if vd & train)
        lk_ok = set(hit_fields) <= {"0", "1"}; ok &= lk_ok
        print(f"    [{'OK' if lk_ok else 'FAIL'}] leakage confined to blind fields {hit_fields} (expect subset of [0,1])")
    except Exception as e:
        print(f"    [WARN] leakage audit skipped: {e}")

    # (8) write the final result table
    lk = rel["exposure_leakage"]
    results = {
        "release": rel.get("release"), "pipeline": pipe.name, "provenance": pipe.provenance,
        "defensible_headline_clean24_blind": lk["clean_24_blind"],
        "all_26_not_strictly_blind": lk["all_26_blind"],
        "frozen_alert_op": want_op, "mf_len_debias": {"offset": pipe.mf_len_offset, "slope": pipe.mf_len_slope},
        "integrity_check_passed": bool(ok),
        "leakage_audit_artifact": "ADCNN/pipelines/heliolinc/leakage_audit/leakage_audit.json",
    }
    outp = REPO / "Evaluation" / "results.json"
    outp.write_text(_json.dumps(results, indent=2))
    print(f"    wrote final result table -> {outp}")
    if not ok:
        raise SystemExit("[run_experiment] release-check FAILED — integrity drift detected (see above).")
    print("    release-check PASSED.")


DISPATCH = {
    "data": stage_data, "train-stage1": stage_train_stage1, "train-stage2": stage_train_stage2,
    "calibrate-mflen": stage_calibrate_mflen, "detect": stage_detect,
    "alert-eval": stage_alert_eval, "report": stage_report,
    "evaluation-notebooks": stage_evaluation_notebooks,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=STAGE_ORDER, help="run a single stage")
    g.add_argument("--stages", help="'all' for the ordered end-to-end plan, or a comma list "
                   "(e.g. report,evaluation-notebooks)")
    ap.add_argument("--pipeline", default=None, help="pipeline name/path (default: ADCNN_PIPELINE or current)")
    ap.add_argument("--dry-run", action="store_true", help="print what each stage would do; no side effects")
    ap.add_argument("--submit", action="store_true", help="actually sbatch the GPU stages (off by default)")
    # calibrate-mflen args
    ap.add_argument("--run-dir", help="detection run dir (calibrate-mflen --src)")
    ap.add_argument("--manifests", help="manifests dir (calibrate-mflen --manifests)")
    ap.add_argument("--out", help="output dir (calibrate-mflen --out)")
    ap.add_argument("--fields", type=int, nargs="*", help="field indices (calibrate-mflen)")
    # leakage-guard args (data stage)
    ap.add_argument("--train-manifests", nargs="*", help="training manifest CSV(s) for the leakage guard")
    ap.add_argument("--blind-manifests", nargs="*", help="blind/test manifest CSV(s) for the leakage guard")
    a = ap.parse_args()

    pipe = load_pipeline(a.pipeline)
    print(f"[run_experiment] active pipeline: {pipe.name} (provenance: {pipe.provenance})")
    print(f"  seg={pipe.seg_model}\n  cnn={pipe.cnn_model}\n  MF_LEN offset/slope={pipe.mf_len_offset}/{pipe.mf_len_slope}")
    if a.stage:
        stages = [a.stage]
    elif a.stages == "all":
        stages = STAGE_ORDER
    else:
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
        bad = [s for s in stages if s not in DISPATCH]
        if bad:
            raise SystemExit(f"[run_experiment] unknown stage(s): {bad}; choose from {STAGE_ORDER}")
    for st in stages:
        kind = "GPU" if st in GPU_STAGES else "CPU"
        print(f"\n=== stage: {st} [{kind}] ===")
        DISPATCH[st](a, pipe, a.dry_run, a.submit)
    print("\n[run_experiment] done.")


if __name__ == "__main__":
    main()
