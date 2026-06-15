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
               "detect", "alert-eval", "report"]
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
    _report(dry)
    tsp = REPO / "Evaluation" / "threshold_selection_plots.py"
    if tsp.exists():
        if dry:
            print(f"  [dry-run] would run: python {tsp}")
        else:
            try:
                _run([sys.executable, str(tsp)], cwd=tsp.parent)
            except subprocess.CalledProcessError as e:
                print(f"  (threshold_selection_plots.py failed: {e}; headline table above is the verdict)")


def _report(dry):
    rep = HL / "regen_v2_report.py"
    if dry:
        print(f"  [dry-run] would run: PYTHONPATH={REPO} python {rep}  (-> blind headline table)")
        return
    _run([sys.executable, str(rep)], cwd=HL)


DISPATCH = {
    "data": stage_data, "train-stage1": stage_train_stage1, "train-stage2": stage_train_stage2,
    "calibrate-mflen": stage_calibrate_mflen, "detect": stage_detect,
    "alert-eval": stage_alert_eval, "report": stage_report,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=STAGE_ORDER, help="run a single stage")
    g.add_argument("--stages", choices=["all"], help="run the ordered end-to-end plan")
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
    stages = STAGE_ORDER if a.stages == "all" else [a.stage]
    for st in stages:
        kind = "GPU" if st in GPU_STAGES else "CPU"
        print(f"\n=== stage: {st} [{kind}] ===")
        DISPATCH[st](a, pipe, a.dry_run, a.submit)
    print("\n[run_experiment] done.")


if __name__ == "__main__":
    main()
