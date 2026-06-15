"""ADCNN training/validation pipeline -- DECIDES the operating point and FREEZES a self-contained
model release.

This is the first of the two top-level pipelines (the second is ``run_night``, which APPLIES the
frozen release). It runs the full model-building protocol and, crucially, makes the shipped ADCNN
score threshold a **formal output of the validation protocol** rather than an inherited constant:

    train-stage1      domain-adapt the stage-1 segmentation model               (GPU; emits sbatch)
    train-stage2      refit the stage-2 cutout-CNN scorer on the new stage-1     (GPU; emits sbatch)
    calibrate-mflen   re-fit + confirm the MF_LEN trail-length de-bias           (CPU)
    validation-detect run detection on the 82 validation injection fields        (GPU; emits sbatch)
    threshold-select  regenerate validation curves -> pre-declared decision rule
                      -> select + CONFIRM the alert operating point              (CPU)
    freeze            assemble the self-contained release dir                    (CPU)

GPU/Butler stages PRINT the exact ``sbatch`` (backend ``heliolinc/train_v2_D_e2e.sh`` +
``TRAIN_V2_D_E2E.md``); they submit only with ``--submit``. The CPU stages (calibrate-mflen,
threshold-select, freeze) run in-process and are reproducible from a clean checkout (committed
validation caches + MF_LEN fit pairs). ``--dry-run`` prints what each stage would do.

    # reproduce/document the protocol + freeze a release from the current frozen model (CPU only):
    python -m ADCNN.pipelines.train_and_validate --config models/current/pipeline.json \
        --out models/current_candidate --stages calibrate-mflen,threshold-select,freeze
    # the full end-to-end plan (GPU stages emit sbatch):
    python -m ADCNN.pipelines.train_and_validate --stages all --dry-run

The freeze step writes a release dir that travels as one unit:
    stage1.pt stage2.pt stage2.json   mflen.json   thresholds.json   md5s.json
    validation_report.json   threshold_sweep.csv   threshold_plots/   pipeline.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ADCNN.config import load_pipeline, REPO
from ADCNN.calibration import calibrate_mflen, threshold_selection
from ADCNN.pipelines.run_experiment import _emit, HL

STAGE_ORDER = ["train-stage1", "train-stage2", "calibrate-mflen",
               "validation-detect", "threshold-select", "freeze"]
GPU_STAGES = {"train-stage1", "train-stage2", "validation-detect"}

DEFAULT_FROZEN_OP = HL / "op_2v_alert.json"


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- stages
def stage_train_stage1(a, pipe, dry, submit):
    _emit("stage-1 fine-tune",
          "sbatch --account=rubin:commissioning --export=ALL,RUN_NAME=current,LR=5e-5,STKBAL=0.85 "
          "variant.slurm   # init from v1 trainable; hard-positive domain adaptation",
          dry=dry, submit=submit)


def stage_train_stage2(a, pipe, dry, submit):
    _emit("stage-2 refit",
          "sbatch --account=rubin:commissioning refit_stage2.slurm   # refit cutout-CNN on the new "
          "stage-1 (leakage-clean panels)", dry=dry, submit=submit)


def stage_calibrate_mflen(a, pipe, dry, submit):
    """Re-fit + CONFIRM the MF_LEN de-bias (Level-1 from committed pairs)."""
    if dry:
        print(f"  [dry-run] re-fit MF_LEN vs frozen {pipe.mf_len_offset}/{pipe.mf_len_slope} "
              f"(fail-loud on drift); would write mflen.json")
        return
    fitted, _ = calibrate_mflen.run(fit_csv=a.mflen_fit_csv, confirm=True)
    print(f"  MF_LEN re-fit offset={fitted['offset']:.4f} slope={fitted['slope']:.4f} "
          f"(n={fitted['fit_n']}) -> CONFIRMED vs frozen {pipe.mf_len_offset}/{pipe.mf_len_slope}")


def stage_validation_detect(a, pipe, dry, submit):
    _emit("validation detect",
          "sbatch --account=rubin:commissioning --export=ALL,"
          f"RUN={HL}/run_lambda,SEGMODEL={pipe.seg_model},CNNMODEL={pipe.cnn_model} "
          "detect_ada.slurm   # 82 off-ecliptic injection fields -> per-pair caches", dry=dry, submit=submit)


def stage_threshold_select(a, pipe, dry, submit):
    """Regenerate validation curves, apply the pre-declared rule, CONFIRM == frozen op (fail loud)."""
    if dry:
        print("  [dry-run] regenerate C/P curves from validation caches -> purity-floor(S)+retention(mfsnr)"
              " rule -> assert == op_2v_alert.json")
        return
    selected, _ = threshold_selection.run(cache_dir=a.cache_dir, frozen_op=a.frozen_op, confirm=True)
    op = selected
    print(f"  selected op: score_min={op['score_min']} mfsnr_min={op['mfsnr_min']} "
          f"chi2_max={op['chi2_max']} rate[{op['rate_lo']},{op['rate_hi']}]  "
          f"(C={op['at_op']['faint_fast_completeness_pct']}% P={op['at_op']['in_sample_purity_pct']}%)")
    print("  CONFIRMED: regenerated selection matches the frozen op.")


def stage_freeze(a, pipe, dry, submit):
    """Assemble the self-contained release dir from the (frozen) model + regenerated calibration."""
    out = Path(a.out)
    if dry:
        print(f"  [dry-run] would freeze release -> {out} "
              "(model pointers + mflen.json + thresholds.json + md5s.json + validation_report.json "
              "+ threshold_sweep.csv + threshold_plots/ + pipeline.json)")
        return
    out.mkdir(parents=True, exist_ok=True)

    # 1. model artifacts -> symlinks (the models/current/ convention; md5 is the immutable identity)
    model_map = {"stage1.pt": pipe.seg_model, "stage2.pt": pipe.cnn_model}
    if pipe.cnn_sidecar:
        model_map["stage2.json"] = pipe.cnn_sidecar
    md5s = {}
    for name, src in model_map.items():
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(Path(src).resolve(), out))
        md5s[name] = _md5(src)
    (out / "md5s.json").write_text(json.dumps(md5s, indent=2))

    # 2. MF_LEN: re-fit + confirm, write mflen.json
    fitted, mflen_rec = calibrate_mflen.run(fit_csv=a.mflen_fit_csv, out=str(out), confirm=True)

    # 3. thresholds: regenerate + confirm, write validation_report.json + threshold_sweep.csv
    selected, report = threshold_selection.run(cache_dir=a.cache_dir, frozen_op=a.frozen_op,
                                               out=str(out), confirm=True)
    frozen_op = json.loads(Path(a.frozen_op).read_text())
    thresholds = {k: selected[k] for k in ("score_min", "mfsnr_min", "chi2_max", "rate_lo", "rate_hi")}
    thresholds.update({
        "ranking": "priorityScore = weakest-member ADCNN score (alert_stream.priority_score)",
        "budget_top_n": frozen_op.get("alerts_top_n", 50),
        "promote_3v": frozen_op.get("promote_3v", True),
        "frozen_op_point": os.path.relpath(Path(a.frozen_op).resolve(), REPO),
        "provenance": "regenerated by ADCNN.calibration.threshold_selection (purity-floor rule), "
                      "confirmed == frozen op_2v_alert.json",
    })
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # 4. threshold selection figures (non-critical; warn but do not fail the freeze)
    plot_dir = out / "threshold_plots"
    plot_dir.mkdir(exist_ok=True)
    _generate_threshold_plots(plot_dir, a.cache_dir)

    # 5. pipeline.json (same schema as models/current + inlined thresholds + release provenance)
    pj = {
        "_comment": "Self-contained ADCNN release frozen by ADCNN.pipelines.train_and_validate. "
                    "Model files are pointers; md5s.json is the immutable identity. Thresholds are the "
                    "FORMAL output of threshold_selection (regenerated + confirmed against the frozen op).",
        "name": out.name,
        "provenance": pipe.provenance,
        # Pointers are REPO-relative paths to THIS release dir's own symlinks: ADCNN.config._resolve()
        # resolves non-absolute model paths against REPO, so a bare "stage1.pt" would (wrongly) point at
        # REPO/stage1.pt. REPO-relative keeps the release self-contained AND loadable by run_night.
        "models": {"segmentation": os.path.relpath(out / "stage1.pt", REPO),
                   "cnn_postproc": os.path.relpath(out / "stage2.pt", REPO),
                   "cnn_sidecar": (os.path.relpath(out / "stage2.json", REPO)
                                   if pipe.cnn_sidecar else None)},
        "mf_len_debias": {"offset": mflen_rec["offset"], "slope": mflen_rec["slope"],
                          "_comment": "re-fit + confirmed; see mflen.json"},
        "cnn_thr_floor": pipe.cnn_thr_floor,
        "alert_op_point": os.path.relpath(Path(a.frozen_op).resolve(), REPO),
        "thresholds": thresholds,
        "md5s": "md5s.json",
        "validation_report": "validation_report.json",
    }
    (out / "pipeline.json").write_text(json.dumps(pj, indent=2))
    print(f"  froze release -> {out}/")
    print(f"    models: {', '.join(model_map)} (md5s.json)")
    print(f"    mflen.json {mflen_rec['offset']}/{mflen_rec['slope']}; "
          f"thresholds.json score_min={thresholds['score_min']} mfsnr_min={thresholds['mfsnr_min']}")
    print(f"    validation_report.json + threshold_sweep.csv + threshold_plots/ + pipeline.json")


def _generate_threshold_plots(plot_dir, cache_dir):
    """Render the threshold-selection figures into the release dir (best-effort)."""
    # qa.plots_thresholds is the canonical figure module after the reorg; fall back to the current
    # Evaluation/ script until then. Either reads the same caches via threshold_selection.make_metrics.
    script = REPO / "Evaluation" / "threshold_selection_plots.py"
    candidates = [
        [sys.executable, "-m", "ADCNN.qa.plots_thresholds", "--cache-dir", str(cache_dir),
         "--out", str(plot_dir)],
        [sys.executable, str(script), "--cache-dir", str(cache_dir), "--out", str(plot_dir)],
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["MPLBACKEND"] = "Agg"
    for cmd in candidates:
        try:
            subprocess.run(cmd, cwd=str(REPO), env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"    threshold_plots/ generated via {cmd[1] if cmd[1]=='-m' else 'Evaluation script'}")
            return
        except Exception:
            continue
    print("    [warn] threshold_plots/ not generated (figure script unavailable); non-critical")


DISPATCH = {
    "train-stage1": stage_train_stage1, "train-stage2": stage_train_stage2,
    "calibrate-mflen": stage_calibrate_mflen, "validation-detect": stage_validation_detect,
    "threshold-select": stage_threshold_select, "freeze": stage_freeze,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="pipeline config to build from (name/path; default: active/current)")
    ap.add_argument("--out", default="models/current_candidate", help="release dir for the freeze stage")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stage", choices=STAGE_ORDER, help="run a single stage")
    g.add_argument("--stages", default="all", help="'all' or a comma list (e.g. "
                   "calibrate-mflen,threshold-select,freeze)")
    ap.add_argument("--dry-run", action="store_true", help="print what each stage would do; no side effects")
    ap.add_argument("--submit", action="store_true", help="actually sbatch the GPU stages (off by default)")
    ap.add_argument("--cache-dir", default=str(threshold_selection.DEFAULT_CACHE_DIR),
                    help="validation per-pair caches (threshold-select)")
    ap.add_argument("--frozen-op", default=str(DEFAULT_FROZEN_OP),
                    help="frozen alert op-point to confirm against")
    ap.add_argument("--mflen-fit-csv", default=None, help="committed MF_LEN fit pairs (calibrate-mflen)")
    a = ap.parse_args()

    pipe = load_pipeline(a.config)
    print(f"[train_and_validate] building from pipeline: {pipe.name} (provenance: {pipe.provenance})")
    print(f"  seg={pipe.seg_model}\n  cnn={pipe.cnn_model}\n  MF_LEN={pipe.mf_len_offset}/{pipe.mf_len_slope}")
    if a.stage:
        stages = [a.stage]
    elif a.stages == "all":
        stages = STAGE_ORDER
    else:
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
        bad = [s for s in stages if s not in DISPATCH]
        if bad:
            raise SystemExit(f"[train_and_validate] unknown stage(s): {bad}; choose from {STAGE_ORDER}")
    for st in stages:
        kind = "GPU" if st in GPU_STAGES else "CPU"
        print(f"\n=== stage: {st} [{kind}] ===")
        DISPATCH[st](a, pipe, a.dry_run, a.submit)
    print("\n[train_and_validate] done.")


if __name__ == "__main__":
    main()
