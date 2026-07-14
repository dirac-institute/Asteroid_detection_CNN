"""adcnn -- single top-level entry point for the ADCNN asteroid-discovery pipelines.

Usage:
    ./adcnn <command> [args...]          # from the repo root (recommended: sets CWD for sbatch)
    python -m ADCNN <command> [args...]  # equivalent, from the repo root

Commands:
    night               APPLY the frozen release to one observing night, end-to-end:
                        manifest -> GPU detect -> known-catalog -> mask -> static-veto catalog
                        -> 2-visit linking (train veto, FPP, report) -> pixel vet -> MPC crossmatch.
                        -> ADCNN.pipelines.run_night

    experiment          Detector-workflow driver for model development: data build, stage-1/2
                        training, MF_LEN calibration, detection, alert eval, reports/notebooks.
                        GPU stages print their sbatch command (submit with --submit).
                        -> ADCNN.pipelines.run_experiment

    train-and-validate  BUILD + VALIDATE a model release: trains, re-derives calibrations,
                        regenerates the validation curves, selects + CONFIRMS the operating
                        point against the frozen op, and freezes a self-contained release dir.
                        -> ADCNN.pipelines.train_and_validate

`./adcnn <command> --help` shows that command's full argument list.

Examples:
    # score a real night (visits known) with the current frozen release:
    ./adcnn night --collection LSSTCam/runs/DRP/20250625_20250705/w_2025_28/DM-51933 \
        --night 60860 --visits 2025062500123,2025062500124 --no-known

    # dry-run the full night chain (prints every stage command, runs nothing):
    ./adcnn night --collection <coll> --night 60867 --tracts 8489 --dry-run

    # reproduce the release-freeze protocol from a clean checkout (CPU only):
    ./adcnn train-and-validate --config models/current/pipeline.json \
        --out models/current_candidate --stages calibrate-mflen,threshold-select,freeze
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "night": "ADCNN.pipelines.run_night",
    "experiment": "ADCNN.pipelines.run_experiment",
    "train-and-validate": "ADCNN.pipelines.train_and_validate",
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"adcnn: unknown command '{cmd}' (choose from: {', '.join(COMMANDS)})", file=sys.stderr)
        return 2
    mod = importlib.import_module(COMMANDS[cmd])
    # every sub-pipeline's main() reads sys.argv via argparse; re-brand argv[0] so
    # --help/usage lines read "adcnn <command> ..." instead of the module path.
    sys.argv = [f"adcnn {cmd}"] + rest
    return mod.main() or 0


if __name__ == "__main__":
    sys.exit(main())
