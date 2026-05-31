"""ENTRY POINT — produce ALL the SIMULATED (injected-trail) diffim datasets in one command.

Streams preliminary_visit_image + template_coadd from the Butler, builds the PSF-matched
difference image, injects realistic asteroid trails, runs LSST-stack detection for the
stack-detection label, and writes one ``<set>.h5`` (+ ``<set>.csv``) per set under ``--save-path``.
Thin wrapper around ``ADCNN.data.dataset_creation.simulate`` (run that module's ``--help`` for
every knob).

The five sets come from ONE deterministic panel partition (cached in ``<save-path>/split.json``),
so they never share a (visit,detector) panel:

    train  (+ val)   stage-1 segmentation training  (val  -> model selection)
    cnn_train (+ cnn_val)  stage-2 cutout-CNN training     (cnn_val -> FP-filter threshold)
    test             held-out evaluation

Build everything (defaults: realistic trails, SNR 2-8, trail length 6-60 px, 20 injections/panel,
seed 2026):

    python -m ADCNN.pipelines.make_sim_data --save-path DATA_DIFFIM --realistic-trail \\
        --n-train 1500 --n-val 150 --n-cnn_train 500 --n-cnn_val 100 --n-test 300 \\
        --mag-mode snr --mag-min 2 --mag-max 8 --parallel 90

Levers — build only some groups (the partition still covers all five, so partial builds stay
consistent with a full build):

    --sets train          # build train + val only
    --sets cnn_train         # build cnn_train + cnn_val only
    --sets test           # build test only
    --sets test --test-sigmas 5 4 3   # ONE gzip'd test.{h5,csv} labelled at each sigma (build once)

Determinism: a fixed ``--seed`` fixes panel selection + partition + each panel's injections,
so every rerun selects the same panels and injects identical trails (a panel that fails to
subtract is dropped — failures are repeatable). ``--exclude-pairs-csv`` keeps EXTERNAL panels
(e.g. the real-asteroid test_real catalog) out of the universe.
"""
from ADCNN.data.dataset_creation.simulate import main

if __name__ == "__main__":
    main()
