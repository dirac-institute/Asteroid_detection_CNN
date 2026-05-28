"""ENTRY POINT — produce SIMULATED (injected-trail) train/test diffim datasets.

Streams preliminary_visit_image + template_coadd from the Butler, builds the PSF-matched
difference image, injects realistic asteroid trails (non-uniform brightness, tapered ends,
slight curvature), runs LSST-stack detection for the stack-detection label, and writes
train.h5/test.h5 (+ catalog CSVs). Thin, documented wrapper around
``ADCNN.data.dataset_creation.simulate`` (run that module's ``--help`` for every knob).

Common recipe for all sets: realistic trails (``--realistic-trail``), trail length 6-60 px,
SNR 2-8 (``--mag-mode snr --mag-min 2 --mag-max 8``), beta 0-180 deg, 20 injections/panel, and
ALWAYS exclude the held-out test pairs so the model never trains on a test (visit,detector):
``--exclude-pairs-csv DATA_DIFFIM/test_5sigma/test.csv DATA_DIFFIM/test_real/test.csv``.

The three datasets differ only in size / split / seed (use a DIFFERENT --seed per set so they
draw disjoint injections):

  TRAIN — stage-1 v7 training data (the deployed reg2 shards):
    python -m ADCNN.pipelines.make_sim_data \\
        --save-path DATA_DIFFIM_realistic --random-subset 4300 --seed 123 \\
        --realistic-trail --skip-prevalidation \\
        --mag-mode snr --mag-min 2 --mag-max 8 \\
        --exclude-pairs-csv DATA_DIFFIM/test_5sigma/test.csv DATA_DIFFIM/test_real/test.csv

  TRAIN2 — dedicated stage-2 cutout-CNN training set (disjoint from train + test):
    python -m ADCNN.pipelines.make_sim_data \\
        --save-path DATA_DIFFIM/train2 --random-subset 500 --seed 4242 \\
        --realistic-trail --skip-prevalidation \\
        --mag-mode snr --mag-min 2 --mag-max 8 \\
        --exclude-pairs-csv DATA_DIFFIM/test_5sigma/test.csv DATA_DIFFIM/test_real/test.csv
    # then cache cutouts: ADCNN.training.cnn_postproc.build_cutout_dataset(v7, train2/train.h5,
    #   train2/train.csv, <cut_dir>) and train on them.

  TEST — held-out evaluation sets at a given stack-detection sigma (--test-only writes test.h5):
    for S in 5 4 3; do python -m ADCNN.pipelines.make_sim_data \\
        --save-path DATA_DIFFIM/test_${S}sigma --random-subset 300 --seed $((900+S)) \\
        --realistic-trail --skip-prevalidation --test-only \\
        --mag-mode snr --mag-min 2 --mag-max 8 \\
        --stack-detection-threshold ${S}; done
"""
from ADCNN.data.dataset_creation.simulate import main

if __name__ == "__main__":
    main()
