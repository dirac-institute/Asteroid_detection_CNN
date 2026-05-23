"""ENTRY POINT — produce SIMULATED (injected-trail) train/test diffim datasets.

Streams preliminary_visit_image + template_coadd from the Butler, builds the
PSF-matched difference image, injects realistic asteroid trails (non-uniform
brightness, tapered ends, slight curvature), runs LSST stack 5-sigma detection
for the stack-detection label, and writes train.h5/test.h5 (+ catalog CSVs).

Production recipe (the data the deployed reg2 model was trained on):
  - realistic trail renderer (--realistic-trail)
  - trail length 6-60 px, SNR 2-8 (mag-mode snr), beta 0-180 deg, 20 injections/panel
  - stack detection threshold 5 sigma
  - EXCLUDE the test_5sigma + test_real (visit,detector) pairs (leakage-safe)

This is a thin, documented wrapper around
``ADCNN.data.dataset_creation.simulate`` which carries the full CLI;
run that module's ``--help`` for every knob. Example production launch:

    python -m ADCNN.pipelines.make_sim_data \\
        --save-path DATA_DIFFIM_realistic --random-subset 4300 \\
        --realistic-trail --skip-prevalidation \\
        --exclude-pairs-csv DATA_DIFFIM/test_5sigma/test.csv DATA_DIFFIM/test_real/test.csv
"""
from ADCNN.data.dataset_creation.simulate import main

if __name__ == "__main__":
    main()
