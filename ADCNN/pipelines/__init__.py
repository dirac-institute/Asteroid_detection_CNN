"""End-to-end entry points for the ADCNN diffim asteroid-trail pipeline.

  make_sim_data      simulated (injected-trail) train/test diffim datasets from the Butler
  make_real_data     real-asteroid test diffim dataset from the Butler
  train_end_to_end   train the v7 detector (reg2 recipe) then the focal cutout CNN 2nd stage
  run_inference      run v7 + cutout CNN on diffim panels -> scored candidate detections

Each is runnable as ``python -m ADCNN.pipelines.<name> --help``.
The deployed models live in the top-level ``models/`` directory.
"""
