"""End-to-end entry points for the ADCNN diffim asteroid-trail pipeline.

  make_sim_data      ALL simulated (injected-trail) sets — train/val, train2/val2, test — from
                     one deterministic panel partition (Butler); levers to build only some
  make_real_data     real-asteroid test diffim dataset from the Butler
  train_end_to_end   train the segmentation model detector (reg2 recipe) then the focal cutout CNN 2nd stage
  run_inference      run segmentation model + cutout CNN on diffim panels -> scored candidate detections
  make_eval_catalogs build detection catalogs on the test sets + catalog-based evaluation metrics

Each is runnable as ``python -m ADCNN.pipelines.<name> --help``.
The deployed models live in the top-level ``models/`` directory.
"""
