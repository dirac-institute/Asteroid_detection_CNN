"""End-to-end entry points for the ADCNN diffim asteroid-trail pipeline.

  make_sim_data           build the simulated injected-trail train/val/test sets from the Butler
  make_real_data          build the real-asteroid test diffim set from the Butler
  train_end_to_end        train the segmentation model + the focal cutout-CNN second stage
  make_eval_catalogs      score the test sets with the deployed models -> detection catalogs + metrics

Each is runnable as ``python -m ADCNN.pipelines.<name> --help``.

The end-to-end inference engine is ``ADCNN.inference.catalog``; for a single h5 -> CSV run,
``python -m ADCNN.inference.catalog`` is the direct CLI.

Deployed weights live in the top-level ``models/`` directory.
"""
