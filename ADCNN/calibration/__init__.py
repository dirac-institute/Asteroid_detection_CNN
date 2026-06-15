"""Calibration stages that turn validation evidence into the frozen operating point.

These are the *formal* outputs of ``ADCNN.pipelines.train_and_validate`` that make the
shipped thresholds a product-level decision rather than inherited constants:

  * :mod:`ADCNN.calibration.threshold_selection` -- regenerate the validation
    completeness/purity curves from the committed per-pair caches, apply the
    pre-declared decision rule, and CONFIRM it re-selects the frozen alert op-point.
  * :mod:`ADCNN.calibration.calibrate_mflen` -- fit the matched-filter trail-length
    de-bias ``(offset, slope)`` for the active stage-1 and confirm the frozen values.
"""
