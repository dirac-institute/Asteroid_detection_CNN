"""Every production module must at least IMPORT.

A BLAS-pinning edit once left ADCNN/inference/catalog.py syntactically invalid, and the whole suite
still passed -- nothing in it imported that module -- so the breakage was discovered only when a
2-hour GPU job died on the first line. A syntax error should cost seconds, not a job.

Import-only: no GPU, no Butler, no pixel data.
"""
import importlib

import pytest

MODULES = [
    "ADCNN.inference.catalog",
    "ADCNN.inference.features",
    "ADCNN.inference.matched_filter",
    "ADCNN.inference.predict",
    "ADCNN.inference.cnn_postproc",
    "ADCNN.inference.mf_trail_length",
    "ADCNN.inference.diffim_io",
    "ADCNN.data.preprocessing",
    "ADCNN.linking.link_2visit",
    "ADCNN.linking.merge_dets",
    "ADCNN.linking.clean_dets",
    "ADCNN.linking.rank_alerts",
    "ADCNN.linking.ingest_diasource",
    "ADCNN.qa.filter_op",
    "ADCNN.qa.alert_cutouts",
    "ADCNN.qa.alert_morphology",
    "ADCNN.qa.alert_sheets",
    "ADCNN.qa.alert_pairs",
    "ADCNN.qa.select_clean",
    "ADCNN.qa.rerank_alerts",
    "ADCNN.pipelines.run_night",
    "ADCNN.pipelines.night_status",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)
