"""Tests for run_night's integrity preflight + alert-op default.

Run with pytest from the repo root, or standalone:
    python -m ADCNN.pipelines.tests.test_run_night_preflight
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ADCNN.config import load_pipeline
from ADCNN.pipelines import run_night as RN

REPO = Path(__file__).resolve().parents[3]
ALERT_OP = REPO / "ADCNN/pipelines/heliolinc/op_2v_alert.json"
DISC_OP = REPO / "ADCNN/pipelines/heliolinc/link_op_point.json"


def test_preflight_passes_on_intact_release():
    pipe = load_pipeline("current")
    RN.preflight(pipe, str(ALERT_OP), discovery=False)  # must not raise


def test_preflight_rejects_tampered_op(tmp_path):
    pipe = load_pipeline("current")
    d = json.loads(ALERT_OP.read_text())
    d["score_min"] = 0.55  # tamper the frozen alert cut
    bad = tmp_path / "op_tampered.json"
    bad.write_text(json.dumps(d))
    raised = False
    try:
        RN.preflight(pipe, str(bad), discovery=False)
    except RN.IntegrityError:
        raised = True
    assert raised, "preflight must FAIL LOUD on a tampered op-point"


def test_discovery_op_validates_against_discovery_golden():
    pipe = load_pipeline("current")
    RN.preflight(pipe, str(DISC_OP), discovery=True)   # discovery op vs discovery golden: passes


def test_discovery_op_rejected_under_alert_product():
    # the discovery op (mfsnr=10) must NOT pass as the default alert product (mfsnr=5)
    pipe = load_pipeline("current")
    raised = False
    try:
        RN.preflight(pipe, str(DISC_OP), discovery=False)
    except RN.IntegrityError:
        raised = True
    assert raised, "discovery op must fail the alert-product preflight (different mfsnr floor)"


if __name__ == "__main__":
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"PASS {name}")
    print("all run_night preflight tests passed")
