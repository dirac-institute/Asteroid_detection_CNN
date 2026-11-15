"""Detect stage: mask-plane parser + fractions, detect_night helpers, merge coverage, MF_K."""
import concurrent.futures
import contextlib
import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.time import Time

from ADCNN.inference import detect_night as dn
from ADCNN.pipelines.usdf_night import mask_planes as mp

V1, V2, V3 = 2026111300017, 2026111300018, 2026111300019


def _mskn_header(planes):
    h = fits.Header()
    for i, (name, value) in enumerate(planes):
        h[f"MSKN{i:04d}"] = name
        h[f"MSKM{i:04d}"] = value
        h[f"MSKD{i:04d}"] = "description"
    return h


# ---------------------------------------------------------------- mask_planes.plane_values

def test_plane_values_mskn_aliases_and_values():
    h = _mskn_header([("BAD", 1), ("SATURATED", 2), ("INTERPOLATED", 4), ("COSMIC_RAY", 8),
                      ("DETECTION_EDGE", 16), ("UNMASKED_NAN", 16384), ("STREAK", 262144)])
    assert mp.plane_values(h) == {"BAD": 1, "SAT": 2, "INTRP": 4, "CR": 8, "EDGE": 16,
                                  "UNMASKEDNAN": 16384, "STREAK": 262144}


def test_plane_values_legacy_mp_bit_index():
    h = fits.Header()
    h["MP_BAD"] = 0
    h["MP_SAT"] = 1
    h["MP_CR"] = 3
    h["HIERARCH MP_DETECTED_NEGATIVE"] = 5
    assert mp.plane_values(h) == {"BAD": 1, "SAT": 2, "CR": 8, "DETECTED_NEGATIVE": 32}


def test_plane_values_no_cards_raises():
    h = fits.Header()
    h["EXTNAME"] = "MASK"
    with pytest.raises(ValueError):
        mp.plane_values(h)


def test_art_planes_and_columns():
    assert mp.ART_PLANES == ["SPIKE", "SAT", "CR", "STREAK", "CROSSTALK", "SAT_TEMPLATE", "BAD", "SUSPECT",
                             "EDGE", "NO_DATA", "VIGNETTED", "INTRP", "ITL_DIP", "UNMASKEDNAN",
                             "HIGH_VARIANCE", "DETECTED_NEGATIVE"]
    assert mp.COLUMNS == [f"m_{p}" for p in mp.ART_PLANES] + ["art_frac"]


# ---------------------------------------------------------------- mask_planes.fractions

def _mask():
    m = np.zeros((50, 60), dtype=np.int32)
    m[20, 10:20] |= 8          # CR strip on row y=20, x=10..19
    m[30, 30] |= 2             # one SAT pixel
    return m


def test_fractions_trail_uses_bit_value():
    m = _mask()
    vals = {"CR": 8, "SAT": 2, "BAD": 1}
    f = mp.fractions(m, vals, 20.0, 20.0, 0.0, 20.0)
    xs = np.round(20.0 + np.linspace(-10.0, 10.0, mp.NSAMP))
    assert f["m_CR"] == pytest.approx(np.mean((xs >= 10) & (xs <= 19)))
    assert f["m_CR"] > 0          # a bit-index reading ((acc >> 8) & 1) would give 0 here
    assert f["m_SAT"] == 0.0 and f["m_BAD"] == 0.0
    assert np.isnan(f["m_SPIKE"])  # plane not defined in this file
    assert f["art_frac"] == f["m_CR"]
    assert set(f) == set(mp.COLUMNS)


def test_fractions_perpendicular_halfwidth():
    m = _mask()
    vals = {"CR": 8}
    assert mp.fractions(m, vals, 15.0, 21.0, 0.0, 6.0)["m_CR"] == 1.0   # 1 px off the strip: OR catches it
    assert mp.fractions(m, vals, 15.0, 22.0, 0.0, 6.0)["m_CR"] == 0.0   # 2 px off: outside +-1
    ys = np.round(20.0 + np.linspace(-3.0, 3.0, mp.NSAMP))     # vertical trail crossing the strip
    assert mp.fractions(m, vals, 15.0, 20.0, 90.0, 6.0)["m_CR"] == pytest.approx(np.mean(ys == 20))


def test_fractions_point_box():
    m = _mask()
    vals = {"SAT": 2}
    assert mp.fractions(m, vals, 30.0, 30.0, 0.0, 0.5)["m_SAT"] == pytest.approx(1 / 9)
    assert mp.fractions(m, vals, 31.0, 31.0, 0.0, 0.0)["m_SAT"] == pytest.approx(1 / 9)
    assert mp.fractions(m, vals, 32.0, 32.0, 0.0, 0.0)["m_SAT"] == 0.0
    assert mp.fractions(m, vals, 30.0, 30.0, 0.0, np.nan)["m_SAT"] == pytest.approx(1 / 9)
    assert mp.fractions(m, vals, 30.0, 30.0, np.nan, 10.0)["m_SAT"] == pytest.approx(1 / 9)


def test_fractions_vector_edges_nan_and_no_mask():
    m = _mask()
    vals = {"CR": 8, "SAT": 2}
    f = mp.fractions(m, vals, np.array([0.0, 59.0, np.nan]), np.array([0.0, 49.0, 5.0]),
                     np.array([45.0, 0.0, 0.0]), np.array([30.0, 0.0, 5.0]))
    assert all(isinstance(v, np.ndarray) and len(v) == 3 for v in f.values())
    assert f["m_CR"][0] == 0.0 and f["m_SAT"][1] == 0.0
    assert np.isnan(f["m_CR"][2]) and np.isnan(f["art_frac"][2])
    g = mp.fractions(None, None, np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.zeros(2), np.ones(2))
    assert all(np.isnan(v).all() for v in g.values())
    assert np.isnan(mp.fractions(None, None, 1.0, 1.0, 0.0, 5.0)["art_frac"])


# ---------------------------------------------------------------- detect_night helpers

def test_obscode_is_x05():
    assert dn.OBSCODE == "X05"


def _write_panel(path, with_mask=True):
    img = np.arange(40 * 50, dtype=np.float32).reshape(40, 50)
    img[0, 0] = np.nan
    w = fits.Header()
    for k, v in dict(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=40.0, CRVAL2=-10.0, CRPIX1=25.0,
                     CRPIX2=20.0, CD1_1=-5.5e-5, CD1_2=0.0, CD2_1=0.0, CD2_2=5.5e-5).items():
        w[k] = v
    p = fits.PrimaryHDU()
    p.header["DATE-AVG"] = "2026-11-14T06:25:30.000"
    hdus = [p, fits.CompImageHDU(img, header=w, name="IMAGE")]
    mask = np.zeros((40, 50), dtype=np.int32)
    mask[5, 7] = 8 | 2
    if with_mask:
        hdus.append(fits.CompImageHDU(mask, header=_mskn_header([("SATURATED", 2), ("COSMIC_RAY", 8)]),
                                      name="MASK"))
    fits.HDUList(hdus).writeto(path)
    return img, mask


def test_read_fits_panel_reads_mask_in_same_open(tmp_path):
    p = str(tmp_path / "panel.fits")
    img, mask = _write_panel(p)
    out_img, wcs, mjd, out_mask, values = dn.read_fits_panel(p)
    assert out_img.dtype == np.float32 and out_img[0, 0] == 0.0 and out_img[3, 4] == img[3, 4]
    assert wcs.has_celestial
    assert mjd == pytest.approx(Time("2026-11-14T06:25:30.000", format="isot").mjd, abs=1e-9)
    assert np.array_equal(out_mask, mask) and values == {"SAT": 2, "CR": 8}


def test_read_fits_panel_without_mask_hdu(tmp_path):
    p = str(tmp_path / "nomask.fits")
    _write_panel(p, with_mask=False)
    img, wcs, mjd, mask, values = dn.read_fits_panel(p)
    assert img.shape == (40, 50) and mask is None and values is None


def test_prefetch_yields_errors_in_order(tmp_path):
    good = str(tmp_path / "good.fits")
    _write_panel(good)
    got = list(dn._prefetch([good, str(tmp_path / "missing.fits"), good], 2))
    assert [g[0] for g in got] == [0, 1, 2]
    assert got[0][1] is not None and got[0][2] is None
    assert got[1][1] is None and got[1][2].startswith("FileNotFoundError")
    assert got[2][1] is not None


def test_log_unreadable_appends_rows(tmp_path):
    p = str(tmp_path / "adcnn_dets.csv.unreadable")
    dn._log_unreadable(p, V1, 42, "OSError: bad, truncated")
    dn._log_unreadable(p, V2, 7, "KeyError: 'x'")
    with open(p, newline="") as f:
        assert list(csv.reader(f)) == [[str(V1), "42", "OSError: bad, truncated"], [str(V2), "7", "KeyError: 'x'"]]


def test_n_devices():
    assert dn.n_devices(0, True, 4, False) == 4
    assert dn.n_devices(2, True, 4, False) == 2
    assert dn.n_devices(4, True, 4, False) == 4          # the default job: --n-gpus 4 on gpu:4
    with pytest.raises(SystemExit):
        dn.n_devices(5, True, 4, False)
    with pytest.raises(SystemExit):
        dn.n_devices(1, False, 0, False)
    assert dn.n_devices(1, False, 0, True) == 1
    assert dn.n_devices(0, False, 0, True) == 1


def test_run_children_exit_codes():
    assert dn.run_children(sys.exit, [(0,), (3,)]) == [0, 3]


def test_main_requires_manifest_out_and_threshold():
    for argv in ([], ["--manifest", "m.csv"], ["--out", "o.csv"], ["--manifest", "m.csv", "--out", "o.csv"]):
        with pytest.raises(SystemExit) as e:
            dn.main(argv)
        assert e.value.code == 2, argv


def test_main_allow_partial_needs_no_threshold(tmp_path, monkeypatch):
    man = tmp_path / "manifest.csv"
    man.write_text(f"visit,detector\n{V1},1\n")
    seen = []
    monkeypatch.setattr(dn, "merge", lambda m, out, partial: seen.append((len(m), out, partial)) or 0)
    with pytest.raises(SystemExit) as e:
        dn.main(["--manifest", str(man), "--out", str(tmp_path / "o.csv"), "--allow-partial"])
    assert e.value.code == 0 and seen == [(1, str(tmp_path / "o.csv"), True)]


# ---------------------------------------------------------------- run_shard (model and CNN workers stubbed)

class _InlinePool:
    """ProcessPoolExecutor stand-in: runs each task in this process when it is submitted."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        f = concurrent.futures.Future()
        f.set_result(fn(*args))
        return f


def _cand(x, y, beta, length, length_raw):
    return pd.DataFrame({"x": [x], "y": [y], "beta": [beta], "length": [length], "length_raw": [length_raw],
                         "score": [0.9], "mf_snr": [6.0], "nn_pmax": [0.8], "m_elong": [3.0], "m_dipole": [0.1],
                         "m_neg_ratio": [0.0], "m_neg_blob": [0.0], "is_dipole": [False]})


def _stub_shard(monkeypatch, tmp_path):
    """Three panels (the second unreadable) on the panel of _write_panel, whose only masked pixel is
    SAT|CR at (x=7, y=5). Detection 0 is a 0.4 px point at (7, 6) with a 30 px raw length; detection 2
    a vertical 20 px trail from (7, 5) with a 0.5 px raw length. Swapping x/y or using length_raw
    moves either off the masked pixel."""
    import torch
    from ADCNN.inference import catalog, predict
    good = str(tmp_path / "panel.fits")
    _, mask = _write_panel(good)
    cands = {0: _cand(7.0, 6.0, 0.0, 0.4, 30.0), 2: _cand(7.0, 15.0, 90.0, 20.0, 0.5)}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")          # run_shard sets it; restored after the test
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: types.SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(predict, "predict_panel_overlap_3ch_full", lambda *a, **k: (None, None))
    results = iter((cands[0], cands[2]))                     # the readable panels, in submission order
    monkeypatch.setattr(catalog, "_worker", lambda args: next(results))
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InlinePool)
    rows = [dict(visit=V1, detector=1, band="r", fits_path=good),
            dict(visit=V2, detector=2, band="r", fits_path=str(tmp_path / "missing.fits")),
            dict(visit=V3, detector=3, band="i", fits_path=good)]
    out = tmp_path / "_shard_adcnn_dets_0.csv"
    args = (0, rows, "seg.pt", "cnn.pt", 0.43, 2, str(out), str(tmp_path / "adcnn_dets.csv.unreadable"), 2)
    return args, out, mask, cands


def test_run_shard_unreadable_not_done_and_mask_columns(tmp_path, monkeypatch):
    args, out, mask, cands = _stub_shard(monkeypatch, tmp_path)
    dn.run_shard(*args)
    assert Path(str(out) + ".done").read_text() == f"{V1},1\n{V3},3\n"
    with open(tmp_path / "adcnn_dets.csv.unreadable", newline="") as f:
        unread = list(csv.reader(f))
    assert [r[:2] for r in unread] == [[str(V2), "2"]] and unread[0][2].startswith("FileNotFoundError")
    cat = pd.read_csv(out)
    assert list(cat.columns) == dn.DET_COLUMNS and cat.visit.tolist() == [V1, V3]
    values = {"SAT": 2, "CR": 8}
    for row, c in zip(cat.itertuples(), (cands[0], cands[2])):
        want = mp.fractions(mask, values, c.x[0], c.y[0], c.beta[0], max(c.length[0], 0.0))
        assert want["m_SAT"] > 0 and want["m_CR"] == want["m_SAT"]
        for k, v in want.items():
            got = getattr(row, k)
            assert (np.isnan(got) and np.isnan(v)) or got == v, (k, got, v)
        assert row.len_db == max(c.length[0], 0.0)


def test_run_shard_logs_done_only_after_rows(tmp_path, monkeypatch):
    args, out, _, _ = _stub_shard(monkeypatch, tmp_path)
    real, calls = pd.DataFrame.to_csv, []

    def fail_second(self, *a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("killed while writing")
        return real(self, *a, **k)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_second)
    with pytest.raises(OSError):
        dn.run_shard(*args)
    assert Path(str(out) + ".done").read_text() == f"{V1},1\n"     # the unwritten panel is not done


# ---------------------------------------------------------------- merge coverage

def _rows(visit, detector, n, mjd):
    d = {c: np.full(n, 0.5) for c in dn.DET_COLUMNS}
    d.update(mjd=mjd, band="r", obscode="X05", visit=visit, detector=detector, x=np.arange(n) * 10.0,
             y=np.arange(n) * 3.0, score=np.linspace(0.5, 0.9, n), is_dipole=False)
    return pd.DataFrame(d)[dn.DET_COLUMNS]


def _setup_shards(tmp_path):
    """shard 0: panel (V1,1) done + 3 rows; shard 1: panel (V2,2) done + 2 rows (one duplicated) and
    rows of (V3,3) that were written but never logged done."""
    s0, s1 = tmp_path / "_shard_adcnn_dets_0.csv", tmp_path / "_shard_adcnn_dets_1.csv"
    _rows(V1, 1, 3, 61358.1).to_csv(s0, index=False)
    r2 = _rows(V2, 2, 2, 61358.0)
    pd.concat([r2, r2.iloc[:1], _rows(V3, 3, 4, 61358.2)]).to_csv(s1, index=False)
    (tmp_path / "_shard_adcnn_dets_0.csv.done").write_text(f"{V1},1\n")
    (tmp_path / "_shard_adcnn_dets_1.csv.done").write_text(f"{V2},2\n")
    return [s0, s1, tmp_path / "_shard_adcnn_dets_0.csv.done", tmp_path / "_shard_adcnn_dets_1.csv.done"]


def _man(*panels):
    return pd.DataFrame(dict(visit=[p[0] for p in panels], detector=[p[1] for p in panels]))


def test_merge_incomplete_returns_3_and_keeps_shards(tmp_path):
    files = _setup_shards(tmp_path)
    out = tmp_path / "adcnn_dets.csv"
    assert dn.merge(_man((V1, 1), (V2, 2), (V3, 3)), str(out)) == 3
    assert not out.exists() and all(f.exists() for f in files)


def test_merge_complete_writes_and_cleans(tmp_path):
    files = _setup_shards(tmp_path)
    out = tmp_path / "adcnn_dets.csv"
    assert dn.merge(_man((V1, 1), (V2, 2)), str(out)) == 0
    cat = pd.read_csv(out)
    assert list(cat.columns) == ["detid"] + dn.DET_COLUMNS
    assert len(cat) == 5 and list(cat.detid) == list(range(5))
    assert list(cat.visit[:2]) == [V2, V2]                   # sorted by mjd
    assert set(zip(cat.visit, cat.detector)) == {(V1, 1), (V2, 2)}
    assert not any(f.exists() for f in files)


def test_merge_allow_partial_keeps_shards(tmp_path):
    files = _setup_shards(tmp_path)
    out = tmp_path / "adcnn_dets.csv"
    assert dn.merge(_man((V1, 1), (V2, 2), (V3, 3)), str(out), allow_partial=True) == 0
    cat = pd.read_csv(out)
    assert set(zip(cat.visit, cat.detector)) == {(V1, 1), (V2, 2)} and len(cat) == 5
    assert all(f.exists() for f in files)


def _load_status():
    spec = importlib.util.spec_from_file_location("usdf_status_detect",
                                                  Path(__file__).resolve().parents[1] / "status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Killed(Exception):
    pass


def test_merge_cleanup_kill_never_completes_a_short_night(tmp_path, monkeypatch):
    """Kill the 100% merge after k of its unlinks (k = 0..all), then merge again as the retried detect
    job does. Whenever the driver then sees detection complete, every done panel's rows are there."""
    status = _load_status()
    real = Path.unlink
    man = _man((V1, 1), (V2, 2))
    for k in range(5):
        d = tmp_path / f"k{k}"
        d.mkdir()
        _setup_shards(d)
        man.to_csv(d / "manifest.csv", index=False)
        n = []

        def unlink(self, missing_ok=False):
            if len(n) == k:
                raise _Killed()
            n.append(self)
            real(self, missing_ok=missing_ok)
        monkeypatch.setattr(Path, "unlink", unlink)
        with contextlib.suppress(_Killed):
            dn.merge(man, str(d / "adcnn_dets.csv"))
        monkeypatch.setattr(Path, "unlink", real)
        dn.merge(man, str(d / "adcnn_dets.csv"))
        if status.coverage(d)["complete"]:
            cat = pd.read_csv(d / "adcnn_dets.csv")
            assert len(cat) == 5 and set(zip(cat.visit, cat.detector)) == {(V1, 1), (V2, 2)}, k


def test_merge_write_is_atomic(tmp_path, monkeypatch):
    files = _setup_shards(tmp_path)
    out = tmp_path / "adcnn_dets.csv"

    def torn(self, path, *a, **k):
        Path(path).write_text("detid,mjd\n0,")
        raise OSError("killed while writing")
    monkeypatch.setattr(pd.DataFrame, "to_csv", torn)
    with pytest.raises(OSError):
        dn.merge(_man((V1, 1), (V2, 2)), str(out))
    assert not out.exists() and all(f.exists() for f in files)


def test_merge_complete_without_detections_writes_header(tmp_path):
    (tmp_path / "_shard_adcnn_dets_0.csv.done").write_text(f"{V1},1\n")
    out = tmp_path / "adcnn_dets.csv"
    assert dn.merge(_man((V1, 1)), str(out)) == 0
    cat = pd.read_csv(out)
    assert len(cat) == 0 and list(cat.columns) == ["detid"] + dn.DET_COLUMNS
    assert not (tmp_path / "_shard_adcnn_dets_0.csv.done").exists()


def test_main_allow_partial_is_merge_only(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("--allow-partial must not run detection")
    monkeypatch.setattr(dn, "run_shard", refuse)
    monkeypatch.setattr(dn, "n_devices", refuse)
    _setup_shards(tmp_path)
    man = tmp_path / "manifest.csv"
    _man((V1, 1), (V2, 2), (V3, 3)).assign(fits_path="/nonexistent.fits", band="r").to_csv(man, index=False)
    out = tmp_path / "adcnn_dets.csv"
    with pytest.raises(SystemExit) as e:
        dn.main(["--manifest", str(man), "--out", str(out), "--allow-partial"])
    assert e.value.code == 0 and out.exists()
    with pytest.raises(SystemExit) as e:
        dn.main(["--manifest", str(man), "--out", str(tmp_path / "other.csv"), "--allow-partial"])
    assert e.value.code == 0 and len(pd.read_csv(tmp_path / "other.csv")) == 0


# ---------------------------------------------------------------- mf_trail_length

def test_mf_k_rederived():
    from ADCNN.inference.mf_trail_length import MF_K
    assert MF_K == 0.8956
