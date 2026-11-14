"""Tests for night.py, stack_tables.py and refcat.py (run in the LSST stack env).

Tests marked `requires_butler` read the embargo repo (night 20261113) and are skipped when the
Butler/S3 credentials are not set up.
"""
import json
import logging
import os
import types

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from astropy.time import Time

from ADCNN.pipelines.usdf_night import night, refcat, stack_tables as st

requires_butler = pytest.mark.skipif(night.credentials_problem() is not None,
                                     reason="no Butler/S3 credentials")

PROMPT_RUN = "LSSTCam/runs/prompt/20261113/ApPipe/pipelines-0d22056-config-8f017ea"

# one dia_source_unfiltered row of 20261113 (visit 2026111300041, detector 164) and Rubin's
# dia_source_detector values for it
UNF = {
    "id": np.int64(171207160173166615),
    "coord_ra": np.float64(5.699003210656764),
    "coord_dec": np.float64(-0.3280478763069687),
    "coord_raErr": np.float32(2.488721975169028e-07),
    "coord_decErr": np.float32(2.4574950430178433e-07),
    "coord_ra_dec_Cov": np.float32(-9.034751571011801e-15),
    "slot_Centroid_x": np.float64(1062.3138178403021),
    "slot_Centroid_y": np.float64(313.1590863353382),
    "slot_PsfFlux_instFlux": np.float64(3048.8451466960887),
    "slot_PsfFlux_instFluxErr": np.float64(233.6310882896293),
    "base_LocalPhotoCalib": np.float64(1.0),
    "is_negative": np.bool_(False),
    "ip_diffim_DipoleFit_classification": np.bool_(False),
    "base_LocalWcs_CDMatrix_1_1": np.float64(-7.157192918956979e-07),
    "base_LocalWcs_CDMatrix_1_2": np.float64(6.530783463584243e-07),
    "base_LocalWcs_CDMatrix_2_1": np.float64(6.540615920721067e-07),
    "base_LocalWcs_CDMatrix_2_2": np.float64(7.158170006805745e-07),
    "ext_trailedSources_Naive_ra": np.float64(326.52883088815383),
    "ext_trailedSources_Naive_dec": np.float64(-18.79575843717177),
    "ext_trailedSources_Naive_length": np.float64(5.901560821883996),
    "ext_trailedSources_Naive_lengthErr": np.float64(1.0081369251376804),
    "ext_trailedSources_Naive_angle": np.float64(-0.558618595387295),
    "ext_trailedSources_Naive_angleErr": np.float64(0.20288470386463564),
    "ext_trailedSources_Naive_x0": np.float64(1059.8223500425238),
    "ext_trailedSources_Naive_y0": np.float64(314.72183229726517),
    "ext_trailedSources_Naive_x1": np.float64(1064.826803305009),
    "ext_trailedSources_Naive_y1": np.float64(311.5939148306432),
    "ext_trailedSources_Naive_flux": np.float64(2958.9378298576944),
    "ext_trailedSources_Naive_fluxErr": np.float64(86.22460732831144),
    "ext_trailedSources_Naive_flag_edge": np.bool_(False),
    **{f"base_PixelFlags_flag_{s}": np.bool_(False) for s in st.PIXEL_FLAGS},
}
DET = {"diaSourceId": 171207160173166615, "x": 1062.3138427734375, "y": 313.1590881347656,
       "ra": 326.5288314021382, "dec": -18.79575879061898, "raErr": 1.425932532583829e-05,
       "decErr": 1.408040952810552e-05, "ra_dec_Cov": -2.9659340955445757e-11,
       "trailRa": 326.52883088815383, "trailDec": -18.79575843717177, "trailLength": 1.1798728704452515,
       "trailLengthErr": 0.2015523463487625, "trailAngle": -79.58123016357422,
       "trailAngleErr": 11.62443733215332, "psfFlux": 3048.84521484375, "psfFluxErr": 233.63108825683594,
       "trailFlux": 2958.937744140625, "trailFluxErr": 86.224609375, "isNegative": False, "isDipole": False,
       "trail_flag_edge": False}


# ---------------------------------------------------------------- fakes
class FakeRef:
    def __init__(self, visit, detector, run=PROMPT_RUN, dataset_type="future_difference_image", band="i"):
        self.dataId = {"visit": visit, "detector": detector, "band": band}
        self.run = run
        self.datasetType = types.SimpleNamespace(name=dataset_type)


class FakeButler:
    """chains: chain -> children; datasets: type -> refs (a missing type raises MissingDatasetTypeError)."""

    def __init__(self, chains=None, datasets=None, records=None, uris=None):
        self.chains, self.datasets = chains or {}, datasets or {}
        self.records, self.uris = records or {}, uris or {}
        self.calls = []
        self.collections = types.SimpleNamespace(query=self._query_collections)

    def _query_collections(self, name, flatten_chains=False):
        from lsst.daf.butler import MissingCollectionError
        assert flatten_chains
        if name not in self.chains:
            raise MissingCollectionError(name)
        return list(self.chains[name])

    def query_datasets(self, dataset_type, collections, where="", find_first=True, limit=-20000, explain=True):
        from lsst.daf.butler import MissingDatasetTypeError
        assert find_first and limit is None and not explain
        self.calls.append((dataset_type, list(collections), where))
        if dataset_type not in self.datasets:
            raise MissingDatasetTypeError(dataset_type)
        found = {}
        for c in collections:
            for r in self.datasets[dataset_type]:
                if r.run == c:
                    found.setdefault((r.dataId["visit"], r.dataId["detector"]), r)
        return list(found.values())

    def query_dimension_records(self, element, where="", limit=-20000, explain=True):
        assert limit is None and not explain
        return self.records.get(element, [])

    def get_many_uris(self, refs):
        return {r: types.SimpleNamespace(primaryURI=self.uris[r]) for r in refs}


def visit_record(visit, begin_mjd, end_mjd, day_obs=20261113):
    span = types.SimpleNamespace(begin=Time(begin_mjd, format="mjd", scale="tai"),
                                 end=Time(end_mjd, format="mjd", scale="tai"))
    return types.SimpleNamespace(id=visit, day_obs=day_obs, timespan=span)


# ---------------------------------------------------------------- night.py
def test_normalize_day():
    assert night.normalize_day("20261113") == "20261113"
    assert night.normalize_day("2026-11-13") == "20261113"
    assert night.normalize_day(20261113) == "20261113"
    for bad in ("2026-1113", "261113", "2026/11/13", "20261332", "2026-02-30", ""):
        with pytest.raises(ValueError):
            night.normalize_day(bad)


def test_chain_where_and_prompt_runs():
    assert night.chain_name("20261113") == "LSSTCam/runs/prompt-20261113"
    assert night.day_where("20261113") == "instrument='LSSTCam' AND visit.day_obs=20261113"
    kids = ["LSSTCam/runs/daytimeAP/20260713/20260714T123323Z",
            "LSSTCam/runs/prompt/20260713/ApPipe/pipelines-69e0100-config-8f017ea",
            "LSSTCam/runs/prompt/20260713/NoPipeline/pipelines-69e0100-config-8f017ea"]
    assert night.prompt_runs(kids) == kids[1:]


def test_credentials_problem(tmp_path):
    ini = tmp_path / "aws.ini"
    env = {"HOME": str(tmp_path)}
    assert "DAF_BUTLER_REPOSITORY_INDEX" in night.credentials_problem(env)
    env["DAF_BUTLER_REPOSITORY_INDEX"] = "/x/repos.yaml"
    assert "not found" in night.credentials_problem(env)          # default $HOME/.lsst path absent
    env["AWS_SHARED_CREDENTIALS_FILE"] = str(ini)
    assert "not found" in night.credentials_problem(env)
    ini.write_text("[default]\naws_access_key_id = k\n")
    assert "no [embargo] profile" in night.credentials_problem(env)
    ini.write_text("[default]\nx = 1\n [embargo] \naws_access_key_id = k%\n")
    assert night.credentials_problem(env) is None
    (tmp_path / ".lsst").mkdir()
    (tmp_path / ".lsst" / "aws-credentials.ini").write_text("[embargo]\n")
    env["AWS_SHARED_CREDENTIALS_FILE"] = ""                       # empty -> default path
    assert night.credentials_problem(env) is None


def test_open_butler_failure_is_one_line(monkeypatch):
    import lsst.daf.butler as dafb

    def boom(repo):
        raise RuntimeError("fe_sendauth: no password supplied\nsecond line")
    monkeypatch.setattr(dafb, "Butler", boom)
    log = logging.getLogger("lsst.daf.butler")
    before = log.level
    with pytest.raises(night.CredentialError) as e:
        night.open_butler("embargo")
    msg = str(e.value)
    assert "\n" not in msg and "embargo" in msg and "fe_sendauth" in msg
    assert log.level == before


def test_find_panels_prompt_children_only():
    """A daytimeAP child earlier in the chain must not shadow prompt panels."""
    day, ap, pr = "20260713", "LSSTCam/runs/daytimeAP/20260713/x", "LSSTCam/runs/prompt/20260713/ApPipe/p"
    di = [FakeRef(1, 1, ap, "difference_image"), FakeRef(1, 1, pr, "difference_image"),
          FakeRef(1, 2, pr, "difference_image")]
    b = FakeButler(chains={night.chain_name(day): [ap, pr]},
                   datasets={"future_difference_image": [], "difference_image": di})
    dataset_type, refs = night.find_panels(b, day)
    assert dataset_type == "difference_image"
    assert sorted((r.dataId["visit"], r.dataId["detector"], r.run) for r in refs) == [(1, 1, pr), (1, 2, pr)]
    assert all(c[1] == [pr] and c[2] == night.day_where(day) for c in b.calls)


def test_find_panels_states():
    day = "20261113"
    chain = night.chain_name(day)
    assert night.find_panels(FakeButler(), day) == (None, [])                    # missing chain
    assert night.find_panels(FakeButler(chains={chain: []}), day) == (None, [])  # chain without children
    empty = FakeButler(chains={chain: [PROMPT_RUN]}, datasets={"future_difference_image": []})
    assert night.find_panels(empty, day) == (None, [])                          # no diffims; no legacy type
    both = FakeButler(chains={chain: [PROMPT_RUN]},
                      datasets={"future_difference_image": [FakeRef(1, 1)],
                                "difference_image": [FakeRef(1, 1, dataset_type="difference_image")]})
    dataset_type, refs = night.find_panels(both, day)
    assert dataset_type == "future_difference_image" and len(refs) == 1


def test_resolve_falls_through_to_main():
    day = "20260713"
    embargo = FakeButler(chains={night.chain_name(day): []})
    main = FakeButler(chains={night.chain_name(day): [PROMPT_RUN]},
                      datasets={"difference_image": [FakeRef(5, 9, dataset_type="difference_image")]})
    opened = []

    def open_fn(repo):
        opened.append(repo)
        return {"embargo": embargo, "main": main}[repo]
    res = night.resolve(day, open_fn)
    assert opened == ["embargo", "main"] and res["repo"] == "main" and res["butler"] is main
    assert res["dataset_type"] == "difference_image" and len(res["refs"]) == 1
    none = night.resolve(day, lambda repo: FakeButler())
    assert none == {"repo": None, "butler": None, "dataset_type": None, "refs": []}

    def denied(repo):
        raise night.CredentialError("no access")
    with pytest.raises(night.CredentialError):
        night.resolve(day, denied)


def test_summary_and_sorted_refs():
    refs = [FakeRef(3, 1), FakeRef(2, 7), FakeRef(2, 5, run=PROMPT_RUN + "x")]
    s = night.summary("20261113", {"repo": "embargo", "dataset_type": "future_difference_image", "refs": refs})
    assert s == {"repo": "embargo", "collection": "LSSTCam/runs/prompt-20261113",
                 "dataset_type": "future_difference_image", "runs": sorted([PROMPT_RUN, PROMPT_RUN + "x"]),
                 "n_panels": 3, "n_visits": 2}
    assert [(r.dataId["visit"], r.dataId["detector"]) for r in night.sorted_refs(refs)] == [(2, 5), (2, 7), (3, 1)]
    assert len(night.sorted_refs(refs, 2)) == 2 and len(night.sorted_refs(refs, None)) == 3


def test_visit_midpoints():
    b = FakeButler(records={"visit": [visit_record(7, 61358.0, 61358.0 + 30 / 86400)]})
    mid = night.visit_midpoints(b, "20261113")
    assert set(mid) == {7} and mid[7][1] == 20261113
    assert abs(mid[7][0] - (61358.0 + 15 / 86400)) * 86400e3 < 1e-3


def test_uri_string():
    from lsst.resources import ResourcePath
    s3 = "s3://embargo@rubin-summit-users/LSSTCam/runs/prompt/x/difference_image.fits"
    assert night.uri_string(ResourcePath(s3)) == s3
    assert night.uri_string(ResourcePath("/sdf/data/rubin/repo/x.fits")) == "/sdf/data/rubin/repo/x.fits"
    assert night.uri_string(ResourcePath("file:///sdf/data/x.fits")) == "/sdf/data/x.fits"


def _manifest_butler():
    from lsst.resources import ResourcePath
    refs = [FakeRef(11, 4, band="r"), FakeRef(10, 2), FakeRef(10, 1)]
    uris = {r: ResourcePath(f"s3://embargo@b/{r.dataId['visit']}_{r.dataId['detector']}.fits") for r in refs}
    recs = [visit_record(10, 61358.0, 61358.0002), visit_record(11, 61358.1, 61358.1004)]
    return FakeButler(records={"visit": recs}, uris=uris), refs


def test_manifest_frame():
    b, refs = _manifest_butler()
    df = night.manifest_frame(b, "embargo", refs, "20261113")
    assert list(df.columns) == night.MANIFEST_COLUMNS
    assert list(zip(df.visit, df.detector)) == [(10, 1), (10, 2), (11, 4)]
    assert df.fits_path.tolist() == ["s3://embargo@b/10_1.fits", "s3://embargo@b/10_2.fits", "s3://embargo@b/11_4.fits"]
    assert df.band.tolist() == ["i", "i", "r"] and (df.repo == "embargo").all() and (df.day_obs == 20261113).all()
    assert df.mjd_tai.iloc[2] == pytest.approx(61358.1002, abs=1e-9)
    assert (df.dataset_type == "future_difference_image").all() and (df.run == PROMPT_RUN).all()
    with pytest.raises(RuntimeError):
        night.manifest_frame(b, "embargo", refs + [refs[0]], "20261113")


def test_update_info_and_write_atomic(tmp_path):
    p = tmp_path / "prep_info.json"
    night.update_info(p, "a", {"x": 1})
    night.update_info(p, "b", {"y": [1, 2]})
    night.update_info(p, "a", {"x": 3})
    assert json.loads(p.read_text()) == {"a": {"x": 3}, "b": {"y": [1, 2]}}
    df = pd.DataFrame({"a": [1, 2], "b": [0.1, np.nan]})
    night.write_atomic(df, tmp_path / "w" / "t.csv")
    night.write_atomic(df, tmp_path / "w" / "t.parquet")
    assert pd.read_csv(tmp_path / "w" / "t.csv").equals(df)
    assert pd.read_parquet(tmp_path / "w" / "t.parquet").equals(df)
    assert sorted(f.name for f in (tmp_path / "w").iterdir()) == ["t.csv", "t.parquet"]


def test_lsst_weekly(monkeypatch):
    monkeypatch.setenv("LSST_DISTRIB_DIR", "/cvmfs/x/lsst_distrib/w_2026_46/conda/envs/e/share/eups/Linux64/lsst_distrib/g0")
    assert night.lsst_weekly() == "w_2026_46"
    monkeypatch.setenv("LSST_DISTRIB_DIR", "/opt/lsst")
    assert night.lsst_weekly() is None


def test_main_check_and_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(night, "credentials_problem", lambda env=None: None)
    monkeypatch.setattr(night, "resolve", lambda day: {"repo": "embargo", "butler": None,
                                                       "dataset_type": "future_difference_image",
                                                       "refs": [FakeRef(1, 2), FakeRef(1, 3)]})
    assert night.main(["check", "2026-11-13"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_panels"] == 2 and out["n_visits"] == 1 and out["collection"] == "LSSTCam/runs/prompt-20261113"
    assert night.main(["check", "2026-13-01"]) == 1
    capsys.readouterr()
    monkeypatch.setattr(night, "credentials_problem", lambda env=None: "S3 credentials file x not found")
    assert night.main(["check", "20261113"]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "not found" in err


def test_main_manifest(monkeypatch, tmp_path):
    b, refs = _manifest_butler()
    monkeypatch.setattr(night, "credentials_problem", lambda env=None: None)
    monkeypatch.setattr(night, "resolve", lambda day: {"repo": "embargo", "butler": b,
                                                       "dataset_type": "future_difference_image", "refs": refs})
    out = tmp_path / "work" / "manifest.csv"
    assert night.main(["manifest", "20261113", "--out", str(out), "--limit", "2"]) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == night.MANIFEST_COLUMNS and len(df) == 2
    info = json.loads((tmp_path / "work" / "prep_info.json").read_text())["manifest"]
    assert info["n_panels"] == 2 and info["limit"] == 2 and info["night"] == "20261113"
    monkeypatch.setattr(night, "resolve", lambda day: {"repo": None, "butler": None, "dataset_type": None,
                                                       "refs": []})
    assert night.main(["manifest", "20261201", "--out", str(out)]) == 0
    assert pd.read_csv(out).empty and list(pd.read_csv(out).columns) == night.MANIFEST_COLUMNS


# ---------------------------------------------------------------- stack_tables.py: Rubin definitions
def _as_arrays(row, n=1):
    return {k: np.full(n, v, dtype=np.asarray(v).dtype) for k, v in row.items()}


def test_dpdd_real_row_is_bit_identical():
    out = st.dpdd(_as_arrays(UNF))
    for ours, theirs in [("x", "x"), ("y", "y")]:
        assert np.float32(out[ours][0]) == np.float32(DET[theirs])
    for ours, theirs in [("psf_ra", "ra"), ("psf_dec", "dec"), ("ra_err", "raErr"), ("dec_err", "decErr"),
                         ("ra_dec_cov", "ra_dec_Cov"), ("trail_ra", "trailRa"), ("trail_dec", "trailDec"),
                         ("trail_length_arcsec", "trailLength"), ("trail_length_err_arcsec", "trailLengthErr"),
                         ("trail_pa_deg", "trailAngle"), ("trail_pa_err_deg", "trailAngleErr"),
                         ("psf_flux_njy", "psfFlux"), ("psf_flux_err_njy", "psfFluxErr"),
                         ("trail_flux_njy", "trailFlux"), ("trail_flux_err_njy", "trailFluxErr")]:
        assert out[ours][0] == DET[theirs], ours
    assert out["is_negative"][0] == 0 and out["rubin_is_dipole"][0] == 0 and out["trail_flag_edge"][0] == 0
    assert out["trail_length_px"][0] == UNF["ext_trailedSources_Naive_length"]
    assert out["trail_angle_img"][0] == UNF["ext_trailedSources_Naive_angle"]


def _synthetic(n=400, seed=3):
    rng = np.random.default_rng(seed)
    rot = rng.uniform(-np.pi, np.pi, n)
    s = np.radians(0.2 / 3600) * rng.uniform(0.98, 1.02, n)
    flip = rng.choice([-1.0, 1.0], n)
    d = {
        "coord_ra": rng.uniform(0, 2 * np.pi, n), "coord_dec": rng.uniform(-1.4, 1.4, n),
        "coord_raErr": rng.uniform(1e-8, 1e-6, n).astype(np.float32),
        "coord_decErr": rng.uniform(1e-8, 1e-6, n).astype(np.float32),
        "coord_ra_dec_Cov": rng.normal(0, 1e-14, n).astype(np.float32),
        "slot_Centroid_x": rng.uniform(0, 4096, n), "slot_Centroid_y": rng.uniform(0, 4004, n),
        "slot_PsfFlux_instFlux": rng.normal(0, 3000, n), "slot_PsfFlux_instFluxErr": rng.uniform(10, 300, n),
        "base_LocalPhotoCalib": rng.uniform(0.5, 2.0, n),
        "is_negative": rng.random(n) < 0.3, "ip_diffim_DipoleFit_classification": rng.random(n) < 0.4,
        "base_LocalWcs_CDMatrix_1_1": flip * s * np.cos(rot), "base_LocalWcs_CDMatrix_1_2": s * np.sin(rot),
        "base_LocalWcs_CDMatrix_2_1": flip * s * np.sin(rot), "base_LocalWcs_CDMatrix_2_2": -s * np.cos(rot),
        "ext_trailedSources_Naive_ra": rng.uniform(0, 360, n), "ext_trailedSources_Naive_dec": rng.uniform(-80, 80, n),
        "ext_trailedSources_Naive_length": rng.uniform(0, 60, n),
        "ext_trailedSources_Naive_lengthErr": rng.uniform(0, 5, n),
        "ext_trailedSources_Naive_angle": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "ext_trailedSources_Naive_angleErr": rng.uniform(0, 0.5, n),
        "ext_trailedSources_Naive_flux": rng.normal(0, 3000, n), "ext_trailedSources_Naive_fluxErr": rng.uniform(10, 300, n),
        "ext_trailedSources_Naive_flag_edge": rng.random(n) < 0.1,
        **{f"base_PixelFlags_flag_{s_}": rng.random(n) < 0.2 for s_ in st.PIXEL_FLAGS},
    }
    for c in ("x0", "y0", "x1", "y1"):
        d["ext_trailedSources_Naive_" + c] = rng.uniform(0, 4000, n)
    for c in ("length", "angle", "flux", "lengthErr"):
        d["ext_trailedSources_Naive_" + c][::17] = np.nan
    return d


def test_dpdd_matches_rubin_functors():
    """Our columns vs Rubin's functor classes built from the installed DiaSource.yaml."""
    import yaml
    from lsst.pipe.tasks.functors import CompositeFunctor, init_fromDict
    with open(os.path.expandvars("$AP_ASSOCIATION_DIR/data/DiaSource.yaml")) as f:
        defs = yaml.safe_load(f)["funcs"]
    pairs = {"x": "x", "y": "y", "psf_ra": "ra", "psf_dec": "dec", "ra_err": "raErr", "dec_err": "decErr",
             "ra_dec_cov": "ra_dec_Cov", "trail_ra": "trailRa", "trail_dec": "trailDec",
             "trail_length_arcsec": "trailLength", "trail_length_err_arcsec": "trailLengthErr",
             "trail_pa_deg": "trailAngle", "trail_pa_err_deg": "trailAngleErr", "psf_flux_njy": "psfFlux",
             "psf_flux_err_njy": "psfFluxErr", "trail_flux_njy": "trailFlux", "trail_flux_err_njy": "trailFluxErr",
             "is_negative": "isNegative", "rubin_is_dipole": "isDipole"}
    funcs = CompositeFunctor({k: init_fromDict(defs[k], name=k) for k in pairs.values()})
    d = _synthetic()
    rubin = funcs(pd.DataFrame(d))
    ours = st.dpdd(d)
    for o, t in pairs.items():
        r = rubin[t].to_numpy()
        if o in ("x", "y", "psf_ra", "psf_dec", "trail_ra", "trail_dec", "is_negative", "rubin_is_dipole"):
            np.testing.assert_array_equal(ours[o], r.astype(ours[o].dtype), err_msg=o)
        else:   # the APDB schema stores these as float32
            np.testing.assert_array_equal(ours[o], r.astype(np.float32).astype(np.float64), err_msg=o)
    assert np.isnan(ours["trail_pa_deg"][::17]).all() and np.isnan(ours["trail_length_arcsec"][::17]).all()


def test_wrap_matches_lsst_geom():
    import lsst.geom as geom
    v = np.array([-7.0, -2 * np.pi, -np.pi, -1e-17, 0.0, 1e-17, np.pi, 2 * np.pi - 1e-16, 2 * np.pi, 9.5, -3.2])
    np.testing.assert_array_equal(st._wrap(v), [geom.Angle(x).wrap().asRadians() for x in v])
    np.testing.assert_array_equal(st._wrap_ctr(v), [geom.Angle(x).wrapCtr().asRadians() for x in v])


def test_position_angle_and_pixel_scale():
    s = np.radians(0.2 / 3600)
    # x towards -RA (east-left image), y towards +Dec: theta=0 points west (PA -90), theta=90deg north (PA 0)
    cd = (-s, 0.0, 0.0, s)
    assert st.pixel_scale_arcsec(*cd) == pytest.approx(0.2)
    pa = st.position_angle_deg(np.array([0.0, np.pi / 2, np.pi / 4]), *cd)
    np.testing.assert_allclose(pa, [-90.0, 0.0, -45.0], atol=1e-9)
    assert np.isnan(st.position_angle_deg(np.array([np.nan]), *cd)).all()
    assert st._f32(np.array([0.1]))[0] == np.float64(np.float32(0.1))


def test_pixel_flags_match_installed_flag_map():
    import yaml
    d = os.path.expandvars("$AP_ASSOCIATION_DIR/data")
    with open(f"{d}/association-flag-map.yaml") as f:
        table = next(t for t in yaml.safe_load_all(f) if t["tableName"] == "DiaSource")
    with open(f"{d}/flag-rename-rules.yaml") as f:
        rules = yaml.safe_load(f)["flag_rename_rules"]
    names = [b["name"] for b in table["columns"][0]["bitList"] if b["name"].startswith("base_PixelFlags_flag_")]
    renamed = set()
    for n in names:
        for old, new in rules:
            if n.startswith(old):
                n = n.replace(old, new)
        renamed.add(n)
    assert renamed == {f"pixelFlags_{s}" for s in st.PIXEL_FLAGS}
    assert len(st.PIXEL_FLAGS) == len(set(st.PIXEL_FLAGS)) == 20


# ---------------------------------------------------------------- stack_tables.py: panel assembly
def _panel_inputs():
    n = 7
    d = _synthetic(n, seed=11)
    d["id"] = np.arange(100, 100 + n, dtype=np.int64)
    d["sky_source"] = np.array([0, 1, 0, 0, 1, 0, 0], bool)
    d["parent"] = np.zeros(n, np.int64)
    d["reliability"] = np.full(n, np.nan, np.float32)
    d["glint_trail"] = np.array([0, 1, 1, 0, 0, 1, 0], bool)   # kept rows 100,102,103,105,106 -> 0,1,0,1,0
    unf = Table(d)
    ours = st.dpdd({k: v[~d["sky_source"]] for k, v in d.items()})
    keep_ids = d["id"][~d["sky_source"]]                     # 100, 102, 103, 105, 106
    det_rows = [3, 0, 4]                                       # ids 105, 100, 106 (shuffled)
    det = pd.DataFrame({"diaSourceId": keep_ids[det_rows], "reliability": np.float32([0.9, 0.6, 0.75]),
                        "glint_trail": [True, False, False], "band": "i", "midpointMjdTai": 61358.5})
    for o, (t, _) in st.COMPARE.items():
        if t not in det:
            det[t] = np.asarray(ours[o])[det_rows]
    apdb = Table({"diaSourceId": keep_ids[[0, 4]], "diaObjectId": np.int64([7, 0]), "ssObjectId": np.int64([0, 42])})
    return unf, det, apdb


def test_panel_rows():
    unf, det, apdb = _panel_inputs()
    df, stats, counts = st.panel_rows(unf, det, apdb, 2026111300041, 164, "i", 61358.5)
    assert list(df.columns) == st.COLUMNS
    assert df.diaSourceId.tolist() == [100, 102, 103, 105, 106]
    assert df.in_detector.tolist() == [1, 0, 0, 1, 1] and df.in_apdb.tolist() == [1, 0, 0, 0, 1]
    np.testing.assert_array_equal(df.reliability.to_numpy(), [np.float32(0.6), np.nan, np.nan, np.float32(0.9),
                                                              np.float32(0.75)])
    np.testing.assert_array_equal(df.glint_trail.to_numpy(), [0.0, 1.0, 0.0, 1.0, 0.0])
    assert df.dia_object_id.tolist() == [7, 0, 0, 0, 0] and df.ss_object_id.tolist() == [0, 0, 0, 0, 42]
    assert (df.visit == 2026111300041).all() and (df.detector == 164).all() and (df.band == "i").all()
    assert df.diaSourceId.dtype == np.int64 and df.in_detector.dtype == np.int8 and df.mjd_tai.dtype == np.float64
    assert counts == {"rows_unfiltered": 7, "rows_sky": 2, "rows": 5, "in_detector": 3, "in_apdb": 2,
                      "detector_rows": 3, "apdb_rows": 2, "detector_not_in_rows": 0, "apdb_not_in_detector": 0,
                      "rows_with_parent": 0}
    assert all(v[3] == 0 and v[0] == 3 for v in stats.values()), {k: v for k, v in stats.items() if v[3]}
    old_schema = det.drop(columns=["pixelFlags_spike"])        # detector tables before the spike flag
    df1, stats1, _ = st.panel_rows(unf, old_schema, apdb, 1, 2, "i", 61358.5)
    assert set(stats1) == set(st.COMPARE) - {"pixelFlags_spike"} and df1.pixelFlags_spike.equals(df.pixelFlags_spike)
    df0, stats0, counts0 = st.panel_rows(unf, None, None, 1, 2, "r", 61358.5)
    assert (df0.in_detector == 0).all() and df0.reliability.isna().all() and (df0.ss_object_id == 0).all()
    assert stats0 == {} and counts0["detector_rows"] == 0


def test_compare_and_merge_stats():
    unf, det, apdb = _panel_inputs()
    pa = st.panel_rows(unf, det, apdb, 1, 1, "i", 61358.5)[0].set_index("diaSourceId").trail_pa_deg
    det.loc[0, "trailAngle"] = pa[105] + 360.0            # same direction: must not count as a difference
    det.loc[2, "trailAngle"] = 179.99                     # id 106
    det.loc[1, "psfFlux"] = np.nan
    det.loc[2, "isDipole"] = 1 - det.loc[2, "isDipole"]
    det.loc[0, "midpointMjdTai"] = 61358.5 + 2e-3 / 86400
    df, stats, _ = st.panel_rows(unf, det, apdb, 1, 1, "i", 61358.5)
    assert stats["trail_pa_deg"][3] == 1
    assert stats["trail_pa_deg"][4] == pytest.approx(abs((pa[106] - 179.99 + 180) % 360 - 180), rel=1e-6)
    assert stats["psf_flux_njy"][1:4] == [1, 1, 1]
    assert stats["rubin_is_dipole"][2:4] == [1, 1]
    assert stats["mjd_tai"][3] == 1 and stats["mjd_tai"][4] * 86400e3 == pytest.approx(2.0, rel=1e-3)
    tot = {}
    st.merge_stats(tot, stats)
    st.merge_stats(tot, stats)
    assert tot["psf_flux_njy"][:4] == [6, 2, 2, 2] and tot["trail_pa_deg"][4] == stats["trail_pa_deg"][4]


def test_table_refs():
    from lsst.daf.butler import MissingDatasetTypeError
    other = PROMPT_RUN.replace("0d22056", "e007bd1")
    b = FakeButler(datasets={"dia_source_unfiltered": [FakeRef(1, 1), FakeRef(1, 2, run=other)],
                             "dia_source_detector": [FakeRef(1, 1)]})
    refs = st.table_refs(b, "20261113", [PROMPT_RUN, other])
    assert set(refs["dia_source_unfiltered"]) == {(PROMPT_RUN, 1, 1), (other, 1, 2)}
    assert set(refs["dia_source_detector"]) == {(PROMPT_RUN, 1, 1)} and refs["dia_source_apdb"] == {}
    assert all(len(c[1]) == 1 for c in b.calls)
    assert issubclass(MissingDatasetTypeError, Exception)


def test_panel_worker(monkeypatch):
    unf, det, apdb = _panel_inputs()

    class Cat:
        def asAstropy(self):
            return unf

    class B:
        def get(self, ref):
            if ref == "bad":
                raise OSError("S3 read failed\nmore")
            return {"u": Cat(), "d": det, "a": apdb}[ref]
    monkeypatch.setattr(st, "_BUTLER", B())
    key, df, stats, counts, err = st._panel((1, 2, "i", 61358.5, ("u", "d", "a")))
    assert key == (1, 2) and err is None and len(df) == 5 and counts["in_apdb"] == 2
    key, df, stats, counts, err = st._panel((1, 2, "i", 61358.5, ("u", "bad", None)))
    assert df is None and err == "dia_source_detector: OSError: S3 read failed"


def test_build_retries_failed_reads_then_refuses(monkeypatch, tmp_path):
    """A failed table read is retried once; a panel that stays unreadable fails the step: prep_info.json
    lists it and stack_sources.parquet is not written, so a rerun of prep retries the step."""
    unf, det, apdb = _panel_inputs()
    failures = {}

    class Cat:
        def __init__(self, t):
            self.t = t

        def asAstropy(self):
            return self.t

    class B:
        def get(self, ref):
            kind, off = ref
            if failures.get(kind, 0) > 0:
                failures[kind] -= 1
                raise OSError("S3 read timed out")
            if kind == "u":
                t = unf.copy()
                t["id"] = t["id"] + off
                return Cat(t)
            t = (det if kind.startswith("d") else apdb).copy()
            t["diaSourceId"] = t["diaSourceId"] + off
            return t

    class Pool:
        """In-process ProcessPoolExecutor (the initializer opens the fake Butler)."""

        def __init__(self, max_workers, mp_context, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, tasks, chunksize=1):
            return map(fn, tasks)

    def run(det_kind, n_fail):
        failures.clear()
        failures[det_kind] = n_fail
        work = tmp_path / det_kind / "work"
        work.mkdir(parents=True)
        man = pd.DataFrame({"visit": [1, 1], "detector": [2, 3], "band": "i", "day_obs": 20261113,
                            "mjd_tai": 61358.5, "fits_path": "x", "repo": "embargo",
                            "dataset_type": "future_difference_image", "run": PROMPT_RUN})
        man.to_csv(work / "manifest.csv", index=False)
        refs = {"dia_source_unfiltered": {(PROMPT_RUN, 1, 2): ("u", 0), (PROMPT_RUN, 1, 3): ("u", 1000)},
                "dia_source_detector": {(PROMPT_RUN, 1, 2): ("d", 0), (PROMPT_RUN, 1, 3): (det_kind, 1000)},
                "dia_source_apdb": {(PROMPT_RUN, 1, 2): ("a", 0), (PROMPT_RUN, 1, 3): ("a", 1000)}}
        monkeypatch.setattr(st, "table_refs", lambda b, day, runs: refs)
        return work, lambda: st.build("20261113", work / "manifest.csv", work / "stack_sources.parquet", 2)

    monkeypatch.setattr(st, "_BUTLER", None)
    monkeypatch.setattr(st, "open_butler", lambda repo: B())
    monkeypatch.setattr(st, "ProcessPoolExecutor", Pool)
    work, build = run("d_flaky", 1)                                        # one timeout, then the read works
    info = build()
    assert info["panels_unreadable"] == [] and len(pd.read_parquet(work / "stack_sources.parquet")) == 10
    work, build = run("d_bad", 2)                                          # fails again on the retry
    with pytest.raises(RuntimeError, match="1 panels unreadable"):
        build()
    assert not (work / "stack_sources.parquet").exists()
    info = json.loads((work / "prep_info.json").read_text())["stack_tables"]
    assert info["panels_unreadable"] == [[1, 3, "dia_source_detector: OSError: S3 read timed out"]]


def test_stack_tables_main_credential_exit(monkeypatch):
    def denied(*a):
        raise night.CredentialError("no access")
    monkeypatch.setattr(st, "build", denied)
    assert st.main(["2026-11-13", "--manifest", "m.csv", "--out", "o.parquet"]) == 2


# ---------------------------------------------------------------- refcat.py
def _square(ra, dec, half_deg):
    pts = [(ra - half_deg, dec - half_deg), (ra + half_deg, dec - half_deg), (ra + half_deg, dec + half_deg),
           (ra - half_deg, dec + half_deg)]
    return refcat.unit_vectors(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))


def test_g_mag_and_unit_vectors():
    m = refcat.g_mag([3631e9, 1e3, 0.0, -5.0, np.nan])
    assert m[0] == pytest.approx(0.0, abs=1e-3) and m[1] == pytest.approx(23.9)
    assert np.isnan(m[2:]).all()
    u = refcat.unit_vectors(np.array([0.0, 90.0, 10.0]), np.array([0.0, 0.0, 90.0]))
    np.testing.assert_allclose(u, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-15)


def test_inside_and_cap():
    verts = _square(10.0, 0.0, 0.1)
    m = np.radians(10 / 3600)
    pts = refcat.unit_vectors(np.array([10.0, 10.1 + 5 / 3600, 10.1 + 15 / 3600, 10.0]),
                              np.array([0.0, 0.0, 0.0, -0.1 - 9 / 3600]))
    assert refcat.inside(verts, pts, m).tolist() == [True, True, False, True]
    assert refcat.inside(verts, pts, 0.0).tolist() == [True, False, False, False]
    assert refcat.inside(verts[::-1], pts, m).tolist() == [True, True, False, True]
    c, r = refcat.cap(verts, m)
    assert np.degrees(r) == pytest.approx(0.1 * np.sqrt(2) + 10 / 3600, rel=1e-3)
    np.testing.assert_allclose(c, refcat.unit_vectors(np.array([10.0]), np.array([0.0]))[0], atol=1e-12)


def test_footprint_mask_matches_brute_force():
    rng = np.random.default_rng(5)
    polys = [_square(10.0, 0.0, 0.1), _square(10.15, 0.05, 0.08), _square(200.0, -60.0, 0.1)]
    ra = np.concatenate([rng.uniform(9.8, 10.4, 4000), rng.uniform(199.7, 200.3, 2000)])
    dec = np.concatenate([rng.uniform(-0.2, 0.25, 4000), rng.uniform(-60.2, -59.8, 2000)])
    pts = refcat.unit_vectors(ra, dec)
    m = np.radians(10 / 3600)
    brute = np.any([refcat.inside(p, pts, m) for p in polys], axis=0)
    got = refcat.footprint_mask(pts, polys, m)
    assert brute.sum() > 1000 and (got == brute).all()
    assert refcat.footprint_mask(np.zeros((0, 3)), polys, m).size == 0


def test_htm_shards_cover_polygon():
    import lsst.sphgeom as sg
    verts = _square(150.0, 2.0, 0.11)
    shards = refcat.htm_shards([verts], np.radians(10 / 3600))
    pix = sg.HtmPixelization(7)
    for v in list(verts) + [verts.sum(0) / np.linalg.norm(verts.sum(0))]:
        assert pix.index(sg.UnitVector3d(*v)) in shards
    assert shards == sorted(set(shards)) and len(shards) < 20


def test_panel_polygons():
    import lsst.sphgeom as sg
    verts = _square(30.0, -10.0, 0.1)
    region = sg.ConvexPolygon([sg.UnitVector3d(*v) for v in verts])
    recs = [types.SimpleNamespace(visit=1, detector=2, region=region),
            types.SimpleNamespace(visit=1, detector=3, region=region)]
    polys = refcat.panel_polygons(FakeButler(records={"visit_detector_region": recs}), "20261113", [(1, 2), (9, 9)])
    assert set(polys) == {(1, 2)} and polys[(1, 2)].shape == (4, 3)
    assert refcat.inside(polys[(1, 2)], verts.sum(0)[None, :] / np.linalg.norm(verts.sum(0)), 0.0).all()


def test_shard_worker(monkeypatch):
    class Cat(dict):
        def isContiguous(self):
            return True
    cat = Cat({"phot_g_mean_flux": np.array([1e3 * 10 ** (0.4 * 3.0), 1e3, -1.0]),   # G = 20.9, 23.9, NaN
               "coord_ra": np.radians([10.0, 11.0, 12.0]), "coord_dec": np.radians([1.0, 2.0, 3.0])})
    monkeypatch.setattr(refcat, "_BUTLER", types.SimpleNamespace(get=lambda ref: cat))
    ra, dec, mag = refcat._shard("ref")
    np.testing.assert_allclose(ra, [10.0])
    np.testing.assert_allclose(dec, [1.0])
    assert mag[0] == pytest.approx(20.9)


def test_refcat_main_credential_exit(monkeypatch):
    def denied(*a):
        raise night.CredentialError("no access")
    monkeypatch.setattr(refcat, "build", denied)
    assert refcat.main(["--manifest", "m.csv", "--out", "o.parquet"]) == 2


# ---------------------------------------------------------------- real data (embargo, 20261113)
@requires_butler
def test_check_real_nights(capsys):
    assert night.main(["check", "20261113"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["repo"] == "embargo" and s["dataset_type"] == "future_difference_image"
    assert s["n_panels"] == 4919 and s["n_visits"] == 46 and s["runs"] == [PROMPT_RUN]
    assert night.main(["check", "2026-12-01"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["n_panels"] == 0 and s["repo"] is None


@pytest.fixture(scope="module")
def real_manifest(tmp_path_factory):
    """Two 20261113 panels: one regular, one without any DIA source table."""
    res = night.resolve("20261113")
    keys = {(2026111300041, 164), (2026111300020, 165)}
    refs = [r for r in res["refs"] if (int(r.dataId["visit"]), int(r.dataId["detector"])) in keys]
    df = night.manifest_frame(res["butler"], res["repo"], refs, "20261113")
    path = tmp_path_factory.mktemp("night") / "work" / "manifest.csv"
    night.write_atomic(df, path)
    return path


@requires_butler
def test_stack_tables_real_panels(real_manifest):
    out = real_manifest.parent / "stack_sources.parquet"
    info = st.build("20261113", real_manifest, out, workers=2)
    assert info["panels_without_unfiltered"] == [[2026111300020, 165]] and info["panels_unreadable"] == []
    assert info["validation_failed_columns"] == [] and info["validation"]["trail_pa_deg"]["n"] > 0
    assert info["validation"]["mjd_tai"]["max_abs_diff_ms"] < 1.0
    df = pd.read_parquet(out)
    assert list(df.columns) == st.COLUMNS and len(df) == info["counts"]["rows"] > 0
    assert UNF["id"] in set(df.diaSourceId)
    assert json.loads((real_manifest.parent / "prep_info.json").read_text())["stack_tables"]["n_panels"] == 2


@requires_butler
def test_refcat_real_panels(real_manifest):
    out = real_manifest.parent / "refcat.parquet"
    info = refcat.build(real_manifest, out, workers=2)
    cat = pd.read_parquet(out)
    assert list(cat.columns) == ["ra", "dec", "gmag"] and len(cat) == info["n_sources"] > 0
    assert (cat.gmag < 21).all() and info["panels_without_region"] == [] and info["n_shards"] >= 1
    assert info["n_sources_shards"] >= info["n_sources"]
