"""Driver: status.py (settings, night dir, lock, state, chain, report) and run_night.sh / the sbatch
templates up to sbatch.

Slurm commands are stubs first on PATH (sbatch records its argv and prints a job id; squeue, sacct
and scancel read or write files in the stub dir); every night dir is under tmp_path. status.py is
loaded by file path, as run_night.sh runs it (no ADCNN import).
"""
import grp
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
W = PKG.parents[2]
_spec = importlib.util.spec_from_file_location("usdf_status", PKG / "status.py")
status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(status)
SLURM = PKG / "slurm"

CHECK = {"repo": "embargo", "collection": "LSSTCam/runs/prompt-20261113",
         "dataset_type": "future_difference_image",
         "runs": ["LSSTCam/runs/prompt/20261113/ApPipe/pipelines-0d22056-config-8f017ea"],
         "n_panels": 4919, "n_visits": 46}
OVERRIDES = tuple(status.DEFAULTS)
P1, P2, P3 = (2026111300017, 42), (2026111300486, 129), (2026111300486, 130)

SBATCH = f"""#!{sys.executable} -I
import json, os, sys
from pathlib import Path
d = Path(os.environ["STUB_DIR"])
fail = os.environ.get("STUB_SBATCH_FAIL")
if fail and any(fail in a for a in sys.argv):
    print("sbatch: error: Batch job submission failed: Invalid partition name specified", file=sys.stderr)
    sys.exit(1)
log = d / "sbatch.jsonl"
n = len(log.read_text().splitlines()) if log.exists() else 0
env = {{k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in ("PYTHONPATH", "LD_LIBRARY_PATH")}}
with open(log, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "env": env}}) + "\\n")
print(f"{{900001 + n}};sdf")
"""
SQUEUE = """#!/bin/bash
[[ -e $STUB_DIR/squeue.fail ]] && { echo "squeue: error: Unable to contact slurm controller" >&2; exit 1; }
[[ -e $STUB_DIR/squeue.txt ]] && cat "$STUB_DIR/squeue.txt"
exit 0
"""
SACCT = """#!/bin/bash
[[ -e $STUB_DIR/sacct.txt ]] && cat "$STUB_DIR/sacct.txt"
exit 0
"""
SCANCEL = """#!/bin/bash
echo "$@" >> "$STUB_DIR/scancel.txt"
"""
# fake LSST stack: loadLSST.sh defines `setup` and puts a python on PATH that answers night check
FAKE_PYTHON = """#!/bin/bash
echo "$PYTHONPATH" > "$STUB_DIR/stack_pythonpath.txt"
[[ $1 == -m && $2 == ADCNN.pipelines.usdf_night.night && $3 == check ]] || exit 9
echo "a stray stdout line"
case $4 in
  20261113) echo '%s' ;;
  20261102|20261201) echo '{"repo": null, "collection": "LSSTCam/runs/prompt-'$4'", "dataset_type": null, "runs": [], "n_panels": 0, "n_visits": 0}' ;;
  *) echo "night $4: S3 credentials file /nope not found: embargo access needs your own USDF embargo credentials" >&2; exit 2 ;;
esac
""" % json.dumps(CHECK)
# fake LSST python for prep.sbatch: records each module call and writes its --out file
FAKE_PREP_PYTHON = """#!/bin/bash
echo "$*" >> "$STUB_DIR/prep_calls.txt"
while [[ $# -gt 0 ]]; do [[ $1 == --out ]] && { echo x > "$2"; }; shift; done
"""


def _write_exec(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o750)


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """Stub Slurm commands first on PATH, no override variables, OUT_ROOT under tmp_path."""
    d = tmp_path / "stub"
    (d / "bin").mkdir(parents=True)
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE), ("sacct", SACCT), ("scancel", SCANCEL)):
        _write_exec(d / "bin" / name, text)
    monkeypatch.setenv("PATH", f"{d / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setenv("STUB_DIR", str(d))
    for k in OVERRIDES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "nights"))
    assert shutil.which("sbatch") == str(d / "bin" / "sbatch")      # never the real sbatch
    return d


@pytest.fixture
def fake_stack(tmp_path, stubs):
    d = tmp_path / "lsst"
    (d / "bin").mkdir(parents=True)
    _write_exec(d / "bin" / "python", FAKE_PYTHON)
    (d / "loadLSST.sh").write_text(f'setup() {{ [[ $1 == lsst_distrib ]]; }}\nexport PATH={d / "bin"}:$PATH\n')
    return d / "loadLSST.sh"


def sbatch_calls(stubs):
    log = stubs / "sbatch.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def script_of(argv) -> str:
    return next(Path(a).stem for a in argv if a.endswith(".sbatch"))


def make_work(n: Path, panels=(P1, P2, P3), prep=True, done=(), dets=False, unread=None):
    """A night's work/ in a given state: manifest + prep outputs, shard .done lines, merged CSV."""
    work = n / "work"
    work.mkdir(parents=True, exist_ok=True)
    if prep:
        rows = "".join(f"{v},{d},r,20261113,61358.0,s3://x/{v}_{d}.fits,embargo,future_difference_image,run\n"
                       for v, d in panels)
        (work / "manifest.csv").write_text(
            "visit,detector,band,day_obs,mjd_tai,fits_path,repo,dataset_type,run\n" + rows)
        (work / "stack_sources.parquet").write_text("x")
        (work / "refcat.parquet").write_text("x")
    if done:
        (work / "_shard_adcnn_dets_0.csv").write_text("detid\n")
        (work / "_shard_adcnn_dets_0.csv.done").write_text("".join(f"{v},{d}\n" for v, d in done))
    if dets:
        (work / "adcnn_dets.csv").write_text("detid\n")
    if unread:
        (work / "adcnn_dets.csv.unreadable").write_text(unread)
    return work


def submitted_night(stubs, day="20261113"):
    assert status.submit(day, CHECK) == 0
    return status.night_dir(os.environ["OUT_ROOT"], day, status.DEFAULT_FLOOR)


# ------------------------------------------------------------------ day, settings, night dir

def test_normalize_day_is_nights_rule():
    for s in ("20261113", "2026-11-13", " 2026-11-13 "):
        assert status.normalize_day(s) == "20261113"
    for bad in ("2026-1113", "202611-13", "20261332", "2026111", "abc", "", "2026/11/13", "2026-11-13x"):
        with pytest.raises(ValueError):
            status.normalize_day(bad)
    assert status.normalize_day.__module__ == "_usdf_night"                # night.py's own rule


def test_settings_defaults():
    s = status.settings({})
    assert s == {"account": "rubin:commissioning", "partition": "ada", "gpus": 4, "cpus": 32, "mem": "96G",
                 "time": "8:00:00", "cpu_partition": "milano", "floor": 0.43,
                 "out_root": str((W / "outputs" / "nights").resolve()), "max_attempts": 3,
                 "stack": "/cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_46/loadLSST.sh"}


def test_settings_overrides_and_validation(tmp_path):
    s = status.settings({"PARTITION": "hopper", "GPUS": "2", "FLOOR": "0.30", "MEM": "", "OUT_ROOT": str(tmp_path),
                         "MAX_ATTEMPTS": "5", "STACK": "/x/loadLSST.sh"})
    assert (s["partition"], s["gpus"], s["floor"], s["mem"], s["out_root"], s["max_attempts"], s["stack"]) == \
        ("hopper", 2, 0.3, "96G", str(tmp_path.resolve()), 5, "/x/loadLSST.sh")
    for bad in ({"GPUS": "0"}, {"GPUS": "x"}, {"GPUS": "-1"}, {"CPUS": "2.5"}, {"MAX_ATTEMPTS": "0"},
                {"FLOOR": "abc"}, {"FLOOR": "1.5"}, {"FLOOR": "0"}):
        with pytest.raises(ValueError):
            status.settings(bad)


def test_night_dir_floor_tag(tmp_path):
    for f in ("0.43", "0.430"):
        assert status.night_dir(tmp_path, "20261113", float(f)) == tmp_path / "20261113"
    for f in ("0.3", "0.30", "0.300"):
        assert status.night_dir(tmp_path, "20261113", float(f)).name == "20261113_floor0.30"
    assert status.night_dir(tmp_path, "20261113", 0.5).name == "20261113_floor0.50"
    assert status.night_dir(tmp_path, "20261113", 0.425).name == "20261113_floor0.425"


def test_make_night_dir_mode_group_acl(tmp_path):
    n = tmp_path / "root" / "20261113"
    status.make_night_dir(n)
    status.make_night_dir(n)                                  # existing: no error, untouched
    gid = grp.getgrnam("rubin_users").gr_gid
    for d in (n, n / "logs", n / "work"):
        st = d.stat()
        assert st.st_gid == gid and st.st_mode & 0o2000, d
        acl = subprocess.run(["getfacl", "-p", str(d)], capture_output=True, text=True, check=True).stdout
        for u in ("mjuric", "chowian", "mrakovci"):
            assert f"user:{u}:rwx" in acl and f"default:user:{u}:rwx" in acl, (d, u)
        assert "other::---" in acl
    assert n.stat().st_mode & 0o7777 == 0o2770


# ------------------------------------------------------------------ lock

def test_lock_stale_rules():
    t = time.time()
    lock = {"user": "u", "job_ids": ["11", "12"], "time": status.now_iso()}
    assert not status.lock_stale(lock, {"12": ()}, t)
    assert status.lock_stale(lock, {}, t)
    assert status.lock_stale(lock, {"13": ()}, t)
    young = {"user": "u", "job_ids": [], "time": status.now_iso()}
    assert not status.lock_stale(young, {}, t + 10)
    assert status.lock_stale(young, {}, t + status.LOCK_GRACE + 1)
    assert status.lock_stale({"user": "u", "job_ids": []}, {}, t)            # no time at all


def test_acquire_lock_exclusive_stale_and_squeue_failure(stubs, tmp_path):
    n = tmp_path / "n"
    n.mkdir()
    rec = status.acquire_lock(n, "alice", "sdfiana001")
    assert status.read_lock(n) == rec and rec["job_ids"] == []
    with pytest.raises(status.Refused, match="locked by alice@sdfiana001"):   # submission in progress
        status.acquire_lock(n, "bob", "h")
    status.set_lock_jobs(n, ["5", "6"])
    (stubs / "squeue.txt").write_text("4|RUNNING|1:00|sdfada001\n6|PENDING|0:00|(Dependency)\n")
    with pytest.raises(status.Refused, match=r"jobs 5,6"):
        status.acquire_lock(n, "bob", "h")
    (stubs / "squeue.fail").write_text("")
    with pytest.raises(status.SlurmError, match="squeue failed"):
        status.acquire_lock(n, "bob", "h")
    (stubs / "squeue.fail").unlink()
    (stubs / "squeue.txt").write_text("4|RUNNING|1:00|sdfada001\n")       # 5, 6 gone: stale
    assert status.acquire_lock(n, "bob", "h")["user"] == "bob"
    assert status.read_lock(n)["job_ids"] == []
    (n / ".lock").write_text("{torn")                                     # unreadable, old -> stale
    old = time.time() - status.LOCK_GRACE - 5
    os.utime(n / ".lock", (old, old))
    assert status.acquire_lock(n, "carol", "h")["user"] == "carol"
    status.release_lock(n)
    status.release_lock(n)
    assert status.read_lock(n) == {}


def test_queued_and_accounted_parse(stubs):
    (stubs / "squeue.txt").write_text("900002|RUNNING|1:02:03|sdfada011\n900003|PENDING|0:00|(Dependency)\n")
    assert status.queued("mrakovci") == {"900002": ("RUNNING", "1:02:03", "sdfada011"),
                                         "900003": ("PENDING", "0:00", "(Dependency)")}
    (stubs / "sacct.txt").write_text("900001|COMPLETED|0:0|00:21:03|sdfmilan001\n"
                                     "900004|CANCELLED by 18743|0:15|00:00:00|None assigned\n")
    assert status.accounted(["900001", "900004"]) == {
        "900001": ("COMPLETED", "0:0", "00:21:03", "sdfmilan001"),
        "900004": ("CANCELLED by 18743", "0:15", "00:00:00", "None assigned")}
    assert status.accounted([]) == {}


# ------------------------------------------------------------------ sbatch lines

def _state(tmp_path, **env):
    s = status.settings({"OUT_ROOT": str(tmp_path), **env})
    n = status.night_dir(s["out_root"], "20261113", s["floor"])
    return {"night": "20261113", "night_dir": str(n), "worktree": str(W), "settings": s}


def test_sbatch_argv_exact(tmp_path):
    state = _state(tmp_path)
    n, sl = Path(state["night_dir"]), str(SLURM)
    common = ["sbatch", "--parsable"]
    assert status.sbatch_argv("prep", state) == common + [
        "--job-name=usdf_prep_20261113", "--account=rubin:commissioning", "--nodes=1", "--ntasks=1",
        "--partition=milano", "--cpus-per-task=16", "--mem=64G", "--time=6:00:00", f"--chdir={n}",
        f"--output={n}/logs/prep_%j.log", f"--error={n}/logs/prep_%j.log", "--open-mode=append",
        f"{sl}/prep.sbatch", str(W), str(n), "20261113",
        "/cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_46/loadLSST.sh"]
    assert status.sbatch_argv("detect", state, "afterok:1") == common + [
        "--job-name=usdf_detect_20261113", "--account=rubin:commissioning", "--nodes=1", "--ntasks=1",
        "--partition=ada", "--gres=gpu:4", "--cpus-per-task=32", "--mem=96G", "--time=8:00:00",
        f"--chdir={n}", f"--output={n}/logs/detect_%j.log", f"--error={n}/logs/detect_%j.log",
        "--open-mode=append", "--dependency=afterok:1", "--kill-on-invalid-dep=yes",
        f"{sl}/detect.sbatch", str(W), str(n), "4", "0.43"]
    assert status.sbatch_argv("finish", state, "afterany:2") == common + [
        "--job-name=usdf_finish_20261113", "--account=rubin:commissioning", "--nodes=1", "--ntasks=1",
        "--partition=milano", "--cpus-per-task=32", "--mem=64G", "--time=2:00:00", f"--chdir={n}",
        f"--output={n}/logs/finish_%j.log", f"--error={n}/logs/finish_%j.log", "--open-mode=append",
        "--dependency=afterany:2", "--kill-on-invalid-dep=yes", f"{sl}/finish.sbatch", str(W), str(n)]
    hop = status.sbatch_argv("detect", _state(tmp_path, PARTITION="hopper", GPUS="2", FLOOR="0.3"))
    assert "--partition=hopper" in hop and "--gres=gpu:2" in hop and hop[-1] == "0.3"
    assert hop[-3].endswith("20261113_floor0.30")


def test_sbatch_env_drops_job_and_python_variables():
    env = {"PATH": "/usr/bin", "HOME": "/h", "SLURM_JOB_ID": "1", "SLURM_MEM_PER_NODE": "64000",
           "SLURM_CPUS_PER_TASK": "32", "SLURM_CONF": "/etc/slurm/slurm.conf", "PYTHONPATH": "/stack",
           "PYTHONHOME": "/x", "LD_LIBRARY_PATH": "/lib", "AWS_SHARED_CREDENTIALS_FILE": "/c",
           "SLURMPATH": "/opt/slurm"}
    assert status.sbatch_env(env) == {"PATH": "/usr/bin", "HOME": "/h", "SLURM_CONF": "/etc/slurm/slurm.conf",
                                      "AWS_SHARED_CREDENTIALS_FILE": "/c", "SLURMPATH": "/opt/slurm"}


# ------------------------------------------------------------------ work/ state

def test_plan_and_coverage(tmp_path):
    n = tmp_path / "n"
    (n / "work").mkdir(parents=True)
    work = n / "work"
    assert status.plan(work) == ["prep", "detect", "finish"] and status.coverage(work) is None
    make_work(n)
    assert status.plan(work) == ["detect", "finish"]
    assert status.coverage(work) == {"total": 3, "done": 0, "complete": False, "missing": [P1, P2, P3]}
    make_work(n, done=(P1, P3, (2026111300999, 1)))                     # a panel not in the manifest
    assert status.coverage(work) == {"total": 3, "done": 2, "complete": False, "missing": [P2]}
    make_work(n, done=(P1, P3), dets=True)                               # partial merge: shards kept
    assert status.plan(work) == ["detect", "finish"] and not status.coverage(work)["complete"]
    for f in work.glob("_shard_*"):
        f.unlink()
    assert status.plan(work) == ["finish"]
    assert status.coverage(work) == {"total": 3, "done": 3, "complete": True, "missing": []}
    (work / "refcat.parquet").unlink()
    assert status.plan(work) == ["prep", "finish"]


def test_unreadable_parse(tmp_path):
    work = make_work(tmp_path / "n", unread=(
        f"{P2[0]},{P2[1]},OSError: read timed out, retrying\n"
        "garbage line\n"
        f"{P2[0]},{P2[1]},FileNotFoundError: [Errno 2] No such file: s3://x,y\n"
        f"{P3[0]},{P3[1]},KeyError: 'MASK'"))
    assert status.unreadable(work) == {P2: "FileNotFoundError: [Errno 2] No such file: s3://x,y",
                                       P3: "KeyError: 'MASK'"}
    assert status.unreadable(tmp_path) == {}


# ------------------------------------------------------------------ submit

def test_submit_fresh_night(stubs, capsys):
    n = submitted_night(stubs)
    out = capsys.readouterr().out
    assert "night 20261113: 4919 panels, 46 visits (embargo, future_difference_image); attempt 1 of 3" in out
    assert "detect job 900002  ada, gpu:4, 32 cpus, 96G, 8:00:00, floor 0.43" in out
    assert f"output    {n}" in out
    state = json.loads((n / "state.json").read_text())
    assert (state["status"], state["attempt"], state["max_attempts"], state["night"], state["floor"]) == \
        ("SUBMITTED", 1, 3, "20261113", 0.43)
    assert state["settings"] == status.settings() and state["check"] == CHECK
    assert state["jobs"] == {"prep": ["900001"], "detect": ["900002"], "finish": ["900003"]}
    assert state["worktree"] == str(W) and state["night_dir"] == str(n)
    lock = status.read_lock(n)
    assert lock["job_ids"] == ["900001", "900002", "900003"] and lock["user"] == state["user"]
    calls = [c["argv"] for c in sbatch_calls(stubs)]
    assert [script_of(c) for c in calls] == list(status.STAGES)
    assert not any(a.startswith("--dependency") for a in calls[0])
    assert "--dependency=afterok:900001" in calls[1] and "--dependency=afterany:900002" in calls[2]
    assert all("--kill-on-invalid-dep=yes" in c for c in calls[1:])


def test_submit_zero_panels_creates_nothing(stubs, tmp_path, capsys):
    empty = dict(CHECK, repo=None, dataset_type=None, runs=[], n_panels=0, n_visits=0)
    assert status.submit("20261201", empty) == 0
    assert capsys.readouterr().out == "no difference images for 20261201 in embargo or main\n"
    assert not (tmp_path / "nights").exists() and sbatch_calls(stubs) == []


def test_submit_refuses_complete_night_unless_force(stubs):
    n = submitted_night(stubs)
    status.release_lock(n)
    for f in status.FINAL_OUTPUTS:
        (n / f).write_text("x")
    with pytest.raises(status.Refused, match="already SUBMITTED"):
        status.submit("20261113", CHECK)
    assert len(sbatch_calls(stubs)) == 3
    assert status.submit("20261113", CHECK, force=True) == 0
    assert not any((n / f).exists() for f in status.FINAL_OUTPUTS)
    assert json.loads((n / "state.json").read_text())["forced"] is True
    assert len(sbatch_calls(stubs)) == 6


def test_submit_refuses_held_lock_and_resumes_after_stale(stubs):
    n = submitted_night(stubs)
    (stubs / "squeue.txt").write_text("900003|PENDING|0:00|(Dependency)\n")
    with pytest.raises(status.Refused, match="locked by"):
        status.submit("20261113", CHECK)
    (stubs / "squeue.txt").write_text("")
    make_work(n, done=(P1,))                                             # prep done, detect partial
    assert status.submit("20261113", CHECK) == 0
    calls = [c["argv"] for c in sbatch_calls(stubs)][3:]
    assert [script_of(c) for c in calls] == ["detect", "finish"]
    assert not any(a.startswith("--dependency") for a in calls[0]) and "--dependency=afterany:900004" in calls[1]
    make_work(n, done=(), dets=True)
    for f in (n / "work").glob("_shard_*"):
        f.unlink()
    status.release_lock(n)
    assert status.submit("20261113", CHECK) == 0                          # detection complete: finish only
    last = sbatch_calls(stubs)[-1]["argv"]
    assert len(sbatch_calls(stubs)) == 6 and script_of(last) == "finish"
    assert not any(a.startswith("--dependency") for a in last)


def test_submit_sbatch_failure_cancels_and_unlocks(stubs, monkeypatch):
    monkeypatch.setenv("STUB_SBATCH_FAIL", "detect.sbatch")
    with pytest.raises(status.SlurmError, match="Invalid partition"):
        status.submit("20261113", CHECK)
    n = status.night_dir(os.environ["OUT_ROOT"], "20261113", 0.43)
    assert (stubs / "scancel.txt").read_text() == "900001\n"
    assert not (n / ".lock").exists()
    state = json.loads((n / "state.json").read_text())
    assert state["status"] == "FAILED" and "Invalid partition" in state["message"]


def test_overrides_persist_into_finish_resubmit(stubs, monkeypatch):
    first = {"ACCOUNT": "rubin:developers", "PARTITION": "hopper", "GPUS": "2", "CPUS": "24", "MEM": "150G",
             "TIME": "12:00:00", "CPU_PARTITION": "roma", "FLOOR": "0.30", "MAX_ATTEMPTS": "2",
             "STACK": "/cvmfs/x/loadLSST.sh"}
    for k, v in first.items():
        monkeypatch.setenv(k, v)
    assert status.submit("20261113", CHECK) == 0
    n = Path(os.environ["OUT_ROOT"]) / "20261113_floor0.30"
    # the finish job runs later in another environment: different overrides, a job's SLURM_*, a stack
    for k, v in {"PARTITION": "ada", "GPUS": "7", "FLOOR": "0.43", "MAX_ATTEMPTS": "9", "ACCOUNT": "other",
                 "SLURM_JOB_ID": "900003", "SLURM_MEM_PER_NODE": "65536", "PYTHONPATH": "/stack/python"}.items():
        monkeypatch.setenv(k, v)
    make_work(n, done=(P1,))
    assert status.finish_step(n) == "retry"
    detect, finish = (c for c in sbatch_calls(stubs)[3:])
    for a in ("--account=rubin:developers", "--partition=hopper", "--gres=gpu:2", "--cpus-per-task=24",
              "--mem=150G", "--time=12:00:00"):
        assert a in detect["argv"], a
    assert detect["argv"][-2:] == ["2", "0.3"] and detect["argv"][-3] == str(n)
    assert "--partition=roma" in finish["argv"] and "--dependency=afterany:900004" in finish["argv"]
    assert not any(a.startswith("--dependency") for a in detect["argv"])
    assert detect["env"] == {} and finish["env"] == {}                    # no SLURM_*, no PYTHONPATH
    state = json.loads((n / "state.json").read_text())
    assert (state["attempt"], state["status"]) == (2, "RESUBMITTED")
    assert state["jobs"]["detect"] == ["900002", "900004"] and state["jobs"]["finish"] == ["900003", "900005"]
    assert "2 of 3 panels missing after attempt 1" in state["message"]
    assert status.read_lock(n)["job_ids"] == ["900004", "900005"]
    assert status.finish_step(n) == "partial"                            # attempt 2 of MAX_ATTEMPTS 2
    assert len(sbatch_calls(stubs)) == 5


# ------------------------------------------------------------------ finish job

def test_finish_step_branches(stubs):
    n = submitted_night(stubs)
    with pytest.raises(status.Failed, match="prep did not write its outputs.*prep_900001.log"):
        status.finish_step(n)
    make_work(n, done=(P1, P2), unread=f"{P3[0]},{P3[1]},OSError: timed out\n")
    state = json.loads((n / "state.json").read_text())
    state["attempt"] = 3
    status.write_state(n, state)
    assert status.finish_step(n) == "partial"
    state = json.loads((n / "state.json").read_text())
    assert (state["panels_total"], state["panels_done"]) == (3, 2)
    assert state["panels_missing"] == [[P3[0], P3[1], "OSError: timed out"]]
    assert set(state["stages"]) == {"prep", "detect", "finish"}
    (n / "work" / "_shard_adcnn_dets_0.csv.done").write_text("")
    with pytest.raises(status.Failed, match="covered 0 of 3 panels in 3 attempts"):
        status.finish_step(n)
    (n / "work" / "adcnn_dets.csv").write_text("detid\n")
    with pytest.raises(status.Failed, match="covered 0 of 3"):           # merged, but no panel done
        status.finish_step(n)
    for f in (n / "work").glob("_shard_*"):
        f.unlink()
    assert status.finish_step(n) == "join"
    state = json.loads((n / "state.json").read_text())
    assert (state["panels_total"], state["panels_done"], state["panels_missing"]) == (3, 3, [])


def test_close_done_partial_and_fail(stubs):
    n = submitted_night(stubs)
    make_work(n, dets=True)
    status.finish_step(n)
    status.close(n)
    state = json.loads((n / "state.json").read_text())
    assert state["status"] == "DONE" and not (n / "work").exists() and not (n / ".lock").exists()
    state.update(panels_missing=[[P3[0], P3[1], ""]], panels_total=3)
    status.write_state(n, state)
    make_work(n, done=(P1, P2))
    status.close(n)
    state = json.loads((n / "state.json").read_text())
    assert state["status"] == "PARTIAL" and state["message"] == "1 of 3 panels missing"
    assert (n / "work" / "manifest.csv").exists()
    status.acquire_lock(n, "u", "h")
    status.fail(n, "join exited 1")
    state = json.loads((n / "state.json").read_text())
    assert (state["status"], state["message"]) == ("FAILED", "join exited 1") and not (n / ".lock").exists()


def test_record_stage_and_collect(tmp_path, monkeypatch):
    n = tmp_path / "n"
    (n / "logs").mkdir(parents=True)
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    status.record_stage(n, "detect", "start", gpus="NVIDIA L40S,NVIDIA L40S")
    rec = json.loads((n / "logs" / "detect_777.json").read_text())
    assert rec["gpus"] == ["NVIDIA L40S", "NVIDIA L40S"] and rec["end"] is None and rec["job_id"] == "777"
    status.record_stage(n, "detect", "end", rc=3)
    rec = json.loads((n / "logs" / "detect_777.json").read_text())
    assert rec["rc"] == 3 and isinstance(rec["seconds"], int) and rec["end"] >= rec["start"]
    monkeypatch.setenv("SLURM_JOB_ID", "778")
    status.record_stage(n, "detect", "start", gpus="")
    assert json.loads((n / "logs" / "detect_778.json").read_text())["gpus"] == []
    got = status.collect_stages(n)
    assert [r["job_id"] for r in got["detect"]] == ["777", "778"] and got["prep"] == got["finish"] == []


# ------------------------------------------------------------------ status report

def test_show_report(stubs, capsys):
    assert status.show("20261113") == 1
    assert "not submitted" in capsys.readouterr().out
    n = submitted_night(stubs)
    make_work(n, done=(P1, P2), unread=f"{P3[0]},{P3[1]},OSError: timed out\n")
    (stubs / "sacct.txt").write_text("900001|COMPLETED|0:0|00:21:03|sdfmilan001\n"
                                     "900002|FAILED|3:0|01:10:00|sdfada011\n")
    (stubs / "squeue.txt").write_text("900003|RUNNING|0:01:00|sdfmilan002\n")
    (n / "logs" / "detect_900002.log").write_text("".join(f"line {i}\n" for i in range(9)) +
                                                  "[discover] 1 panels missing: shards kept\n")
    capsys.readouterr()
    assert status.show("20261113") == 0
    out = capsys.readouterr().out
    assert f"night     20261113  {n}" in out
    assert "state     SUBMITTED  attempt 1/3" in out
    assert "source    embargo LSSTCam/runs/prompt-20261113 future_difference_image: 4919 panels, 46 visits" in out
    assert "panels    2/3 done, 1 unreadable" in out
    assert "COMPLETED" in out and "RUNNING" in out
    assert f"error     detect 900002 FAILED: {n}/logs/detect_900002.log" in out
    assert "[discover] 1 panels missing: shards kept" in out and "line 4" not in out and "line 5" in out
    assert "900001,900002,900003" in out and out.rstrip().endswith("held")
    (stubs / "squeue.txt").write_text("")
    (n / "logs" / "finish_900003.json").write_text(json.dumps(
        {"stage": "finish", "job_id": "900003", "start": status.now_iso(), "rc": 1}))
    assert status.show("20261113") == 0
    out = capsys.readouterr().out
    assert f"error     finish 900003 unknown: {n}/logs/finish_900003.log" in out
    assert ": stale" in out and "note      no job of this night is queued: OUT_ROOT=" in out
    assert out.rstrip().endswith(" run_night.sh 20261113 resumes it")


# ------------------------------------------------------------------ run_night.sh and the templates

def run_sh(args, env=None):
    return subprocess.run(["bash", str(W / "run_night.sh"), *args], capture_output=True, text=True,
                          env=env or os.environ.copy())


def test_run_night_usage(stubs):
    for args in ([], ["--force"], ["status"], ["status", "20261113", "x"], ["20261113", "--bogus"],
                 ["20261113", "--force", "x"], ["-h"], ["--force", "20261113"]):
        r = run_sh(args)
        assert r.returncode == 1 and r.stdout == "" and sbatch_calls(stubs) == [], args
        assert r.stderr.splitlines() == ["usage: run_night.sh <YYYY-MM-DD|YYYYMMDD> [--force] | "
                                         "run_night.sh status <YYYY-MM-DD|YYYYMMDD>"], args


def test_run_night_bad_day_and_override(stubs):
    r = run_sh(["2026-1113"])
    assert r.returncode == 1 and len(r.stderr.splitlines()) == 1 and "YYYYMMDD" in r.stderr
    r = run_sh(["20261113"], dict(os.environ, GPUS="zero"))
    assert r.returncode == 1 and r.stderr.strip() == "run_night: GPUS must be a positive integer, got 'zero'"
    assert sbatch_calls(stubs) == []


def test_run_night_check_paths(stubs, fake_stack, tmp_path):
    env = dict(os.environ, STACK=str(fake_stack), PYTHONPATH="/user/python")
    r = run_sh(["2026-11-13"], env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("night 20261113: 4919 panels, 46 visits (embargo, future_difference_image)")
    assert (stubs / "stack_pythonpath.txt").read_text().strip() == f"{W}:/user/python"
    assert len(sbatch_calls(stubs)) == 3 and all(c["env"] == {} for c in sbatch_calls(stubs))
    assert (tmp_path / "nights" / "20261113" / "state.json").exists()
    for day in ("20261102", "2026-12-01"):
        r = run_sh([day], env)
        assert (r.returncode, r.stderr) == (0, "")
        assert r.stdout == f"no difference images for {day.replace('-', '')} in embargo or main\n"
    r = run_sh(["20990101"], env)
    assert r.returncode == 2 and "embargo credentials" in r.stderr and r.stdout == ""
    assert sorted(p.name for p in (tmp_path / "nights").iterdir()) == ["20261113"]
    assert len(sbatch_calls(stubs)) == 3
    (stubs / "squeue.txt").write_text("900002|RUNNING|0:10|sdfada011\n")
    r = run_sh(["20261113"], env)
    assert r.returncode == 1 and "locked by" in r.stderr
    r = run_sh(["status", "20261113"], env)
    assert r.returncode == 0 and "state     SUBMITTED" in r.stdout and "held" in r.stdout
    r = run_sh(["20261113"], dict(env, STACK=str(tmp_path / "missing.sh")))
    assert r.returncode == 1 and "cannot set up the LSST stack" in r.stderr
    (stubs / "squeue.txt").write_text("")
    (tmp_path / "nights" / "20261113" / "run_info.json").write_text("{}")
    r = run_sh(["20261113"], env)
    assert r.returncode == 1 and "already SUBMITTED" in r.stderr and len(sbatch_calls(stubs)) == 3
    r = run_sh(["20261113", "--force"], env)
    assert r.returncode == 0, r.stderr
    assert len(sbatch_calls(stubs)) == 6 and not (tmp_path / "nights" / "20261113" / "run_info.json").exists()


def run_template(name, args, env):
    return subprocess.run(["bash", str(SLURM / name), *args], capture_output=True, text=True, env=env)


def test_finish_template_retry_partial_fail(stubs):
    n = submitted_night(stubs)
    make_work(n, done=(P1,))
    env = dict(os.environ, SLURM_JOB_ID="900003")
    r = run_template("finish.sbatch", [str(W), str(n)], env)
    assert r.returncode == 0, r.stderr
    assert [script_of(c["argv"]) for c in sbatch_calls(stubs)[3:]] == ["detect", "finish"]
    state = json.loads((n / "state.json").read_text())
    assert (state["attempt"], state["status"]) == (2, "RESUBMITTED")
    assert json.loads((n / "logs" / "finish_900003.json").read_text())["rc"] == 0
    (n / "work" / "manifest.csv").unlink()
    r = run_template("finish.sbatch", [str(W), str(n)], dict(env, SLURM_JOB_ID="900005"))
    assert r.returncode == 1 and "prep did not write its outputs" in r.stderr
    state = json.loads((n / "state.json").read_text())
    assert state["status"] == "FAILED" and not (n / ".lock").exists()
    assert state["message"].startswith("prep did not write its outputs")
    assert json.loads((n / "logs" / "finish_900005.json").read_text())["rc"] == 1


def test_prep_template_runs_missing_steps(stubs, tmp_path):
    d = tmp_path / "lsst"
    (d / "bin").mkdir(parents=True)
    _write_exec(d / "bin" / "python", FAKE_PREP_PYTHON)
    stack = d / "loadLSST.sh"
    stack.write_text(f'setup() {{ [[ $1 == lsst_distrib ]]; }}\nexport PATH={d / "bin"}:$PATH\n')
    n = tmp_path / "n"
    (n / "work").mkdir(parents=True)
    (n / "logs").mkdir()
    env = dict(os.environ, SLURM_JOB_ID="41")
    r = run_template("prep.sbatch", [str(W), str(n), "20261113", str(stack)], env)
    assert r.returncode == 0, r.stderr
    w = n / "work"
    assert (stubs / "prep_calls.txt").read_text().splitlines() == [
        f"-m ADCNN.pipelines.usdf_night.night manifest 20261113 --out {w}/manifest.csv",
        f"-m ADCNN.pipelines.usdf_night.stack_tables 20261113 --manifest {w}/manifest.csv "
        f"--out {w}/stack_sources.parquet --workers 16",
        f"-m ADCNN.pipelines.usdf_night.refcat --manifest {w}/manifest.csv --out {w}/refcat.parquet --workers 16"]
    (w / "stack_sources.parquet").unlink()
    (stubs / "prep_calls.txt").unlink()
    r = run_template("prep.sbatch", [str(W), str(n), "20261113", str(stack)], dict(env, SLURM_JOB_ID="42"))
    assert r.returncode == 0
    assert (stubs / "prep_calls.txt").read_text().splitlines() == [
        f"-m ADCNN.pipelines.usdf_night.stack_tables 20261113 --manifest {w}/manifest.csv "
        f"--out {w}/stack_sources.parquet --workers 16"]
    assert json.loads((n / "logs" / "prep_42.json").read_text())["rc"] == 0
    (stubs / "prep_calls.txt").unlink()
    r = run_template("prep.sbatch", [str(W), str(n), "20261113", str(stack)], dict(env, SLURM_JOB_ID="44"))
    assert r.returncode == 0 and not (stubs / "prep_calls.txt").exists()
    _write_exec(d / "bin" / "python", "#!/bin/bash\nexit 4\n")
    (w / "refcat.parquet").unlink()
    r = run_template("prep.sbatch", [str(W), str(n), "20261113", str(stack)], dict(env, SLURM_JOB_ID="43"))
    assert r.returncode == 4 and json.loads((n / "logs" / "prep_43.json").read_text())["rc"] == 4


def test_detect_template_model_md5_and_exit_code(stubs, tmp_path):
    """Real models pass the md5 check; detect_night then refuses the CPU-only login node (exit 1)."""
    n = tmp_path / "n"
    make_work(n)
    (n / "logs").mkdir()
    env = {k: v for k, v in os.environ.items() if k != "ADCNN_ALLOW_CPU"}
    env.update(SLURM_JOB_ID="51", CUDA_VISIBLE_DEVICES="")              # same outcome on a GPU host
    r = run_template("detect.sbatch", [str(W), str(n), "1", "0.43"], env)
    assert r.returncode == 1 and "CUDA unavailable" in r.stderr and "FAILED" not in r.stdout
    rec = json.loads((n / "logs" / "detect_51.json").read_text())
    assert rec["rc"] == 1 and rec["gpus"] == []


def test_detect_template_refuses_wrong_models(stubs, tmp_path):
    """A worktree whose model files do not match the v2_D2 md5s never reaches detect_night."""
    fake = tmp_path / "W"
    (fake / "ADCNN" / "pipelines" / "usdf_night").mkdir(parents=True)
    for f in ("status.py", "night.py"):
        shutil.copy(PKG / f, fake / "ADCNN" / "pipelines" / "usdf_night" / f)
    (fake / "models" / "v2_D2").mkdir(parents=True)
    for f in ("segmentation_scripted.pt", "cnn_postproc.pt"):
        (fake / "models" / "v2_D2" / f).write_text("not the model")
    n = tmp_path / "n"
    make_work(n)
    (n / "logs").mkdir()
    r = run_template("detect.sbatch", [str(fake), str(n), "1", "0.43"], dict(os.environ, SLURM_JOB_ID="52"))
    assert r.returncode == 1 and "did NOT match" in r.stderr and "cnn_postproc.pt: FAILED" in r.stdout
    assert "discover" not in r.stdout + r.stderr
    assert json.loads((n / "logs" / "detect_52.json").read_text())["rc"] == 1
