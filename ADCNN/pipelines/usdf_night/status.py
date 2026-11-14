"""Bookkeeping of one usdf-night run: settings, night directory, lock, state.json, the Slurm chain
and the status report.

Stdlib only: run_night.sh and the slurm/*.sbatch stages run it by file path with the CNN python
(`python -I .../status.py ...`), so a status call never imports ADCNN (and torch).

  status.py prepare <day>                          day_obs + loadLSST.sh path; checks the overrides
  status.py submit <day> --check JSON [--force]    gates, night dir, lock, state.json, prep -> detect -> finish
  status.py show <day>                             night dir, state, jobs, panels, errors, lock
  status.py stage <N> <stage> start|end [RC] [--gpus NAMES]   job record N/logs/<stage>_<jobid>.json
  status.py finish <N>                             finish job: prints join, partial or retry; exit 1 = FAILED
  status.py close <N>                              after join: DONE (work/ removed) or PARTIAL; lock removed
  status.py fail <N> <message>                     state FAILED; lock removed

Settings come from the environment at submit (DEFAULTS) and are kept in state.json; the resubmits
from inside the finish job read them from there. Night dir N = $OUT_ROOT/<day>, <day>_floor<FLOOR>
when FLOOR != 0.43. N/.lock is stale when none of its job IDs is queued; a lock without job IDs (a
submission in progress) is stale after LOCK_GRACE seconds.
"""
from __future__ import annotations

import argparse
import collections
import csv
import grp
import importlib.util
import json
import os
import pwd
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True                 # the checkout is shared: no __pycache__ in it

WORKTREE = Path(__file__).resolve().parents[3]
DEFAULT_FLOOR = 0.43
DEFAULTS = {"ACCOUNT": "rubin:commissioning", "PARTITION": "ada", "GPUS": "4", "CPUS": "32",
            "MEM": "96G", "TIME": "8:00:00", "CPU_PARTITION": "milano", "FLOOR": "0.43",
            "OUT_ROOT": str(WORKTREE / "outputs" / "nights"), "MAX_ATTEMPTS": "3",
            "STACK": "/cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_46/loadLSST.sh"}
CPU_JOBS = {"prep": ("16", "64G", "6:00:00"), "finish": ("32", "64G", "2:00:00")}   # cpus, mem, time
STAGES = ("prep", "detect", "finish")
PREP_OUTPUTS = ("manifest.csv", "stack_sources.parquet", "refcat.parquet")
FINAL_OUTPUTS = ("detections.csv", "hl_dets.csv", "colformat.txt", "run_info.json")
DETS = "adcnn_dets.csv"
GROUP = "rubin_users"
ACL_USERS = ("mjuric", "chowian", "mrakovci")
LOCK_GRACE = 600
FAILED_JOB_STATES = ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
                     "BOOT_FAIL", "DEADLINE")


def _load_night():
    """night.py by file path (its top level is stdlib only), for its day_obs rule."""
    spec = importlib.util.spec_from_file_location("_usdf_night", Path(__file__).with_name("night.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


normalize_day = _load_night().normalize_day


class Refused(RuntimeError):
    """A gate refused the submission (complete night, held lock)."""


class Failed(RuntimeError):
    """The finish job cannot produce the night (recorded as FAILED)."""


class SlurmError(RuntimeError):
    """A Slurm command failed."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _epoch(iso) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return 0.0


def read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def write_state(n: Path, state: dict) -> None:
    state["updated"] = now_iso()
    write_json(n / "state.json", state)


# ------------------------------------------------------------------ settings and the night dir

def settings(env=None) -> dict:
    """DEFAULTS overridden by the (non-empty) environment variables, validated; keys lower case."""
    env = os.environ if env is None else env
    s = {k.lower(): env.get(k) or v for k, v in DEFAULTS.items()}
    for k in ("gpus", "cpus", "max_attempts"):
        if not s[k].isdigit() or int(s[k]) < 1:
            raise ValueError(f"{k.upper()} must be a positive integer, got {s[k]!r}")
        s[k] = int(s[k])
    try:
        s["floor"] = float(s["floor"])
    except ValueError:
        s["floor"] = -1.0
    if not 0 < s["floor"] < 1:
        raise ValueError(f"FLOOR must be a number in (0, 1), got {env.get('FLOOR')!r}")
    s["out_root"] = str(Path(s["out_root"]).expanduser().resolve())
    return s


def night_dir(out_root, day: str, floor: float) -> Path:
    """$OUT_ROOT/<day>, or <day>_floor<FLOOR> (two decimals unless more are needed) off the default."""
    if floor == DEFAULT_FLOOR:
        return Path(out_root) / day
    tag = f"{floor:.2f}"
    return Path(out_root) / f"{day}_floor{tag if float(tag) == floor else repr(floor)}"


def make_night_dir(n: Path) -> None:
    """Create N (mode 2770, group rubin_users, ACL + default ACL for ACL_USERS) unless it exists,
    and its logs/ and work/, which inherit N's default ACL and group."""
    n.parent.mkdir(parents=True, exist_ok=True)
    try:
        n.mkdir(mode=0o2770)
    except FileExistsError:
        pass
    else:
        os.chown(n, -1, grp.getgrnam(GROUP).gr_gid)
        os.chmod(n, 0o2770)
        acl = ",".join(f"u:{u}:rwx" for u in ACL_USERS)
        subprocess.run(["setfacl", "-m", f"{acl},{acl.replace('u:', 'd:u:')}", str(n)], check=True)
    for sub in ("logs", "work"):
        (n / sub).mkdir(mode=0o2770, exist_ok=True)


# ------------------------------------------------------------------ Slurm

def _run(argv, env=None) -> str:
    p = subprocess.run(argv, capture_output=True, text=True, env=env)
    if p.returncode:
        msg = (p.stderr.strip().splitlines() or [""])[0]
        raise SlurmError(f"{argv[0]} failed (exit {p.returncode}): {msg}")
    return p.stdout


def queued(user: str) -> dict:
    """{job id: (state, time used, nodes or pending reason)} of the user's jobs in squeue."""
    rows = (line.split("|", 3) for line in _run(["squeue", "-h", "-u", user, "-o", "%i|%T|%M|%R"]).splitlines())
    return {r[0].strip(): tuple(r[1:]) for r in rows if len(r) == 4}


def accounted(job_ids) -> dict:
    """{job id: (state, exit code, elapsed, nodes)} from sacct."""
    if not job_ids:
        return {}
    out = _run(["sacct", "-n", "-P", "-X", "-j", ",".join(job_ids), "-o", "JobID,State,ExitCode,Elapsed,NodeList"])
    rows = (line.split("|") for line in out.splitlines())
    return {r[0]: tuple(r[1:5]) for r in rows if len(r) >= 5}


def sbatch_env(env=None) -> dict:
    """The submitting environment minus the job's own SLURM_* variables (a nested sbatch from the
    finish job would inherit its allocation; SLURM_CONF is kept) and minus a loaded Python/stack."""
    env = os.environ if env is None else env
    drop = {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"}
    return {k: v for k, v in env.items()
            if k not in drop and (not k.startswith("SLURM_") or k == "SLURM_CONF")}


def sbatch_argv(stage: str, state: dict, dependency=None) -> list:
    """The full sbatch command of one stage, from state.json alone (every resource flag explicit)."""
    s, n = state["settings"], Path(state["night_dir"])
    if stage == "detect":
        res = [f"--partition={s['partition']}", f"--gres=gpu:{s['gpus']}", f"--cpus-per-task={s['cpus']}",
               f"--mem={s['mem']}", f"--time={s['time']}"]
    else:
        cpus, mem, time = CPU_JOBS[stage]
        res = [f"--partition={s['cpu_partition']}", f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={time}"]
    log = n / "logs" / f"{stage}_%j.log"
    argv = ["sbatch", "--parsable", f"--job-name=usdf_{stage}_{n.name}", f"--account={s['account']}",
            "--nodes=1", "--ntasks=1", *res, f"--chdir={n}", f"--output={log}", f"--error={log}",
            "--open-mode=append"]
    if dependency:
        argv += [f"--dependency={dependency}", "--kill-on-invalid-dep=yes"]
    args = {"prep": [state["night"], s["stack"]], "detect": [str(s["gpus"]), repr(s["floor"])], "finish": []}
    script = Path(state["worktree"]) / "ADCNN" / "pipelines" / "usdf_night" / "slurm" / f"{stage}.sbatch"
    return argv + [str(script), state["worktree"], str(n), *args[stage]]


def submit_chain(state: dict, stages) -> dict:
    """sbatch the stages in order, each after the previous one (detect afterok, finish afterany;
    --kill-on-invalid-dep so a failed prep cancels detect and finish still runs to record it).
    {stage: job id}. On an sbatch failure the jobs already submitted are cancelled."""
    env, ids = sbatch_env(), {}
    try:
        for stage in stages:
            dep = f"{'afterok' if stage == 'detect' else 'afterany'}:{list(ids.values())[-1]}" if ids else None
            job = _run(sbatch_argv(stage, state, dep), env).strip().split(";")[0]
            if not job.isdigit():
                raise SlurmError(f"sbatch printed no job id for {stage}: {job!r}")
            ids[stage] = job
    except (SlurmError, OSError):
        if ids:
            subprocess.run(["scancel", *ids.values()], env=env, capture_output=True)
        raise
    return ids


# ------------------------------------------------------------------ lock

def read_lock(n: Path) -> dict:
    """N/.lock, {} when absent; an unreadable lock counts as one without job IDs from its mtime."""
    path = n / ".lock"
    try:
        lock = json.loads(path.read_text())
        return lock if isinstance(lock, dict) else {}
    except FileNotFoundError:
        return {}
    except ValueError:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        except FileNotFoundError:
            return {}
        return {"user": "?", "host": "?", "job_ids": [], "time": mtime}


def lock_stale(lock: dict, queued_ids, now: float) -> bool:
    ids = {str(j) for j in lock.get("job_ids") or []}
    if ids:
        return not ids & {str(j) for j in queued_ids}
    return now - _epoch(lock.get("time")) > LOCK_GRACE


def lock_held(n: Path, now=None) -> dict:
    """The lock record when N/.lock is held, else {}. SlurmError when squeue cannot be asked."""
    lock = read_lock(n)
    if not lock:
        return {}
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    q = queued(lock["user"]) if lock.get("job_ids") else {}
    return {} if lock_stale(lock, q, now) else lock


def describe_lock(lock: dict) -> str:
    jobs = ",".join(lock.get("job_ids") or []) or "none yet"
    return f"{lock.get('user')}@{lock.get('host')} (jobs {jobs}) since {lock.get('time')}"


def acquire_lock(n: Path, user: str, host: str, now=None) -> dict:
    """Take N/.lock (exclusive create; a stale lock is replaced). Refused when it is held."""
    rec = {"user": user, "host": host, "job_ids": [], "time": now_iso()}
    path = n / ".lock"
    try:
        with open(path, "x") as f:
            json.dump(rec, f)
        return rec
    except FileExistsError:
        pass
    held = lock_held(n, now)
    if held:
        raise Refused(f"night {n.name} is locked by {describe_lock(held)}; see run_night.sh status")
    write_json(path, rec)
    return rec


def set_lock_jobs(n: Path, job_ids) -> None:
    lock = read_lock(n)
    lock.update(job_ids=list(job_ids), time=now_iso())
    write_json(n / ".lock", lock)


def release_lock(n: Path) -> None:
    (n / ".lock").unlink(missing_ok=True)


# ------------------------------------------------------------------ work/ contents

def manifest_panels(work: Path):
    """[(visit, detector)] of work/manifest.csv, None when it is not written yet."""
    try:
        with open(work / "manifest.csv", newline="") as f:
            return [(int(r["visit"]), int(r["detector"])) for r in csv.DictReader(f)]
    except FileNotFoundError:
        return None


def done_panels(work: Path) -> set:
    """(visit, detector) pairs in any detect shard .done log (same reading as detect_night)."""
    done = set()
    for f in work.glob(f"_shard_{Path(DETS).stem}_*.csv.done"):
        for line in open(f):
            p = line.strip().split(",")
            if len(p) == 2:
                done.add((int(p[0]), int(p[1])))
    return done


def detect_complete(work: Path) -> bool:
    """The detect merge ran at 100% coverage: adcnn_dets.csv written and its shards removed."""
    return (work / DETS).exists() and not any(work.glob(f"_shard_{Path(DETS).stem}_*"))


def coverage(work: Path):
    """{total, done, complete, missing [(visit, detector)]}, None before the manifest exists."""
    panels = manifest_panels(work)
    if panels is None:
        return None
    complete = detect_complete(work)
    done = set(panels) if complete else done_panels(work) & set(panels)
    return {"total": len(panels), "done": len(done), "complete": complete,
            "missing": [p for p in panels if p not in done]}


def unreadable(work: Path) -> dict:
    """{(visit, detector): last error} from detect's <out>.unreadable log (visit,detector,error)."""
    out = {}
    try:
        with open(work / f"{DETS}.unreadable", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split(",", 2)
                if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
                    out[(int(p[0]), int(p[1]))] = p[2]
    except FileNotFoundError:
        pass
    return out


def plan(work: Path) -> list:
    """Stages a (re)submission runs: prep unless all its outputs exist (they are written atomically),
    detect unless its merge is complete, finish always."""
    stages = [] if all((work / f).exists() for f in PREP_OUTPUTS) else ["prep"]
    if not detect_complete(work):
        stages.append("detect")
    return stages + ["finish"]


# ------------------------------------------------------------------ submit (run_night.sh)

def record_jobs(n: Path, state: dict, ids: dict) -> None:
    for stage, job in ids.items():
        state["jobs"].setdefault(stage, []).append(job)
    write_state(n, state)
    set_lock_jobs(n, ids.values())


def _resources(stage: str, s: dict) -> str:
    if stage == "detect":
        return f"{s['partition']}, gpu:{s['gpus']}, {s['cpus']} cpus, {s['mem']}, {s['time']}, floor {s['floor']!r}"
    cpus, mem, time = CPU_JOBS[stage]
    return f"{s['cpu_partition']}, {cpus} cpus, {mem}, {time}"


def _env_prefix(s: dict) -> str:
    """FLOOR=/OUT_ROOT= needed to find this night again with run_night.sh status."""
    out = [f"FLOOR={s['floor']!r} "] if s["floor"] != DEFAULT_FLOOR else []
    if s["out_root"] != str(Path(DEFAULTS["OUT_ROOT"]).resolve()):
        out.append(f"OUT_ROOT={s['out_root']} ")
    return "".join(out)


def submit(day: str, check: dict, force=False, env=None) -> int:
    """Gate, create N, lock, write state.json and submit the chain (plan()). Refused on a complete
    night (run_info.json present, unless force) or a held lock; nothing is created for 0 panels."""
    s = settings(env)
    if not check.get("n_panels"):
        print(f"no difference images for {day} in embargo or main")
        return 0
    n = night_dir(s["out_root"], day, s["floor"])
    old = read_json(n / "state.json")
    if (n / "run_info.json").exists() and not force:
        raise Refused(f"night {n.name} is already {old.get('status', 'complete')} ({n}); "
                      "--force runs it again (an incomplete work/ is resumed)")
    make_night_dir(n)
    user, host = pwd.getpwuid(os.getuid()).pw_name, socket.gethostname().split(".")[0]
    state = {"night": day, "floor": s["floor"], "night_dir": str(n), "worktree": str(WORKTREE),
             "user": user, "host": host, "settings": s, "check": check, "jobs": {}, "attempt": 1,
             "max_attempts": s["max_attempts"], "status": "SUBMITTED", "message": "", "forced": bool(force),
             "created": old.get("created") or now_iso(), "submitted": now_iso()}
    acquire_lock(n, user, host)
    try:
        if force:
            for f in FINAL_OUTPUTS:
                (n / f).unlink(missing_ok=True)
        write_state(n, state)
        ids = submit_chain(state, plan(n / "work"))
        record_jobs(n, state, ids)
    except BaseException as e:
        state.update(status="FAILED", message=f"submission failed: {e}")
        write_state(n, state)
        release_lock(n)
        raise
    print(f"night {n.name}: {check['n_panels']} panels, {check.get('n_visits')} visits "
          f"({check.get('repo')}, {check.get('dataset_type')}); attempt 1 of {s['max_attempts']}")
    for stage in STAGES:
        print(f"  {stage:<7}" + (f"job {ids[stage]}  {_resources(stage, s)}" if stage in ids
                                 else "skipped (its outputs are in work/)"))
    print(f"output    {n}")
    print(f"progress  {_env_prefix(s)}run_night.sh status {day}")
    return 0


# ------------------------------------------------------------------ inside the jobs

def record_stage(n: Path, stage: str, event: str, rc=None, gpus=None) -> None:
    """Start/end record of this Slurm job in N/logs/<stage>_<jobid>.json (one writer per file)."""
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = n / "logs" / f"{stage}_{job}.json"
    rec = read_json(path) or {"stage": stage, "job_id": job, "host": socket.gethostname().split(".")[0]}
    t = now_iso()
    if event == "start":
        rec.update(start=t, end=None, rc=None, seconds=None)
        if gpus is not None:
            rec["gpus"] = [g.strip() for g in gpus.split(",") if g.strip()]
    else:
        rec.update(end=t, rc=rc, seconds=round(_epoch(t) - _epoch(rec["start"])) if rec.get("start") else None)
    write_json(path, rec)


def collect_stages(n: Path) -> dict:
    """{stage: [job records by start time, then job id]} from N/logs/*.json (start has one-second
    resolution, so two jobs can share it)."""
    out = {s: [] for s in STAGES}
    for p in (n / "logs").glob("*_*.json"):
        rec = read_json(p)
        if rec.get("stage") in out:
            out[rec["stage"]].append(rec)
    for recs in out.values():
        recs.sort(key=lambda r: (r.get("start") or "",
                                 int(r["job_id"]) if str(r.get("job_id", "")).isdigit() else 0))
    return out


def finish_step(n: Path) -> str:
    """The finish job's decision, applied. 'retry': detection incomplete with attempts left, detect +
    finish resubmitted. 'join': detection complete. 'partial': attempts exhausted, the done panels
    are joined (panels_missing in state.json). Failed when prep left no outputs or nothing was
    detected. state.json gets stages, panels_total, panels_done, panels_missing before a join."""
    state = read_json(n / "state.json")
    work = n / "work"
    state["stages"] = collect_stages(n)
    cov = coverage(work)
    if cov is None or not all((work / f).exists() for f in PREP_OUTPUTS):
        prep = (state.get("jobs", {}).get("prep") or ["*"])[-1]
        raise Failed(f"prep did not write its outputs (see {n / 'logs' / f'prep_{prep}.log'})")
    if not cov["complete"] and state["attempt"] < state["max_attempts"]:
        state["attempt"] += 1
        state["status"] = "RESUBMITTED"
        state["message"] = (f"{len(cov['missing'])} of {cov['total']} panels missing after attempt "
                            f"{state['attempt'] - 1}")
        write_state(n, state)
        record_jobs(n, state, submit_chain(state, ["detect", "finish"]))
        return "retry"
    if cov["done"] == 0:
        raise Failed(f"detection covered 0 of {cov['total']} panels in {state['attempt']} attempts")
    bad = unreadable(work)
    state.update(panels_total=cov["total"], panels_done=cov["done"],
                 panels_missing=[[v, d, bad.get((v, d), "")] for v, d in cov["missing"]])
    write_state(n, state)
    return "join" if cov["complete"] else "partial"


def close(n: Path) -> None:
    """After a successful join: DONE (work/ removed) or PARTIAL (work/ kept for a --force resume)."""
    state = read_json(n / "state.json")
    miss = state.get("panels_missing") or []
    state.update(status="PARTIAL" if miss else "DONE", finished=now_iso(),
                 message=f"{len(miss)} of {state.get('panels_total')} panels missing" if miss else "")
    write_state(n, state)
    if not miss:
        shutil.rmtree(n / "work", ignore_errors=True)
    release_lock(n)


def fail(n: Path, message: str) -> None:
    state = read_json(n / "state.json")
    state.update(status="FAILED", message=message, finished=now_iso())
    write_state(n, state)
    release_lock(n)
    print(f"night {n.name}: FAILED: {message}", file=sys.stderr)


# ------------------------------------------------------------------ status report

def _tail(path: Path, k=5) -> list:
    try:
        with open(path, errors="replace") as f:
            return [line.rstrip("\n") for line in collections.deque(f, maxlen=k)]
    except FileNotFoundError:
        return ["(no log)"]


def show(day: str, env=None) -> int:
    """Print night dir, state, attempt, panels done/total, job states (squeue, else sacct), the
    tail of every failed job's log, and the lock (held or stale)."""
    s = settings(env)
    n = night_dir(s["out_root"], day, s["floor"])
    if not n.is_dir():
        print(f"night {day}: not submitted ({n} does not exist)")
        return 1
    state, work = read_json(n / "state.json"), n / "work"
    print(f"night     {n.name}  {n}")
    print(f"state     {state.get('status', '?')}  attempt {state.get('attempt', '?')}/"
          f"{state.get('max_attempts', '?')}  user {state.get('user', '?')}  updated {state.get('updated', '?')}")
    if state.get("message"):
        print(f"message   {state['message']}")
    chk = state.get("check") or {}
    if chk:
        print(f"source    {chk.get('repo')} {chk.get('collection')} {chk.get('dataset_type')}: "
              f"{chk.get('n_panels')} panels, {chk.get('n_visits')} visits")
    cov = coverage(work)
    if cov:
        bad = set(unreadable(work)) & set(cov["missing"])
        print(f"panels    {cov['done']}/{cov['total']} done" + (" (merged)" if cov["complete"] else "")
              + (f", {len(bad)} unreadable" if bad else ""))
    elif "panels_total" in state:
        print(f"panels    {state['panels_done']}/{state['panels_total']} done")
    else:
        print("panels    manifest not written yet")
    miss = state.get("panels_missing") or []
    if miss:
        print(f"missing   {len(miss)} panels: " + ", ".join(f"{v}/{d}" + (f" ({e})" if e else "")
                                                       for v, d, e in miss[:10]) + (" ..." if len(miss) > 10 else ""))
    done = [f for f in FINAL_OUTPUTS if (n / f).exists()]
    if done:
        print(f"outputs   {' '.join(done)}")

    jobs = sorted(((j, stage) for stage in STAGES for j in state.get("jobs", {}).get(stage, [])),
                  key=lambda x: int(x[0]))
    user = state.get("user", "")
    q, acc = {}, {}
    try:
        q = queued(user) if jobs else {}
    except SlurmError as e:
        print(f"squeue    {e}")
    try:
        acc = accounted([j for j, _ in jobs if j not in q])
    except SlurmError as e:
        print(f"sacct     {e}")
    recs = {(r["stage"], r["job_id"]): r for rs in collect_stages(n).values() for r in rs}
    failed = []
    if jobs:
        print("jobs")
    for j, stage in jobs:
        if j in q:
            st, ec, el, where = q[j][0], "", q[j][1], q[j][2]
        else:
            st, ec, el, where = acc.get(j, ("unknown", "", "", ""))
        print(f"  {stage:<7}{j:>10}  {st:<20}{ec:<6}{el:>11}  {where}")
        rc = recs.get((stage, j), {}).get("rc")
        if st.split(" ")[0] in FAILED_JOB_STATES or rc not in (None, 0):
            failed.append((stage, j, st))
    for stage, j, st in failed:
        log = n / "logs" / f"{stage}_{j}.log"
        print(f"error     {stage} {j} {st}: {log}")
        for line in _tail(log):
            print(f"            {line}")

    lock = read_lock(n)
    held = {}
    if lock:
        try:
            held = lock_held(n)
            print(f"lock      {describe_lock(lock)}: " + ("held" if held else "stale"))
        except SlurmError as e:
            print(f"lock      {describe_lock(lock)}: unknown ({e})")
            held = lock
    else:
        print("lock      none")
    if state.get("status") in ("SUBMITTED", "RESUBMITTED") and not held:
        print(f"note      no job of this night is queued: {_env_prefix(s)}run_night.sh {day} resumes it")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="status.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").add_argument("day")
    p = sub.add_parser("submit")
    p.add_argument("day")
    p.add_argument("--check", required=True, help="night.py check JSON")
    p.add_argument("--force", action="store_true")
    sub.add_parser("show").add_argument("day")
    p = sub.add_parser("stage")
    p.add_argument("night_dir", type=Path)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("event", choices=("start", "end"))
    p.add_argument("rc", type=int, nargs="?")
    p.add_argument("--gpus", help="comma-separated GPU names (nvidia-smi)")
    for cmd in ("finish", "close"):
        sub.add_parser(cmd).add_argument("night_dir", type=Path)
    p = sub.add_parser("fail")
    p.add_argument("night_dir", type=Path)
    p.add_argument("message")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            day, s = normalize_day(a.day), settings()
            print(f"{day}\n{s['stack']}")
        elif a.cmd == "submit":
            return submit(normalize_day(a.day), json.loads(a.check), a.force)
        elif a.cmd == "show":
            return show(normalize_day(a.day))
        elif a.cmd == "stage":
            record_stage(a.night_dir, a.stage, a.event, a.rc, a.gpus)
        elif a.cmd == "finish":
            try:
                print(finish_step(a.night_dir))
            except (Failed, SlurmError, OSError) as e:
                fail(a.night_dir, str(e))
                return 1
        elif a.cmd == "close":
            close(a.night_dir)
        else:
            fail(a.night_dir, a.message)
    except (Refused, SlurmError, ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f"run_night: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
