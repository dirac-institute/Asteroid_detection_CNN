"""run_night's entry guards, exercised through the real CLI (2026-08-16 audit F7/F10).

F7: --dry-run must not MATERIALISE anything. It used to mkdir the run dir (and stream/), write
runtime_report.json and a runtime plot -- so a dry-run against a fresh --out left behind an empty
run_night_<N>/ that night_status --all and the campaign driver then treated as a real, broken
night.

F10: --visits auto means "reuse the existing manifest" (what regen_campaign passes for nights
regenerated from kept detection artifacts). With the manifest MISSING, the literal string used to
be forwarded into build_manifest, dying deep in an int() parse; it must instead fail at the guard,
naming the manifest and the remedy, before any stage runs or any sbatch is submitted.
"""
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _run(args, timeout=300):
    return subprocess.run([sys.executable, "-m", "ADCNN.pipelines.run_night"] + args,
                          cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO),
                          capture_output=True, text=True, timeout=timeout)


def test_dry_run_materialises_nothing(tmp_path):
    out = tmp_path / "fresh" / "run"
    r = _run(["--night", "20260706", "--visits", "2026070600001-2026070600010",
              "--collection", "X/unused", "--dry-run", "--out", str(out)])
    assert r.returncode == 0, f"dry-run failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    assert not out.parent.exists(), (
        f"--dry-run created {out.parent}: " +
        "; ".join(str(p) for p in out.parent.rglob("*")))


def test_visits_auto_without_manifest_fails_at_the_guard(tmp_path):
    out = tmp_path / "no_manifest"
    r = _run(["--night", "20260706", "--visits", "auto", "--collection", "X/unused",
              "--out", str(out)])
    assert r.returncode != 0, "auto with no manifest must fail"
    assert "--visits auto requires an existing manifest" in (r.stderr + r.stdout), (
        f"expected the guard's message, got:\n{r.stderr[-1500:]}")
    # and it must have failed BEFORE any stage side effect: at most the empty run dir exists
    left = [p for p in out.rglob("*")] if out.exists() else []
    assert not left, f"the failed run left artifacts: {left}"
