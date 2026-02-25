#!/usr/bin/env python3
import os, sys, json, subprocess, argparse

def test_device(idx: int, timeout_s: int) -> tuple[bool, str]:
    code = f"""
import torch
torch.cuda.set_device({idx})
# Force context creation
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print("OK")
"""
    env = os.environ.copy()
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        ok = (out.returncode == 0) and ("OK" in out.stdout)
        msg = (out.stderr.strip() or out.stdout.strip())
        return ok, msg
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_s} seconds"
    except Exception as e:
        return False, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("GPU_HEALTHCHECK_TIMEOUT", "90")))
    args = ap.parse_args()

    # count GPUs via nvidia-smi; fallback to torch
    try:
        q = subprocess.run(["nvidia-smi", "--list-gpus"], stdout=subprocess.PIPE, text=True, check=True)
        num = len([ln for ln in q.stdout.splitlines() if ln.strip()])
    except Exception:
        import torch
        num = torch.cuda.device_count()

    healthy = []
    bad = {}
    for i in range(num):
        ok, msg = test_device(i, args.timeout)
        if ok:
            healthy.append(i)
        else:
            bad[str(i)] = msg

    print(json.dumps({"healthy": healthy, "bad": bad}))
    sys.exit(0 if healthy else 1)

if __name__ == "__main__":
    main()