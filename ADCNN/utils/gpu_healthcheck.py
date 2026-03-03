#!/usr/bin/env python3
import os, sys, json, subprocess

def test_device(idx: int) -> tuple[bool, str]:
    # Probe with a tiny CUDA op; catches ECC failures reliably.
    code = f"""
import torch, os
torch.cuda.set_device({idx})
x = torch.randn(1, device='cuda')
torch.cuda.synchronize()
print('OK')
"""
    env = os.environ.copy()
    # keep full visibility; we intentionally address absolute index `idx`
    try:
        out = subprocess.run([sys.executable, "-c", code], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        ok = (out.returncode == 0) and ("OK" in out.stdout)
        msg = out.stderr.strip() or out.stdout.strip()
        return ok, msg
    except Exception as e:
        return False, str(e)

def main():
    # count GPUs via nvidia-smi; fallback to torch if desired
    try:
        q = subprocess.run(["nvidia-smi", "--list-gpus"], stdout=subprocess.PIPE, text=True, check=True)
        num = len([ln for ln in q.stdout.splitlines() if ln.strip()])
    except Exception:
        import torch
        num = torch.cuda.device_count()

    healthy = []
    bad = {}
    for i in range(num):
        ok, msg = test_device(i)
        if ok:
            healthy.append(i)
        else:
            bad[i] = msg

    print(json.dumps({"healthy": healthy, "bad": bad}))
    sys.exit(0 if healthy else 1)

if __name__ == "__main__":
    main()
