#!/bin/bash
#SBATCH --job-name=GPU_TEST
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x_%j.out

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs

echo "=== SLURM / HOST ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo

echo "=== MODULES / ENV (trim) ==="
env | egrep -i '^(CONDA|PATH=|LD_LIBRARY_PATH=|CUDA|NVIDIA|NCCL|SLURM|PYTHONPATH)=' | sort || true
echo

echo "=== NVIDIA-SMI (host) ==="
command -v nvidia-smi || true
nvidia-smi -L || true
nvidia-smi || true
echo

echo "=== DRIVER / DEVICE FILES ==="
ls -l /dev/nvidia* || true
ls -l /proc/driver/nvidia/version || true
cat /proc/driver/nvidia/version 2>/dev/null || true
echo

# === Conda ===
echo "=== CONDA ACTIVATE ==="
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
which python
python -V
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo

echo "=== PYTORCH CUDA QUICK INFO ==="
python - <<'PY'
import os, torch, time
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[{i}] name={p.name} cc={p.major}.{p.minor} mem={p.total_memory/1e9:.2f} GB")
PY
echo

echo "=== PER-GPU ALLOC TEST (with generous timeout) ==="
# Your healthcheck times out at 10s; make this test slower and more verbose.
python - <<'PY'
import os, time, torch, traceback

def test_gpu(i):
    t0=time.time()
    try:
        torch.cuda.set_device(i)
        # Force context init
        x = torch.randn((4096,4096), device="cuda")  # ~64 MB float32
        y = x @ x.t()
        torch.cuda.synchronize()
        dt=time.time()-t0
        print(f"GPU {i}: OK (dt={dt:.2f}s)  name={torch.cuda.get_device_name(i)}")
        return True
    except Exception as e:
        dt=time.time()-t0
        print(f"GPU {i}: FAIL (dt={dt:.2f}s)  {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return False

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

ok=[]
for i in range(torch.cuda.device_count()):
    if test_gpu(i):
        ok.append(i)
print("Healthy GPUs:", ok)
PY
echo

echo "=== NCCL SMOKE TEST (single-process) ==="
# This catches some hangs caused by NCCL init in weird driver states.
python - <<'PY'
import os, torch, time
print("torch.distributed available:", torch.distributed.is_available())
print("NCCL available:", torch.distributed.is_nccl_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available() and torch.distributed.is_nccl_available():
    # Not initializing a process group (single task), just touching NCCL symbols
    a = torch.randn(1024, device="cuda")
    torch.cuda.synchronize()
    print("CUDA sync OK")
PY
echo

echo "=== DONE ==="

