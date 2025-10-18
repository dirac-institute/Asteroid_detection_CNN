import os, torch, torch.distributed as dist
from datetime import timedelta

def init_distributed(backend: str = "nccl", timeout_sec: int = 1800):
    """Initialize torch.distributed if RUNNING in multi-GPU."""
    if dist.is_available() and (int(os.environ.get("WORLD_SIZE", "1")) > 1):
        # torchrun sets these env vars; SLURM + srun can too (via --export)
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend=backend,
            init_method=os.environ.get("INIT_METHOD", "env://"),
            timeout=timedelta(seconds=timeout_sec),
        )
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1

def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean of scalar tensor across processes."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= dist.get_world_size()
    return y

def broadcast_scalar_float(value: float, src: int = 0, device: torch.device | None = None) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([value], device=dev, dtype=torch.float32)
    dist.broadcast(t, src)
    return float(t.item())