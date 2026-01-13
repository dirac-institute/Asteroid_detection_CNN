import os, torch, torch.distributed as dist
from datetime import timedelta

"""
def init_distributed(backend: str = "nccl", timeout_sec: int = 1800):
    # If already initialized, just return current ranks
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size

    # Not initialized yet: check env from torchrun / SLURM
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend=backend,
            init_method=os.environ.get("INIT_METHOD", "env://"),
            timeout=timedelta(seconds=timeout_sec),
        )
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size

    # Single-process fallback
    return False, 0, 0, 1"""

"""def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # (optional but helpful for debugging)
        # print(f"[rank {rank}] cuda avail={torch.cuda.is_available()} count={torch.cuda.device_count()} local_rank={local_rank}", flush=True)

        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, local_rank, world_size

    return False, 0, 0, 1"""
def init_distributed(backend: str = "nccl", init_method: str = "env://"):
    # If already initialized, just report ranks (do NOT re-init)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size

    # Initialize only if launched with torchrun / SLURM env
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend=backend, init_method=init_method)
        return True, rank, local_rank, world_size

    # Single-process fallback
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