import time
import torch
from typing import Optional, Tuple


class DataLoaderProfiler:
    """Profile DataLoader throughput without training."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timings = []
        self.batch_counts = []

    def profile_throughput(
        self,
        loader,
        max_batches: int = 100,
        warmup_batches: int = 10,
        verbose: bool = True
    ) -> Tuple[float, float]:
        """Profile DataLoader throughput.

        Args:
            loader: DataLoader to profile
            max_batches: Maximum batches to process
            warmup_batches: Number of warmup iterations before timing
            verbose: Print results

        Returns:
            (tiles_per_second, batches_per_second)
        """

        # Warmup
        for i, batch in enumerate(loader):
            if i >= warmup_batches:
                break
            # Move to device to measure full pipeline
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device, non_blocking=True) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(self.device, non_blocking=True)

        # Timed measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.time()

        total_tiles = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device, non_blocking=True) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(self.device, non_blocking=True)

            # Count tiles (first tensor is usually the images)
            if isinstance(batch, (list, tuple)):
                batch_size = batch[0].size(0) if torch.is_tensor(batch[0]) else 1
            else:
                batch_size = batch.size(0) if torch.is_tensor(batch) else 1

            total_tiles += batch_size

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_end = time.time()

        elapsed = t_end - t_start
        batches_processed = min(max_batches, i + 1)

        tiles_per_sec = total_tiles / elapsed if elapsed > 0 else 0
        batches_per_sec = batches_processed / elapsed if elapsed > 0 else 0

        if verbose:
            print(f"DataLoader Profiling Results:")
            print(f"  Batches processed: {batches_processed}")
            print(f"  Total tiles: {total_tiles}")
            print(f"  Time elapsed: {elapsed:.2f}s")
            print(f"  Throughput: {tiles_per_sec:.1f} tiles/sec ({batches_per_sec:.2f} batches/sec)")

        return tiles_per_sec, batches_per_sec


def profile_dataloaders(train_loader, val_loader, device=None):
    """Quick profile of both train and val loaders."""
    profiler = DataLoaderProfiler(device=device)

    print("\n=== Training DataLoader ===")
    train_tiles_sec, train_batches_sec = profiler.profile_throughput(
        train_loader, max_batches=50, warmup_batches=5
    )

    print("\n=== Validation DataLoader ===")
    val_tiles_sec, val_batches_sec = profiler.profile_throughput(
        val_loader, max_batches=50, warmup_batches=5
    )

    return {
        "train_tiles_per_sec": train_tiles_sec,
        "train_batches_per_sec": train_batches_sec,
        "val_tiles_per_sec": val_tiles_sec,
        "val_batches_per_sec": val_batches_sec,
    }

