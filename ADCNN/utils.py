import random, numpy as np, torch, h5py

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def split_indices(h5_path: str, val_frac: float = 0.1, seed: int = 1337):
    with h5py.File(h5_path, "r") as f:
        N = f["images"].shape[0]
    idx = np.arange(N); rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int((1.0 - val_frac) * N)
    return np.sort(idx[:split]), np.sort(idx[split:])
