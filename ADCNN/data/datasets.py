import math, h5py, numpy as np, torch
from torch.utils.data import Dataset

def robust_stats_mad(arr: np.ndarray) -> tuple[np.float32, np.float32]:
    med = np.median(arr)
    # Protect against NaN median (e.g., all-NaN input)
    if not np.isfinite(med):
        med = 0.0
    mad = np.median(np.abs(arr - med))
    sigma = 1.4826 * (mad + 1e-12)
    # Protect against non-finite or zero sigma
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return np.float32(med), np.float32(sigma)

class H5TiledDataset(Dataset):
    """Stream tiles from (N,H,W) images; per-image robust norm, k-sigma clip, pad edges."""
    def __init__(self, h5_path, tile=128, k_sigma=5.0, crop_for_stats=512, precompute_stats=True):
        self.h5_path, self.tile, self.k_sigma, self.crop_for_stats = h5_path, int(tile), float(k_sigma), int(crop_for_stats)
        self._h5 = self._x = self._y = None
        self._stats_cache = {}
        self.precompute_stats = bool(precompute_stats)

        with h5py.File(self.h5_path, "r") as f:
            self.N, self.H, self.W = f["images"].shape
            assert f["masks"].shape == (self.N, self.H, self.W)

        Hb = math.ceil(self.H/self.tile); Wb = math.ceil(self.W/self.tile)
        self.indices = [(i, r, c) for i in range(self.N) for r in range(Hb) for c in range(Wb)]

        # Optionally precompute stats at init time for all panels
        if self.precompute_stats and self.N <= 2000:  # Only if reasonable number of panels
            self._precompute_all_stats()

    def _precompute_all_stats(self):
        """Precompute statistics for all panels to avoid per-tile branching."""
        self._ensure()
        for i in range(self.N):
            _ = self._image_stats(i)  # Populate cache

    def _ensure(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._x, self._y = self._h5["images"], self._h5["masks"]

    def _image_stats(self, i):
        if i in self._stats_cache: return self._stats_cache[i]
        s = min(self.crop_for_stats, self.H, self.W)
        h0, w0 = (self.H-s)//2, (self.W-s)//2
        crop = self._x[i, h0:h0+s, w0:w0+s].astype("float32")
        med, sig = robust_stats_mad(crop); self._stats_cache[i] = (med, sig); return med, sig

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        self._ensure()
        i, r, c = self.indices[idx]; t = self.tile
        r0, c0 = r*t, c*t; r1, c1 = min(r0+t, self.H), min(c0+t, self.W)
        x = self._x[i, r0:r1, c0:c1].astype("float32"); y = self._y[i, r0:r1, c0:c1].astype("float32")
        if x.shape != (t, t):
            xp = np.zeros((t,t), np.float32); yp = np.zeros((t,t), np.float32)
            xp[:x.shape[0], :x.shape[1]] = x; yp[:y.shape[0], :y.shape[1]] = y; x, y = xp, yp
        med, sig = self._image_stats(i); x = np.clip((x-med)/sig, -5, 5)
        return torch.from_numpy(x[None,...]), torch.from_numpy(y[None,...])

class SubsetDS(Dataset):
    """Select full panels by id while reusing tiling of base dataset."""
    def __init__(self, base: H5TiledDataset, panel_ids: np.ndarray):
        self.base, self.panel_ids = base, np.asarray(panel_ids)
        t = base.tile; Hb, Wb = math.ceil(base.H/t), math.ceil(base.W/t)
        base_map = {(i,r,c):k for k,(i,r,c) in enumerate(base.indices)}
        self.map = [base_map[(i,r,c)] for i in self.panel_ids for r in range(Hb) for c in range(Wb)]
    def __len__(self): return len(self.map)
    def __getitem__(self, k): return self.base[self.map[k]]

def panels_with_positives(h5_path, max_panels=None):
    ids=[]
    with h5py.File(h5_path,'r') as f:
        Y=f['masks']; N,H,W=Y.shape
        rng = np.random.default_rng(0)
        order = rng.permutation(N) if max_panels else np.arange(N)
        for i in order:
            yi = Y[i]
            if yi.any(): ids.append(i)
            if max_panels and len(ids)>=max_panels: break
    return np.array(sorted(ids))

def norm_medmad_clip(x, clip=5.0, eps=1e-6):
    # x: torch.Tensor [B,1,H,W] or [1,H,W]
    if x.ndim == 4:
        med = x.median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
    else:
        med = x.median().view(1,1,1)
    mad = (x - med).abs().median()
    sigma = 1.4826 * mad + eps
    z = (x - med) / sigma
    return z.clamp_(-clip, clip)

class WithTransform(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        x = norm_medmad_clip(x)
        return x, y
