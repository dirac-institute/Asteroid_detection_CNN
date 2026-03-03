import random, numpy as np, torch, h5py, cv2

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def split_indices(h5_path: str, val_frac: float = 0.1, seed: int = 1337):
    with h5py.File(h5_path, "r") as f:
        N = f["images"].shape[0]
    idx = np.arange(N); rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int((1.0 - val_frac) * N)
    return np.sort(idx[:split]), np.sort(idx[split:])

def draw_one_line(mask, origin, angle, length, true_value=1, line_thickness=500):
    x0, y0 = origin
    x_size = length * np.cos((np.pi / 180) * angle)
    y_size = length * np.sin((np.pi / 180) * angle)
    x1 = x0 - x_size / 2
    y1 = y0 - y_size / 2
    x0 = x0 + x_size / 2
    y0 = y0 + y_size / 2
    one_line_mask = cv2.line(np.zeros(mask.shape), (int(x0), int(y0)), (int(x1), int(y1)), 1, thickness=line_thickness)
    mask[one_line_mask != 0] = true_value
    return mask