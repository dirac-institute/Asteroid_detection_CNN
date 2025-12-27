import math, numpy as np, torch, h5py
import torch.nn.functional as F
from .models.unet_res_se import UNetResSEASPP

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = UNetResSEASPP(in_ch=1, out_ch=1)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print("Loaded model from:", ckpt_path)
    return model


@torch.no_grad()
def predict_tiles_to_full(h5_path, loader, model, tile=128):
    """Assemble full-size per-panel predictions from tile predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    with h5py.File(h5_path, "r") as f:
        N, H, W = f["images"].shape
        Hb, Wb = math.ceil(H / tile), math.ceil(W / tile)
        tiles_per_panel = Hb * Wb
    xb0, _ = next(iter(loader))
    xb0 = xb0.to(next(model.parameters()).device)
    out0 = model(xb0[:1])
    oh, ow = out0.shape[-2], out0.shape[-1]
    full_preds = np.zeros((N, H, W), dtype=np.float32)
    tile_buf = []
    ptr = 0
    for xb, _ in loader:
        xb = xb.to(next(model.parameters()).device)
        out = model(xb)
        probs = out.detach()[:, 0]
        if (oh, ow) != (tile, tile):
            probs = F.interpolate(probs.unsqueeze(1), size=(tile, tile), mode='bilinear', align_corners=False).squeeze(1)
        probs = torch.sigmoid(probs)
        probs = probs.cpu().numpy()
        tile_buf.extend(list(probs))
        while len(tile_buf) >= tiles_per_panel:
            p = ptr // tiles_per_panel
            if p >= full_preds.shape[0]: break
            panel = np.zeros((Hb * tile, Wb * tile), dtype=np.float32)
            for r in range(Hb):
                for c in range(Wb):
                    t_idx = r * Wb + c
                    tile_img = tile_buf[t_idx]
                    r0, c0 = r * tile, c * tile
                    panel[r0:r0 + tile, c0:c0 + tile] = tile_img
            full_preds[p] = panel[:H, :W]
            tile_buf = tile_buf[tiles_per_panel:]
            ptr += tiles_per_panel
    return full_preds