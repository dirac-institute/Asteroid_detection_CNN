"""Export a trained diffim NN checkpoint to a portable TorchScript file.

Once exported, the .pt file can be loaded as:
    model = torch.jit.load("models/segmentation_model.pt")
    seg_logits, orient_sin, orient_cos, raw_seg, agg = model(x)
without needing the UNetResSEOrientHough class definition.

Usage:
    python -m ADCNN.inference.export \
        --ckpt <run_dir>/ckpts/best.pt \
        --out  models/segmentation_model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ADCNN.core.detector import UNetResSEOrientHough


def export_torchscript(ckpt_path: str, out_path: str, *,
                       in_ch: int = 3, kernel_lens=(11, 21, 41),
                       n_angles: int = 12, tile: int = 128,
                       widths=(48, 96, 192, 384, 768),
                       optimize_for_inference: bool = True) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    # Override from saved args if present.
    kernel_lens = tuple(args.get("kernel_lens", kernel_lens))
    n_angles = int(args.get("n_angles", n_angles))
    tile = int(args.get("tile", tile))
    widths = tuple(args.get("widths", widths))

    model = UNetResSEOrientHough(
        in_ch=in_ch, widths=widths,
        kernel_lens=kernel_lens, n_angles=n_angles,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    example = torch.zeros(1, in_ch, tile, tile, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example, check_trace=False)
    if optimize_for_inference:
        scripted = torch.jit.optimize_for_inference(scripted)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    # Sanity check: roundtrip and compare on a random batch.
    rt = torch.jit.load(str(out_path))
    x = torch.randn(2, in_ch, tile, tile)
    with torch.no_grad():
        out_orig = model(x)
        out_rt = rt(x)
    diffs = [float((a - b).abs().max()) for a, b in zip(out_orig, out_rt)]

    summary = {
        "ckpt": str(ckpt_path),
        "out": str(out_path),
        "in_ch": int(in_ch),
        "widths": list(widths),
        "kernel_lens": list(kernel_lens),
        "n_angles": int(n_angles),
        "tile": int(tile),
        "n_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "model_size_MB": Path(out_path).stat().st_size / 1e6,
        "agg_alpha": float(ckpt["model"]["agg_alpha"]),
        "roundtrip_max_abs_diff": diffs,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--no-optimize", action="store_true")
    args = ap.parse_args()

    s = export_torchscript(
        args.ckpt, args.out,
        in_ch=args.in_ch, tile=args.tile,
        optimize_for_inference=not args.no_optimize,
    )
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
