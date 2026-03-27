from __future__ import annotations

import argparse

import h5py

from ADCNN.data.soft_masks import SoftMaskGenerator


def cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Precompute CSV-derived soft masks to an on-disk cache.")
    ap.add_argument("--train-h5", type=str, required=True)
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--cache-dir", type=str, required=True)
    ap.add_argument("--sigma-pix", type=float, default=2.0)
    ap.add_argument("--line-width", type=int, default=1)
    ap.add_argument("--truncate", type=float, default=4.0)
    ap.add_argument("--dtype", type=str, default="float16")
    return ap.parse_args()


def main() -> None:
    args = cli()
    with h5py.File(args.train_h5, "r") as f:
        n, h, w = f["images"].shape

    gen = SoftMaskGenerator(
        csv_path=args.train_csv,
        image_shape=(h, w),
        sigma_pix=float(args.sigma_pix),
        line_width=int(args.line_width),
        truncate=float(args.truncate),
        cache_dir=str(args.cache_dir),
        cache_size=1,
        dtype=str(args.dtype),
    )

    for image_id in range(int(n)):
        _ = gen.panel_mask(int(image_id))

    print(f"[precompute-soft-masks] cached {n} panels to {args.cache_dir}")


if __name__ == "__main__":
    main()
