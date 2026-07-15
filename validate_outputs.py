from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--fine", type=Path, required=True)
    ap.add_argument("--prefix", default="2p")
    args = ap.parse_args()
    required = {
        "bok": args.base / f"{args.prefix}_bok.png",
        "edof": args.base / f"{args.prefix}_edof.png",
        "hair": args.base / f"{args.prefix}_bok_hair.png",
        "matte": args.base / f"{args.prefix}_bok_mat4k.png",
        "fine": args.fine / "fine_hair_mask.png",
        "fine_alpha": args.fine / "fine_hair_alpha_16bit.png",
    }
    images = {name: cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for name, path in required.items()}
    missing = [name for name, image in images.items() if image is None]
    if missing:
        raise FileNotFoundError(f"Missing/unreadable: {missing}")
    sizes = {name: [image.shape[1], image.shape[0]] for name, image in images.items()}
    if len({tuple(size) for size in sizes.values()}) != 1:
        raise AssertionError(f"Resolution mismatch: {sizes}")
    binary_values = np.unique(images["fine"])
    if not set(binary_values.tolist()).issubset({0, 255}):
        raise AssertionError(f"Fine mask is not binary: {binary_values[:20]}")
    if images["matte"].dtype != np.uint16 or images["fine_alpha"].dtype != np.uint16:
        raise AssertionError("Matte and fine alpha must be 16-bit PNGs")
    report = {
        "ok": True,
        "sizes_wh": sizes,
        "fine_mask_values": binary_values.tolist(),
        "fine_pixels": int((images["fine"] > 0).sum()),
        "fine_fraction_percent": float((images["fine"] > 0).mean() * 100),
        "matte_dtype": str(images["matte"].dtype),
        "fine_alpha_dtype": str(images["fine_alpha"].dtype),
    }
    out = args.fine / "validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
