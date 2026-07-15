from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from run_pipeline import HAIR_CLASS_ID, run_sapiens2, save_gray


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cached Sapiens2 and save probabilities, labels and coarse Hair mask.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(__file__).parent / "models")
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")
    probability, labels = run_sapiens2(image, device, args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    save_gray(args.output / "2p_bok_hair_probability_16bit.png", probability, bit16=True)
    save_gray(args.output / "2p_bok_hair.png", (probability >= args.threshold).astype(np.float32))
    cv2.imwrite(str(args.output / "2p_bok_sapiens2_labels.png"), labels)
    print(f"Hair pixels: {int((labels == HAIR_CLASS_ID).sum())}; saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
