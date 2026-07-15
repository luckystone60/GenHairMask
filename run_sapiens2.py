from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from run_pipeline import HAIR_CLASS_ID, run_sapiens2, save_gray


def main() -> None:
    ap = argparse.ArgumentParser(description="运行已缓存的 Sapiens2，并保存 Hair 概率、标签和粗 mask。")
    ap.add_argument("--image", type=Path, required=True, help="输入 BOK 图像")
    ap.add_argument("--output", type=Path, required=True, help="输出目录")
    ap.add_argument("--prefix", help="输出前缀；默认由输入文件名去掉末尾 _bok 得到")
    ap.add_argument("--cache", type=Path, default=Path(__file__).parent / "models", help="模型缓存目录")
    ap.add_argument("--threshold", type=float, default=0.45, help="粗 Hair mask 阈值")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")
    probability, labels = run_sapiens2(image, device, args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    prefix = args.prefix or (stem[:-4] if stem.casefold().endswith("_bok") else stem)
    save_gray(args.output / f"{prefix}_bok_hair_probability_16bit.png", probability, bit16=True)
    save_gray(args.output / f"{prefix}_bok_hair.png", (probability >= args.threshold).astype(np.float32))
    cv2.imwrite(str(args.output / f"{prefix}_bok_sapiens2_labels.png"), labels)
    print(f"Hair 像素：{int((labels == HAIR_CLASS_ID).sum())}；输出：{args.output.resolve()}")


if __name__ == "__main__":
    main()
