from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from extract_fine_hair import ROLE_SUFFIXES, build_image_index, find_by_suffix


def main() -> None:
    ap = argparse.ArgumentParser(description="验证细发丝输出的尺寸、值域和位深。")
    ap.add_argument("--base", type=Path, required=True, help="基础输入图目录")
    ap.add_argument("--fine", type=Path, required=True, help="单样本细发丝输出目录")
    ap.add_argument("--prefix", default="2p", help="输入文件公共前缀")
    args = ap.parse_args()

    index = build_image_index(args.base)
    base_paths = {
        role: find_by_suffix(index, args.prefix, ROLE_SUFFIXES[role])
        for role in ("bok", "edof", "hair", "matte")
    }
    missing_base = [role for role in ("bok", "edof", "matte") if base_paths[role] is None]
    if missing_base:
        raise FileNotFoundError(f"缺少基础输入：{missing_base}")

    required = {
        "bok": base_paths["bok"],
        "edof": base_paths["edof"],
        "matte": base_paths["matte"],
        "fine_01": args.fine / "fine_hair_mask_01.png",
        "fine_255": args.fine / "fine_hair_mask.png",
        "fine_alpha": args.fine / "fine_hair_alpha_16bit.png",
        "blend": args.fine / "fine_hair_bok_blend.png",
    }
    if base_paths["hair"] is not None:
        required["hair"] = base_paths["hair"]
    images = {name: cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for name, path in required.items()}
    missing = [name for name, image in images.items() if image is None]
    if missing:
        raise FileNotFoundError(f"文件缺失或无法读取：{missing}")
    sizes = {name: [image.shape[1], image.shape[0]] for name, image in images.items()}
    if len({tuple(size) for size in sizes.values()}) != 1:
        raise AssertionError(f"分辨率不一致：{sizes}")
    binary_01_values = np.unique(images["fine_01"])
    binary_255_values = np.unique(images["fine_255"])
    if not set(binary_01_values.tolist()).issubset({0, 1}):
        raise AssertionError(f"01 mask 不是严格二值：{binary_01_values[:20]}")
    if not set(binary_255_values.tolist()).issubset({0, 255}):
        raise AssertionError(f"兼容 mask 不是 0/255 二值：{binary_255_values[:20]}")
    if not np.array_equal(images["fine_01"] > 0, images["fine_255"] > 0):
        raise AssertionError("01 mask 与 0/255 兼容 mask 内容不一致")
    if images["matte"].dtype != np.uint16 or images["fine_alpha"].dtype != np.uint16:
        raise AssertionError("matte 和细发丝 alpha 必须是 16-bit PNG")
    if images["blend"].ndim != 3 or images["blend"].shape[2] != 3:
        raise AssertionError("BOK blending 必须是三通道图像")

    metadata_path = args.fine / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"缺少运行元数据：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if base_paths["hair"] is None:
        source = metadata.get("fallback", {}).get("hair_mask_source")
        if source != "matte_fallback":
            raise AssertionError("Hair mask 缺失，但 metadata 未记录 matte fallback")
    report = {
        "ok": True,
        "sizes_wh": sizes,
        "fine_mask_01_values": binary_01_values.tolist(),
        "fine_mask_255_values": binary_255_values.tolist(),
        "fine_pixels": int((images["fine_01"] > 0).sum()),
        "fine_fraction_percent": float((images["fine_01"] > 0).mean() * 100),
        "matte_dtype": str(images["matte"].dtype),
        "fine_alpha_dtype": str(images["fine_alpha"].dtype),
        "hair_input_present": base_paths["hair"] is not None,
        "hair_mask_source": metadata.get("fallback", {}).get("hair_mask_source"),
        "base_inputs": {name: str(path) if path else None for name, path in base_paths.items()},
    }
    out = args.fine / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
