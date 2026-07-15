from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp")


def read_color(path: Path) -> np.ndarray:
    """读取 BGR 彩色图像。"""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取彩色图像：{path}")
    return image


def read_float_mask(path: Path) -> np.ndarray:
    """读取 uint8/uint16/float mask，并明确转换为 0～1。"""

    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise FileNotFoundError(f"无法读取 mask：{path}")
    if source.ndim == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    if source.dtype == np.uint8:
        scale = 1.0 if int(source.max(initial=0)) <= 1 else 255.0
    elif source.dtype == np.uint16:
        scale = 1.0 if int(source.max(initial=0)) <= 1 else 65535.0
    else:
        source = source.astype(np.float32)
        if not np.isfinite(source).all() or float(source.min(initial=0.0)) < 0.0:
            raise ValueError(f"mask 含有非法数值：{path}")
        maximum = float(source.max(initial=0.0))
        if maximum <= 1.0:
            scale = 1.0
        elif maximum <= 255.0:
            scale = 255.0
        elif maximum <= 65535.0:
            scale = 65535.0
        else:
            raise ValueError(f"无法判断 mask 数值范围：{path}，最大值={maximum}")
    return np.clip(source.astype(np.float32) / scale, 0.0, 1.0)


def write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    """写入图像，失败时立即报错。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, params or []):
        raise OSError(f"图像写入失败：{path}")


def ellipse(radius: int) -> np.ndarray:
    """生成指定半径的椭圆形形态学核。"""

    radius = max(int(radius), 0)
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )


def find_prefixed_image(prefix_path: Path, suffix: str) -> Path | None:
    """按公共前缀和后缀查找图像，优先使用无损格式。"""

    base = prefix_path.parent / f"{prefix_path.name}{suffix}"
    for extension in IMAGE_EXTENSIONS:
        # 不能使用 with_suffix，否则带点号的前缀（如 portrait.v2）会被误截断。
        candidate = Path(f"{base}{extension}")
        if candidate.is_file():
            return candidate
    return None


def first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    """返回目录中第一个存在的候选文件。"""

    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path | None, str]:
    """解析显式路径或 prefix/fine-dir 形式的输入。"""

    prefix_name = "sample"
    if args.prefix:
        prefix_path = Path(args.prefix).expanduser()
        prefix_name = prefix_path.name
    else:
        prefix_path = None

    bok = args.bok.expanduser() if args.bok is not None else None
    if bok is None and prefix_path is not None:
        bok = find_prefixed_image(prefix_path, "_bok")
    if bok is None:
        raise ValueError("必须提供 --bok，或提供能找到 <prefix>_bok.* 的 --prefix")
    if prefix_path is None:
        prefix_name = bok.stem[:-4] if bok.stem.casefold().endswith("_bok") else bok.stem

    mask = args.mask
    alpha = args.alpha
    if mask is None:
        if args.fine_dir is None:
            raise ValueError("未显式提供 --mask 时，必须用 --fine-dir 指定细发丝结果目录")
        fine_dir = args.fine_dir.expanduser()
        mask = first_existing(
            fine_dir,
            (
                "fine_hair_mask_01.png",
                "fine_hair_mask.png",
                f"{prefix_name}_fine_hair_mask_01.png",
                f"{prefix_name}_fine_hair_mask.png",
            ),
        )
        if mask is None:
            raise FileNotFoundError(f"在细发丝结果目录中找不到二值 mask：{fine_dir}")
        if alpha is None:
            alpha = first_existing(
                fine_dir,
                (
                    "fine_hair_alpha_16bit.png",
                    f"{prefix_name}_fine_hair_alpha_16bit.png",
                ),
            )

    return Path(bok), Path(mask), Path(alpha) if alpha is not None else None, prefix_name


def compute_roi(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    """计算 mask 外接框并按滤波上下文外扩。"""

    height, width = mask.shape
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, width, height
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin + 1),
        min(height, int(ys.max()) + margin + 1),
    )


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """把 0～1 sRGB 转为线性光空间，减轻羽化处的暗边。"""

    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """把线性光空间转换回 0～1 sRGB。"""

    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.maximum(image, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def build_processing_masks(
    binary_mask: np.ndarray,
    alpha_mask: np.ndarray | None,
    alpha_threshold: float,
    expand_radius: int,
    feather_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成发丝核心、背景重建区域和柔和融合权重。"""

    core = binary_mask.copy()
    if alpha_mask is not None:
        core |= alpha_mask >= alpha_threshold

    if expand_radius > 0:
        inpaint_mask = cv2.dilate(
            core.astype(np.uint8),
            ellipse(expand_radius),
        ).astype(bool)
    else:
        inpaint_mask = core.copy()

    # 高斯羽化只影响发丝周围数像素；原始 mask 像素强制完全替换，避免残留亮丝。
    blend_weight = inpaint_mask.astype(np.float32)
    if feather_sigma > 0.0:
        blend_weight = cv2.GaussianBlur(
            blend_weight,
            (0, 0),
            sigmaX=feather_sigma,
            sigmaY=feather_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
    blend_weight = np.clip(blend_weight, 0.0, 1.0)
    blend_weight[core] = 1.0
    return core, inpaint_mask, blend_weight


def reconstruct_background(
    bok_roi: np.ndarray,
    inpaint_mask: np.ndarray,
    method: str,
    inpaint_radius: float,
    inpaint_backend: str,
    median_kernel: int,
    background_blur_sigma: float,
    background_blur_strength: float,
) -> np.ndarray:
    """重建细发丝后方背景，再做轻微低通以贴近 BOK 的模糊形态。"""

    mask_u8 = inpaint_mask.astype(np.uint8) * 255
    if method in {"inpaint", "hybrid"}:
        flag = cv2.INPAINT_TELEA if inpaint_backend == "telea" else cv2.INPAINT_NS
        inpainted = cv2.inpaint(bok_roi, mask_u8, inpaint_radius, flag)
    else:
        inpainted = bok_roi.copy()

    if method in {"median", "hybrid"}:
        shortest = min(bok_roi.shape[:2])
        effective_kernel = min(median_kernel, shortest if shortest % 2 == 1 else shortest - 1)
        if effective_kernel <= 1:
            raise ValueError("ROI 太小，无法执行中值滤波")
        median = cv2.medianBlur(bok_roi, effective_kernel)
        if method == "median":
            reconstructed = median
        else:
            # 中值结果只占较小权重，用于压掉 Telea 偶尔产生的细亮纹。
            reconstructed = cv2.addWeighted(inpainted, 0.82, median, 0.18, 0.0)
    else:
        reconstructed = inpainted

    if background_blur_sigma > 0.0 and background_blur_strength > 0.0:
        smoothed = cv2.GaussianBlur(
            reconstructed,
            (0, 0),
            sigmaX=background_blur_sigma,
            sigmaY=background_blur_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
        reconstructed = cv2.addWeighted(
            reconstructed,
            1.0 - background_blur_strength,
            smoothed,
            background_blur_strength,
            0.0,
        )
    return reconstructed


def feather_blend(
    original: np.ndarray,
    reconstructed: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """在线性光空间羽化融合，保持 mask 外像素逐值不变。"""

    source = original.astype(np.float32) / 255.0
    target = reconstructed.astype(np.float32) / 255.0
    alpha = weight[..., None]
    blended_linear = (
        srgb_to_linear(source) * (1.0 - alpha)
        + srgb_to_linear(target) * alpha
    )
    blended = np.clip(linear_to_srgb(blended_linear), 0.0, 1.0)
    result = np.rint(blended * 255.0).astype(np.uint8)
    result[weight <= 0.0] = original[weight <= 0.0]
    return result


def make_comparison(original: np.ndarray, result: np.ndarray) -> np.ndarray:
    """生成原图与处理结果的左右对比图。"""

    comparison = np.concatenate((original, result), axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, original.shape[1] / 1800.0)
    thickness = max(2, int(round(scale * 2)))
    cv2.putText(comparison, "BOK", (24, 52), font, scale, (0, 0, 0), thickness + 3)
    cv2.putText(comparison, "BOK", (24, 52), font, scale, (255, 255, 255), thickness)
    offset = original.shape[1]
    cv2.putText(
        comparison,
        "HAIR BOKEH",
        (offset + 24, 52),
        font,
        scale,
        (0, 0, 0),
        thickness + 3,
    )
    cv2.putText(
        comparison,
        "HAIR BOKEH",
        (offset + 24, 52),
        font,
        scale,
        (255, 255, 255),
        thickness,
    )
    return comparison


def process(args: argparse.Namespace) -> dict:
    """执行细发丝背景化并保存完整分辨率结果。"""

    bok_path, mask_path, alpha_path, prefix_name = resolve_inputs(args)
    bok = read_color(bok_path)
    binary = read_float_mask(mask_path) >= args.mask_threshold
    alpha = read_float_mask(alpha_path) if alpha_path is not None else None

    if binary.shape != bok.shape[:2]:
        raise ValueError(f"BOK 与二值 mask 尺寸不一致：{bok.shape[:2]} vs {binary.shape}")
    if alpha is not None and alpha.shape != binary.shape:
        raise ValueError(f"二值 mask 与 alpha 尺寸不一致：{binary.shape} vs {alpha.shape}")
    if alpha is not None and binary.any():
        alpha_support_pixels = int(np.count_nonzero(alpha >= args.alpha_threshold))
        binary_pixels = int(binary.sum())
        if alpha_support_pixels > max(binary_pixels * 8, binary_pixels + 10_000):
            raise ValueError(
                "--alpha 的有效区域远大于细发丝 mask；请传入 fine_hair_alpha_16bit.png，"
                "不要传入完整人像 matte"
            )
    if not binary.any() and (alpha is None or not np.any(alpha >= args.alpha_threshold)):
        raise ValueError("细发丝 mask 为空，没有需要处理的区域")

    support = binary.copy()
    if alpha is not None:
        support |= alpha >= args.alpha_threshold
    filter_halo = max(
        24,
        args.expand_radius + int(np.ceil(4.0 * args.feather_sigma)) + 4,
        args.expand_radius + int(np.ceil(4.0 * args.background_blur_sigma)) + 4,
        args.median_kernel // 2 + args.expand_radius + 4,
        int(np.ceil(args.inpaint_radius)) + args.expand_radius + 8,
    )
    roi_margin = max(args.roi_margin, filter_halo) if args.roi_margin >= 0 else filter_halo
    if args.no_roi:
        x0, y0, x1, y1 = 0, 0, bok.shape[1], bok.shape[0]
    else:
        x0, y0, x1, y1 = compute_roi(support, roi_margin)
    roi = np.s_[y0:y1, x0:x1]

    bok_roi = bok[roi]
    binary_roi = binary[roi]
    alpha_roi = alpha[roi] if alpha is not None else None
    core, inpaint_mask, blend_weight = build_processing_masks(
        binary_roi,
        alpha_roi,
        args.alpha_threshold,
        args.expand_radius,
        args.feather_sigma,
    )
    reconstructed = reconstruct_background(
        bok_roi,
        inpaint_mask,
        args.method,
        args.inpaint_radius,
        args.inpaint_backend,
        args.median_kernel,
        args.background_blur_sigma,
        args.background_blur_strength,
    )
    result_roi = feather_blend(bok_roi, reconstructed, blend_weight)

    result = bok.copy()
    result[roi] = result_roi
    output_dir = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "hair_bokeh.png", result)
    write_image(
        output_dir / "hair_bokeh_compare.jpg",
        make_comparison(bok, result),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )

    if not args.no_debug:
        debug_dir = output_dir / "debug"
        write_image(debug_dir / "00_bok_roi.jpg", bok_roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_image(debug_dir / "01_fine_hair_core.png", core.astype(np.uint8) * 255)
        write_image(debug_dir / "02_inpaint_region.png", inpaint_mask.astype(np.uint8) * 255)
        write_image(
            debug_dir / "03_blend_weight_16bit.png",
            np.rint(blend_weight * 65535.0).astype(np.uint16),
        )
        write_image(debug_dir / "04_reconstructed_background.png", reconstructed)
        difference = cv2.absdiff(bok_roi, result_roi)
        write_image(debug_dir / "05_absolute_difference.png", difference)
        write_image(debug_dir / "06_hair_bokeh_roi.png", result_roi)

    changed = np.any(result != bok, axis=2)
    metadata = {
        "prefix": prefix_name,
        "inputs": {
            "bok": str(bok_path),
            "mask": str(mask_path),
            "alpha": str(alpha_path) if alpha_path is not None else None,
        },
        "resolution_wh": [int(bok.shape[1]), int(bok.shape[0])],
        "roi_bbox_xyxy": [x0, y0, x1, y1],
        "roi_enabled": not args.no_roi,
        "parameters": {
            "method": args.method,
            "mask_threshold": args.mask_threshold,
            "alpha_threshold": args.alpha_threshold,
            "expand_radius": args.expand_radius,
            "feather_sigma": args.feather_sigma,
            "inpaint_radius": args.inpaint_radius,
            "inpaint_backend": args.inpaint_backend,
            "median_kernel": args.median_kernel,
            "background_blur_sigma": args.background_blur_sigma,
            "background_blur_strength": args.background_blur_strength,
        },
        "pixel_counts": {
            "fine_hair_core": int(core.sum()),
            "inpaint_region": int(inpaint_mask.sum()),
            "changed_full_image": int(changed.sum()),
        },
        "outputs": {
            "hair_bokeh": str(output_dir / "hair_bokeh.png"),
            "comparison": str(output_dir / "hair_bokeh_compare.jpg"),
        },
    }
    (output_dir / "hair_bokeh_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[{prefix_name}] 完成：method={args.method}，ROI={x1 - x0}x{y1 - y0}，"
        f"发丝核心={int(core.sum())} 像素，输出={output_dir}"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """构造中文命令行接口。"""

    parser = argparse.ArgumentParser(
        description=(
            "在 BOK 图上将细小发丝 mask 区域重建为邻近背景，并用柔和虚化与羽化融合"
            "避免突兀边界。"
        )
    )
    parser.add_argument("--prefix", help="公共输入前缀，例如 D:\\base\\2p")
    parser.add_argument("--bok", type=Path, help="显式指定 BOK 图像")
    parser.add_argument("--mask", type=Path, help="显式指定细发丝 0/1 或 0/255 mask")
    parser.add_argument("--alpha", type=Path, help="可选的细发丝 16-bit alpha")
    parser.add_argument(
        "--fine-dir",
        type=Path,
        help="自动寻找 fine_hair_mask_01.png 和 fine_hair_alpha_16bit.png 的目录",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument(
        "--method",
        choices=("inpaint", "hybrid", "median"),
        default="inpaint",
        help="背景重建方法；默认 inpaint，median 仅用于快速对照",
    )
    parser.add_argument(
        "--inpaint-backend",
        choices=("telea", "ns"),
        default="telea",
        help="OpenCV 背景重建算法",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="二值 mask 阈值")
    parser.add_argument("--alpha-threshold", type=float, default=0.008, help="alpha 补充支持阈值")
    parser.add_argument("--expand-radius", type=int, default=2, help="去除抗锯齿发丝边缘的外扩半径")
    parser.add_argument("--feather-sigma", type=float, default=2.2, help="融合边缘高斯羽化 sigma")
    parser.add_argument("--inpaint-radius", type=float, default=3.0, help="背景重建邻域半径")
    parser.add_argument("--median-kernel", type=int, default=15, help="median/hybrid 的中值核，必须为奇数")
    parser.add_argument(
        "--background-blur-sigma",
        type=float,
        default=1.6,
        help="重建背景的轻微高斯虚化 sigma",
    )
    parser.add_argument(
        "--background-blur-strength",
        type=float,
        default=0.65,
        help="轻微高斯虚化混合强度",
    )
    parser.add_argument("--roi-margin", type=int, default=-1, help="ROI 外扩下限；-1 表示自动")
    parser.add_argument("--no-roi", action="store_true", help="关闭 ROI 裁剪，按完整分辨率处理")
    parser.add_argument("--no-debug", action="store_true", help="不保存数字编号调试图")
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """检查输入模式和数值参数。"""

    if args.prefix is None and args.bok is None:
        parser.error("必须提供 --prefix 或 --bok")
    if args.mask is None and args.fine_dir is None:
        parser.error("必须提供 --mask，或通过 --fine-dir 自动发现 mask")
    if args.expand_radius < 0 or args.roi_margin < -1:
        parser.error("--expand-radius 必须大于等于 0；--roi-margin 必须为 -1 或非负数")
    if args.feather_sigma < 0.0 or args.background_blur_sigma < 0.0:
        parser.error("sigma 参数必须大于等于 0")
    if args.inpaint_radius <= 0.0:
        parser.error("--inpaint-radius 必须大于 0")
    if args.median_kernel <= 1 or args.median_kernel % 2 == 0:
        parser.error("--median-kernel 必须是大于 1 的奇数")
    for name in ("mask_threshold", "alpha_threshold", "background_blur_strength"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} 必须位于 0～1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args, parser)
    try:
        process(args)
    except (cv2.error, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
