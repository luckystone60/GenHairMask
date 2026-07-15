from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# 批处理会按此顺序查找图像；优先使用无损 PNG。
IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp")
MEDIAN_KERNELS = (7, 11, 17, 25, 35)
REFERENCE_AREA = 2509 * 3760
HAIR_CLASS_ID = 4

# 同时兼容 prepare_base_images.py 的规范命名和 run_pipeline.py 的原始命名。
ROLE_SUFFIXES = {
    "bok": ("_bok",),
    "edof": ("_edof",),
    "matte": ("_biref_bok_mat4k", "_bok_mat4k", "_bok_biref_bok_mat4k"),
    "hair": ("_hair_mask", "_bok_hair", "_hair"),
    "hair_probability": (
        "_hair_probability_16bit",
        "_bok_hair_probability_16bit",
        "_bok_sapiens2_hair_probability_16bit",
        "_sapiens2_hair_probability_16bit",
    ),
    "sapiens2_labels": ("_sapiens2_labels", "_bok_sapiens2_labels"),
}


@dataclass(frozen=True)
class SampleFiles:
    """一组同坐标系输入文件。"""

    prefix: str
    bok: Path
    edof: Path
    matte: Path
    hair: Path | None = None
    hair_probability: Path | None = None
    sapiens2_labels: Path | None = None


def read_color(path: Path) -> np.ndarray:
    """读取三通道 BGR 图像。"""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取彩色图像：{path}")
    return image


def read_float_mask(path: Path) -> np.ndarray:
    """读取灰度 mask，并明确转换到 0～1 浮点范围。"""

    src = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(f"无法读取 mask：{path}")
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if src.dtype == np.uint16:
        if int(src.max(initial=0)) <= 1:
            return src.astype(np.float32)
        return src.astype(np.float32) / 65535.0
    if src.dtype == np.uint8:
        if int(src.max(initial=0)) <= 1:
            return src.astype(np.float32)
        return src.astype(np.float32) / 255.0

    src = src.astype(np.float32)
    if not np.isfinite(src).all():
        raise ValueError(f"mask 含有 NaN 或 Inf：{path}")
    maximum = float(src.max(initial=0.0))
    minimum = float(src.min(initial=0.0))
    if minimum < 0:
        raise ValueError(f"mask 含有负值：{path}")
    if maximum <= 1.0:
        return np.clip(src, 0.0, 1.0)
    if maximum <= 255.0:
        return np.clip(src / 255.0, 0.0, 1.0)
    if maximum <= 65535.0:
        return np.clip(src / 65535.0, 0.0, 1.0)
    raise ValueError(f"无法判断浮点 mask 的数值范围：{path}，最大值={maximum}")


def read_labels(path: Path) -> np.ndarray:
    """读取 Sapiens2 类别标签图。"""

    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError(f"无法读取 Sapiens2 标签图：{path}")
    if labels.ndim == 3:
        labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
    return labels


def ellipse(radius: int) -> np.ndarray:
    """生成指定半径的椭圆形形态学核。"""

    radius = max(int(radius), 0)
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )


def robust_normalize(
    values: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 99.0,
) -> np.ndarray:
    """只在有效搜索区内用百分位数归一化，避免极少数强边缘支配范围。"""

    selected = values[valid_mask & (values > 0)]
    scale = float(np.percentile(selected, percentile)) if selected.size else 1.0
    return np.clip(values / max(scale, 1e-6), 0.0, 1.0)


def median_line_response(
    bgr: np.ndarray,
    search: np.ndarray,
    kernels: tuple[int, ...] = MEDIAN_KERNELS,
) -> np.ndarray:
    """计算多尺度中值残差，用于响应细线和单根发丝。"""

    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    src = lum.astype(np.float32)
    response = np.zeros(src.shape, np.float32)
    shortest = min(lum.shape)
    valid_kernels = tuple(k for k in kernels if 1 < k <= shortest and k % 2 == 1)
    for kernel_size in valid_kernels:
        median = cv2.medianBlur(lum, kernel_size).astype(np.float32)
        response = np.maximum(response, np.abs(src - median))
    return robust_normalize(response, search, 99.25)


def blur_loss_map(bok: np.ndarray, edof: np.ndarray) -> np.ndarray:
    """比较 BOK 与 EDOF 的局部梯度，生成被虚化背景的负证据。"""

    def gradient(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        return cv2.dilate(magnitude, ellipse(4))

    gradient_bok = gradient(bok)
    gradient_edof = gradient(edof)
    loss = np.clip(
        (gradient_edof - 1.12 * gradient_bok) / (gradient_edof + 5.0),
        0.0,
        1.0,
    )
    return loss * np.clip((gradient_edof - 5.0) / 25.0, 0.0, 1.0)


def keep_seed_connected(weak: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """仅保留与指定种子八连通的弱候选。"""

    joined = (weak | seed).astype(np.uint8)
    count, labels = cv2.connectedComponents(joined, connectivity=8)
    if count <= 1:
        return np.zeros_like(weak)
    label_ids = np.unique(labels[seed])
    label_ids = label_ids[label_ids != 0]
    return np.isin(labels, label_ids) & weak


def filter_components(
    mask: np.ndarray,
    score: np.ndarray,
    min_area: int,
    min_extent: float,
    min_score: float,
) -> np.ndarray:
    """按面积、外接框尺度和最高置信度过滤连通域。"""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros(mask.shape, bool)
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        extent = float(np.hypot(width, height))
        pixels = labels == label_id
        if (
            area >= min_area
            and extent >= min_extent
            and float(score[pixels].max(initial=0.0)) >= min_score
        ):
            keep[pixels] = True
    return keep


def bridge_directional_gaps(
    mask: np.ndarray,
    allowed_region: np.ndarray,
    score: np.ndarray,
    line: np.ndarray,
    alpha: np.ndarray,
    blur_loss: np.ndarray,
    radius: int,
    score_min: float,
    line_min: float,
    alpha_min: float,
    blur_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在四个主方向上补齐短缺口，并严格限制新增像素的证据与范围。"""

    if radius <= 0 or not mask.any():
        return mask.copy(), np.zeros_like(mask)

    size = radius * 2 + 1
    kernels = (
        np.ones((1, size), np.uint8),
        np.ones((size, 1), np.uint8),
        np.eye(size, dtype=np.uint8),
        np.fliplr(np.eye(size, dtype=np.uint8)),
    )

    # 只用细发丝结果本身做闭运算，不让粗 Hair mask 或其他语义区域参与连接。
    source = mask.astype(np.uint8)
    closed = mask.copy()
    for kernel in kernels:
        closed |= cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel).astype(bool)

    # 新增桥接像素必须来自已经通过连接性、alpha 和语义排除的 before_thin。
    # score 较高可以直接通过；较弱像素必须同时具有线状响应和最低 alpha。
    evidence = (
        (score >= score_min)
        | ((line >= line_min) & (alpha >= alpha_min))
    ) & (blur_loss <= blur_max)
    bridge = closed & ~mask & allowed_region & evidence
    return mask | bridge, bridge


def clean_hair_seed(seed: np.ndarray, requested_min_area: int) -> tuple[np.ndarray, int]:
    """清理孤立语义误检，并按分辨率降低小图的面积门槛。"""

    if requested_min_area <= 1:
        return seed.copy(), max(requested_min_area, 0)
    scaled = int(round(requested_min_area * seed.size / REFERENCE_AREA))
    effective_min_area = max(1, min(requested_min_area, scaled))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed.astype(np.uint8),
        connectivity=8,
    )
    clean = np.zeros_like(seed)
    largest_id = 0
    largest_area = 0
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_id = label_id
            largest_area = area
        if area >= effective_min_area:
            clean[labels == label_id] = True

    # 小图上如果所有组件都略低于阈值，至少保留最大组件，避免整批意外清空。
    if not clean.any() and largest_id != 0:
        clean[labels == largest_id] = True
    return clean, effective_min_area


def compute_roi(
    support: np.ndarray,
    width: int,
    height: int,
    margin: int,
    disable_roi: bool,
) -> tuple[int, int, int, int]:
    """由支持区域计算外扩后的开区间 ROI 坐标。"""

    if disable_roi:
        return 0, 0, width, height
    ys, xs = np.where(support)
    if xs.size == 0:
        return 0, 0, width, height
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(width, int(xs.max()) + margin + 1)
    y1 = min(height, int(ys.max()) + margin + 1)
    return x0, y0, x1, y1


def write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    """可靠写图，写入失败时立即报错。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, params or []):
        raise OSError(f"图像写入失败：{path}")


def save_binary_01(path: Path, mask: np.ndarray) -> None:
    """保存像素值严格为 0 或 1 的 uint8 PNG。"""

    write_image(path, mask.astype(np.uint8))


def save_binary_visual(path: Path, mask: np.ndarray) -> None:
    """保存便于普通看图软件查看的 0/255 二值 PNG。"""

    write_image(path, mask.astype(np.uint8) * 255)


def save_u8(path: Path, values: np.ndarray) -> None:
    """把 0～1 浮点图保存为 8-bit PNG。"""

    image = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    write_image(path, image)


def save_u16(path: Path, values: np.ndarray) -> None:
    """把 0～1 浮点图保存为 16-bit PNG。"""

    image = np.rint(np.clip(values, 0.0, 1.0) * 65535.0).astype(np.uint16)
    write_image(path, image)


def build_image_index(input_dir: Path) -> dict[str, Path]:
    """建立大小写不敏感的当前目录图像索引。"""

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在：{input_dir}")
    return {
        path.name.casefold(): path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    }


def find_by_suffix(
    index: dict[str, Path],
    prefix: str,
    suffixes: tuple[str, ...],
) -> Path | None:
    """按后缀优先级和扩展名优先级查找输入文件。"""

    for extension in IMAGE_EXTENSIONS:
        for suffix in suffixes:
            candidate = f"{prefix}{suffix}{extension}".casefold()
            if candidate in index:
                return index[candidate]
    return None


def explicit_or_inferred(
    explicit: Path | None,
    index: dict[str, Path],
    prefix: str,
    role: str,
    required: bool,
) -> Path | None:
    """优先采用显式路径，否则按 prefix 和角色后缀自动发现。"""

    if explicit is not None:
        path = explicit.expanduser()
        if path.is_file():
            return path
        if required:
            raise FileNotFoundError(f"显式指定的 {role} 文件不存在：{path}")
        print(f"[警告] 可选 {role} 文件不存在，将启用后备逻辑：{path}")
        return None
    path = find_by_suffix(index, prefix, ROLE_SUFFIXES[role])
    if path is None and required:
        expected = "、".join(f"{prefix}{suffix}.png" for suffix in ROLE_SUFFIXES[role])
        raise FileNotFoundError(f"缺少 {role} 输入，尝试过：{expected}")
    return path


def resolve_sample(
    input_dir: Path,
    prefix: str,
    args: argparse.Namespace,
    allow_explicit: bool,
) -> SampleFiles:
    """解析一个前缀对应的全部输入路径。"""

    if not prefix or prefix in {".", ".."}:
        raise ValueError("prefix 不能为空")
    index = build_image_index(input_dir)
    explicit = {
        "bok": args.bok if allow_explicit else None,
        "edof": args.edof if allow_explicit else None,
        "matte": args.matte if allow_explicit else None,
        "hair": args.hair if allow_explicit else None,
        "hair_probability": args.hair_probability if allow_explicit else None,
        "sapiens2_labels": args.sapiens2_labels if allow_explicit else None,
    }
    return SampleFiles(
        prefix=prefix,
        bok=explicit_or_inferred(explicit["bok"], index, prefix, "bok", True),
        edof=explicit_or_inferred(explicit["edof"], index, prefix, "edof", True),
        matte=explicit_or_inferred(explicit["matte"], index, prefix, "matte", True),
        hair=explicit_or_inferred(explicit["hair"], index, prefix, "hair", False),
        hair_probability=explicit_or_inferred(
            explicit["hair_probability"],
            index,
            prefix,
            "hair_probability",
            False,
        ),
        sapiens2_labels=explicit_or_inferred(
            explicit["sapiens2_labels"],
            index,
            prefix,
            "sapiens2_labels",
            False,
        ),
    )


def discover_prefixes(input_dir: Path) -> list[str]:
    """从目录内所有 <prefix>_bok.* 图像发现批处理任务。"""

    prefixes: dict[str, str] = {}
    for path in input_dir.iterdir():
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem.casefold().endswith("_bok") and len(stem) > 4:
            prefix = stem[:-4]
            prefixes.setdefault(prefix.casefold(), prefix)
    return sorted(prefixes.values(), key=str.casefold)


def sample_paths_to_json(sample: SampleFiles) -> dict[str, str | None]:
    """把输入路径转换成可序列化字典。"""

    return {
        "prefix": sample.prefix,
        "bok": str(sample.bok),
        "edof": str(sample.edof),
        "matte": str(sample.matte),
        "hair": str(sample.hair) if sample.hair else None,
        "hair_probability": (
            str(sample.hair_probability) if sample.hair_probability else None
        ),
        "sapiens2_labels": (
            str(sample.sapiens2_labels) if sample.sapiens2_labels else None
        ),
    }


def make_blending(
    bok: np.ndarray,
    mask: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """用红色把严格 01 mask 混合到完整 BOK，生成质检预览图。"""

    tint = np.array([0.0, 0.0, 255.0], np.float32)
    alpha = (mask.astype(np.float32) * opacity)[..., None]
    blended = bok.astype(np.float32) * (1.0 - alpha) + tint * alpha
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def run_algorithm(
    bok: np.ndarray,
    edof: np.ndarray,
    alpha: np.ndarray,
    semantic_probability: np.ndarray,
    labels: np.ndarray | None,
    hair_seed: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """在已经裁剪的 ROI 内执行细碎边缘发丝检测。"""

    distance_to_support = cv2.distanceTransform(
        (~hair_seed).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    search = distance_to_support <= float(args.outer_radius)

    # 只保留人像与头发主体边缘，不允许大片不透明内部进入最终结果。
    person_core = cv2.erode(
        (alpha >= 0.72).astype(np.uint8),
        ellipse(10),
    ).astype(bool)
    coarse_exclusion = cv2.erode(
        hair_seed.astype(np.uint8),
        ellipse(args.inner_band),
    ).astype(bool)
    if labels is not None:
        nonhair_person = (labels != 0) & (labels != HAIR_CLASS_ID)
        nonhair_exclusion = cv2.dilate(
            nonhair_person.astype(np.uint8),
            ellipse(args.nonhair_radius),
        ).astype(bool)
    else:
        nonhair_exclusion = np.zeros_like(search)
    fringe_zone = search & ~person_core & ~coarse_exclusion & ~nonhair_exclusion

    line = median_line_response(bok, search)
    blur_loss = blur_loss_map(bok, edof)
    semantic = cv2.dilate(semantic_probability, ellipse(18))
    alpha_edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, ellipse(2))
    alpha_edge = robust_normalize(alpha_edge, search, 99.0)

    score = (
        0.47 * np.sqrt(np.clip(alpha, 0.0, 1.0))
        + 0.27 * line
        + 0.12 * np.sqrt(np.clip(semantic, 0.0, 1.0))
        + 0.14 * alpha_edge
        - 0.24 * blur_loss
    )
    score = np.clip(score, 0.0, 1.0) * fringe_zone
    evidence = (alpha >= args.alpha_min) & (
        (line >= 0.055) | (alpha_edge >= 0.10) | (alpha >= 0.22)
    )
    weak = fringe_zone & evidence & (score >= args.low_score)
    strong = fringe_zone & evidence & (
        (score >= args.high_score) | ((alpha >= 0.45) & (line >= 0.08))
    )

    # 允许候选通过一像素桥连接到语义头发种子，随后再移除种子主体。
    weak_bridge = cv2.dilate(
        weak.astype(np.uint8),
        ellipse(1),
    ).astype(bool)
    connection_seed = cv2.dilate(
        hair_seed.astype(np.uint8),
        ellipse(3),
    ).astype(bool)
    connected = keep_seed_connected(weak_bridge, connection_seed)
    connected &= weak
    hysteresis = keep_seed_connected(connected, strong)
    before_thin = filter_components(
        hysteresis,
        score,
        args.min_area,
        args.min_extent,
        args.component_score_min,
    )

    if args.thin_radius > 0:
        broad = cv2.morphologyEx(
            before_thin.astype(np.uint8),
            cv2.MORPH_OPEN,
            ellipse(args.thin_radius),
        ).astype(bool)
        final = before_thin & ~broad
        final = filter_components(
            final,
            score,
            args.min_area,
            args.min_extent,
            args.component_score_min,
        )
    else:
        broad = np.zeros_like(before_thin)
        final = before_thin

    # 宽结构剔除容易把一根发丝切成几段；仅在已验证候选内部补回四方向短缺口。
    # 这一步不会向 before_thin 之外生长，因此比直接膨胀或全局降低阈值更稳健。
    final, gap_bridge = bridge_directional_gaps(
        final,
        before_thin,
        score,
        line,
        alpha,
        blur_loss,
        args.gap_close_radius,
        args.gap_score_min,
        args.gap_line_min,
        args.gap_alpha_min,
        args.gap_blur_max,
    )

    # Alpha 保留 BiRefNet 的覆盖率；score 是检测置信度，二者不能混用。
    fine_alpha = alpha * final.astype(np.float32)
    return {
        "search": search,
        "person_core": person_core,
        "coarse_exclusion": coarse_exclusion,
        "nonhair_exclusion": nonhair_exclusion,
        "fringe_zone": fringe_zone,
        "line": line,
        "blur_loss": blur_loss,
        "alpha_edge": alpha_edge,
        "score": score,
        "weak": weak,
        "strong": strong,
        "connected": hysteresis,
        "before_thin": before_thin,
        "broad": broad,
        "gap_bridge": gap_bridge,
        "final": final,
        "fine_alpha": fine_alpha,
    }


def save_debug(
    output_dir: Path,
    bok_roi: np.ndarray,
    roi_support: np.ndarray,
    hair_seed: np.ndarray,
    stages: dict[str, np.ndarray],
) -> None:
    """按固定数字步骤保存 ROI 尺寸调试图。"""

    debug_dir = output_dir / "debug"
    write_image(
        debug_dir / "00_bok_roi.jpg",
        bok_roi,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    save_binary_visual(debug_dir / "01_roi_support.png", roi_support)
    save_binary_visual(debug_dir / "02_hair_seed.png", hair_seed)
    save_binary_visual(debug_dir / "03_search_region.png", stages["search"])
    save_binary_visual(debug_dir / "04_person_core.png", stages["person_core"])
    save_binary_visual(
        debug_dir / "05_nonhair_exclusion.png",
        stages["nonhair_exclusion"],
    )
    save_binary_visual(debug_dir / "06_fringe_zone.png", stages["fringe_zone"])
    save_u8(debug_dir / "07_line_response.png", stages["line"])
    save_u8(debug_dir / "08_edof_blur_loss.png", stages["blur_loss"])
    save_u8(debug_dir / "09_alpha_edge.png", stages["alpha_edge"])
    save_u16(debug_dir / "10_score_16bit.png", stages["score"])
    save_binary_visual(debug_dir / "11_weak_candidate.png", stages["weak"])
    save_binary_visual(debug_dir / "12_strong_candidate.png", stages["strong"])
    save_binary_visual(debug_dir / "13_connected_candidate.png", stages["connected"])
    save_binary_visual(
        debug_dir / "14_before_thin_filter.png",
        stages["before_thin"],
    )
    save_binary_visual(debug_dir / "15_removed_broad.png", stages["broad"])
    save_binary_visual(
        debug_dir / "16_directional_gap_bridge.png",
        stages["gap_bridge"],
    )
    save_binary_visual(debug_dir / "17_final_mask_visual.png", stages["final"])


def process_one(
    sample: SampleFiles,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """处理一个样本，并把 ROI 结果回填到完整 BOK 分辨率。"""

    bok = read_color(sample.bok)
    edof = read_color(sample.edof)
    alpha = read_float_mask(sample.matte)
    labels = read_labels(sample.sapiens2_labels) if sample.sapiens2_labels else None
    hair = read_float_mask(sample.hair) if sample.hair else None
    probability = (
        read_float_mask(sample.hair_probability)
        if sample.hair_probability
        else None
    )

    height, width = alpha.shape
    checked = [bok, edof]
    checked.extend(x for x in (hair, probability, labels) if x is not None)
    if any(image.shape[:2] != (height, width) for image in checked):
        shapes = [image.shape[:2] for image in checked]
        raise ValueError(
            f"[{sample.prefix}] 输入尺寸不一致，matte={(height, width)}，其他={shapes}"
        )

    hair_is_valid = hair is not None and bool(
        np.any(hair >= args.roi_threshold)
    )
    if hair_is_valid:
        roi_source_full = hair
        hair_mask_source = "hair_mask"
    else:
        roi_source_full = alpha
        hair_mask_source = "matte_fallback"
        reason = "缺少 hair mask" if hair is None else "hair mask 为空"
        print(
            f"[{sample.prefix}] 警告：{reason}，使用 matting 作为 ROI 后备；"
            "仅靠 matte 时更容易把人像轮廓误判为发丝。"
        )

    # 种子优先使用 Hair 语义；只有所有 Hair 先验均缺失时才用 matte。
    seed_parts: list[tuple[str, np.ndarray]] = []
    if labels is not None:
        seed_parts.append(("sapiens2_labels", labels == HAIR_CLASS_ID))
    if probability is not None:
        seed_parts.append(
            (
                "hair_probability",
                probability >= args.hair_prob_threshold,
            )
        )
    if hair_is_valid:
        seed_parts.append(("hair_mask", hair >= 0.5))

    if seed_parts:
        raw_seed = np.zeros((height, width), bool)
        for _, seed in seed_parts:
            raw_seed |= seed
        semantic_names = "+".join(name for name, _ in seed_parts)
        if raw_seed.any():
            semantic_seed_source = semantic_names
        else:
            raw_seed = alpha >= args.matte_seed_threshold
            semantic_seed_source = f"matte_fallback(empty:{semantic_names})"
            print(
                f"[{sample.prefix}] 警告：Hair 先验文件存在但没有有效 Hair 像素，"
                "改用 matte 构造种子。"
            )
    else:
        raw_seed = alpha >= args.matte_seed_threshold
        semantic_seed_source = "matte_fallback"

    hair_seed_full, effective_seed_area = clean_hair_seed(
        raw_seed,
        args.min_seed_area,
    )

    if not hair_seed_full.any() and args.empty_policy == "error":
        raise ValueError(f"[{sample.prefix}] 无法得到有效种子")

    filter_halo = max(
        24,
        max(MEDIAN_KERNELS) // 2 + 6,
        18 + args.nonhair_radius,
        args.inner_band + args.thin_radius + 4,
        args.gap_close_radius * 2 + 4,
    )
    safe_roi_margin = args.outer_radius + filter_halo
    if 0 <= args.roi_margin < safe_roi_margin:
        print(
            f"[{sample.prefix}] 警告：--roi-margin={args.roi_margin} 小于安全值 "
            f"{safe_roi_margin}，已自动提升以避免截断滤波上下文。"
        )
    roi_margin = (
        max(args.roi_margin, safe_roi_margin)
        if args.roi_margin >= 0
        else safe_roi_margin
    )
    # ROI 同时覆盖粗 Hair/matte 支持和最终实际种子，防止语义种子被裁掉。
    roi_support_full = (roi_source_full >= args.roi_threshold) | hair_seed_full
    if not roi_support_full.any():
        roi_support_full = hair_seed_full.copy()
    x0, y0, x1, y1 = compute_roi(
        roi_support_full,
        width,
        height,
        roi_margin,
        args.no_roi,
    )
    roi = np.s_[y0:y1, x0:x1]

    bok_roi = bok[roi]
    edof_roi = edof[roi]
    alpha_roi = alpha[roi]
    labels_roi = labels[roi] if labels is not None else None
    hair_seed_roi = hair_seed_full[roi]
    roi_support = roi_support_full[roi]

    if probability is not None:
        semantic_probability_full = probability
    elif hair_is_valid:
        semantic_probability_full = hair
    elif labels is not None:
        semantic_probability_full = (labels == HAIR_CLASS_ID).astype(np.float32)
    else:
        semantic_probability_full = alpha
    semantic_probability_roi = semantic_probability_full[roi]

    if hair_seed_roi.any():
        stages = run_algorithm(
            bok_roi,
            edof_roi,
            alpha_roi,
            semantic_probability_roi,
            labels_roi,
            hair_seed_roi,
            args,
        )
        empty_reason = None
    else:
        if args.empty_policy == "error":
            raise ValueError(f"[{sample.prefix}] ROI 内没有有效种子")
        # 默认空种子策略输出全零结果，保证批处理继续且不制造伪 mask。
        zero_bool = np.zeros(alpha_roi.shape, bool)
        zero_float = np.zeros(alpha_roi.shape, np.float32)
        stages = {
            "search": zero_bool,
            "person_core": zero_bool,
            "coarse_exclusion": zero_bool,
            "nonhair_exclusion": zero_bool,
            "fringe_zone": zero_bool,
            "line": zero_float,
            "blur_loss": zero_float,
            "alpha_edge": zero_float,
            "score": zero_float,
            "weak": zero_bool,
            "strong": zero_bool,
            "connected": zero_bool,
            "before_thin": zero_bool,
            "broad": zero_bool,
            "gap_bridge": zero_bool,
            "final": zero_bool,
            "fine_alpha": zero_float,
        }
        empty_reason = "有效种子为空，按 empty-policy=zero 输出空结果"
        print(f"[{sample.prefix}] 警告：{empty_reason}")

    full_final = np.zeros((height, width), bool)
    full_alpha = np.zeros((height, width), np.float32)
    full_score = np.zeros((height, width), np.float32)
    full_final[roi] = stages["final"]
    full_alpha[roi] = stages["fine_alpha"]
    full_score[roi] = stages["score"]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_binary_01(output_dir / "fine_hair_mask_01.png", full_final)
    # 兼容旧消费者：同一二值结果另存为普通看图软件可见的 0/255 PNG。
    save_binary_visual(output_dir / "fine_hair_mask.png", full_final)
    save_u16(output_dir / "fine_hair_alpha_16bit.png", full_alpha)
    save_u16(output_dir / "fine_hair_score_16bit.png", full_score)

    blending = make_blending(bok, full_final, args.blend_opacity)
    write_image(output_dir / "fine_hair_bok_blend.png", blending)
    write_image(
        output_dir / "fine_hair_overlay.jpg",
        blending,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not args.no_debug:
        save_debug(
            output_dir,
            bok_roi,
            roi_support,
            hair_seed_roi,
            stages,
        )

    metadata = {
        "prefix": sample.prefix,
        "algorithm": (
            "BiRefNet alpha 边缘 + Sapiens2 Hair 软先验 + "
            "多尺度中值细线响应 + EDOF/BOK 虚化负证据 + "
            "受证据约束的四方向短缺口连接"
        ),
        "inputs": sample_paths_to_json(sample),
        "fallback": {
            "hair_mask_source": hair_mask_source,
            "semantic_seed_source": semantic_seed_source,
            "empty_reason": empty_reason,
        },
        "resolution_wh": [width, height],
        "roi": {
            "enabled": not args.no_roi,
            "bbox_xyxy": [x0, y0, x1, y1],
            "crop_size_wh": [x1 - x0, y1 - y0],
            "area_fraction": float(
                ((x1 - x0) * (y1 - y0)) / max(width * height, 1)
            ),
            "margin": roi_margin,
            "debug_images_are_roi_sized": True,
        },
        "parameters": {
            "outer_radius": args.outer_radius,
            "inner_band": args.inner_band,
            "nonhair_radius": args.nonhair_radius,
            "thin_radius": args.thin_radius,
            "gap_close_radius": args.gap_close_radius,
            "gap_score_min": args.gap_score_min,
            "gap_line_min": args.gap_line_min,
            "gap_alpha_min": args.gap_alpha_min,
            "gap_blur_max": args.gap_blur_max,
            "hair_prob_threshold": args.hair_prob_threshold,
            "matte_seed_threshold": args.matte_seed_threshold,
            "roi_threshold": args.roi_threshold,
            "low_score": args.low_score,
            "high_score": args.high_score,
            "component_score_min": args.component_score_min,
            "alpha_min": args.alpha_min,
            "min_seed_area_requested": args.min_seed_area,
            "min_seed_area_effective": effective_seed_area,
            "min_area": args.min_area,
            "min_extent": args.min_extent,
            "blend_opacity": args.blend_opacity,
        },
        "pixel_counts": {
            "hair_seed": int(hair_seed_full.sum()),
            "weak_candidate": int(stages["weak"].sum()),
            "before_thin": int(stages["before_thin"].sum()),
            "removed_broad": int(stages["broad"].sum()),
            "directional_gap_bridge": int(stages["gap_bridge"].sum()),
            "final": int(full_final.sum()),
        },
        "outputs": {
            "mask_01": str(output_dir / "fine_hair_mask_01.png"),
            "mask_255_compat": str(output_dir / "fine_hair_mask.png"),
            "alpha_16bit": str(output_dir / "fine_hair_alpha_16bit.png"),
            "score_16bit": str(output_dir / "fine_hair_score_16bit.png"),
            "bok_blending": str(output_dir / "fine_hair_bok_blend.png"),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[{sample.prefix}] 完成：ROI={x1 - x0}x{y1 - y0}，"
        f"细发丝像素={int(full_final.sum())}，输出={output_dir}"
    )
    return metadata


def infer_single_job(args: argparse.Namespace) -> tuple[Path, str]:
    """根据 --prefix、--input-dir 或旧式 --bok 推导单样本位置。"""

    if args.prefix:
        prefix_path = Path(args.prefix).expanduser()
        if prefix_path.parent != Path("."):
            if args.input_dir is not None:
                raise ValueError(
                    "--prefix 已包含目录时不能再同时指定 --input-dir"
                )
            input_dir = prefix_path.parent
            prefix = prefix_path.name
        else:
            input_dir = args.input_dir or (
                args.bok.parent if args.bok is not None else Path(".")
            )
            prefix = prefix_path.name
    elif args.bok is not None:
        input_dir = args.input_dir or args.bok.parent
        stem = args.bok.stem
        prefix = stem[:-4] if stem.casefold().endswith("_bok") else stem
    else:
        raise ValueError(
            "单图模式必须提供 --prefix，或使用旧式 --bok/--edof/--matte 参数"
        )
    return input_dir, prefix


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """检查互斥模式和数值参数。"""

    explicit_paths = (
        args.bok,
        args.edof,
        args.hair,
        args.hair_probability,
        args.sapiens2_labels,
        args.matte,
    )
    if args.batch and (args.prefix or any(path is not None for path in explicit_paths)):
        parser.error("--batch 不能与 --prefix 或显式输入路径混用")
    if args.batch and args.input_dir is None:
        parser.error("--batch 必须同时指定 --input-dir")
    radii = (
        args.outer_radius,
        args.inner_band,
        args.nonhair_radius,
        args.thin_radius,
        args.gap_close_radius,
    )
    if any(radius < 0 for radius in radii):
        parser.error("所有 radius 参数必须大于或等于 0")
    if args.roi_margin < -1:
        parser.error("--roi-margin 只能是 -1 或大于等于 0")
    if args.min_seed_area < 0 or args.min_area < 0 or args.min_extent < 0:
        parser.error("面积和尺度阈值必须大于或等于 0")
    if args.low_score > args.high_score:
        parser.error("--low-score 不能大于 --high-score")
    for name in (
        "hair_prob_threshold",
        "matte_seed_threshold",
        "roi_threshold",
        "low_score",
        "high_score",
        "component_score_min",
        "alpha_min",
        "gap_score_min",
        "gap_line_min",
        "gap_alpha_min",
        "gap_blur_max",
        "blend_opacity",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} 必须位于 0～1")


def build_parser() -> argparse.ArgumentParser:
    """构造中文命令行接口。"""

    parser = argparse.ArgumentParser(
        description=(
            "提取 BOK 中细碎、边缘、飘散发丝；支持 prefix 自动发现、"
            "目录批处理和 Hair mask 缺失时的 matte 后备。"
        )
    )
    parser.add_argument(
        "--prefix",
        help=(
            "单样本前缀，可写 2p（配合 --input-dir），"
            "也可直接写 D:\\base\\2p"
        ),
    )
    parser.add_argument("--input-dir", type=Path, help="自动发现输入文件的目录")
    parser.add_argument("--batch", action="store_true", help="批量处理目录内全部 *_bok 图像")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="批处理中首个样本失败时立即停止",
    )

    # 以下参数保留旧版显式路径调用方式，也可覆盖 prefix 自动发现的单个文件。
    parser.add_argument("--bok", type=Path, help="显式指定 BOK 图像")
    parser.add_argument("--edof", type=Path, help="显式指定同尺寸 EDOF 图像")
    parser.add_argument("--hair", type=Path, help="显式指定粗 Hair mask，可缺省")
    parser.add_argument(
        "--hair-probability",
        type=Path,
        help="显式指定 Sapiens2 Hair 概率图，可缺省",
    )
    parser.add_argument(
        "--sapiens2-labels",
        type=Path,
        help="显式指定 Sapiens2 全类别标签图，可缺省",
    )
    parser.add_argument("--matte", type=Path, help="显式指定 BiRefNet matte")
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        required=True,
        help="输出目录；批处理时每个 prefix 建立独立子目录",
    )

    parser.add_argument("--outer-radius", type=int, default=120, help="Hair 种子向外搜索半径")
    parser.add_argument("--inner-band", type=int, default=12, help="允许保留的 Hair 内侧边缘宽度")
    parser.add_argument("--nonhair-radius", type=int, default=3, help="非 Hair 人体类别排除半径")
    parser.add_argument("--thin-radius", type=int, default=8, help="删除宽边界结构的开运算半径")
    parser.add_argument(
        "--gap-close-radius",
        type=int,
        default=4,
        help="沿水平、垂直和两个对角方向连接的最大短缺口半径；0 表示关闭",
    )
    parser.add_argument(
        "--gap-score-min",
        type=float,
        default=0.16,
        help="允许短缺口桥接的最低综合分数",
    )
    parser.add_argument(
        "--gap-line-min",
        type=float,
        default=0.08,
        help="低综合分桥接像素所需的最低线状响应",
    )
    parser.add_argument(
        "--gap-alpha-min",
        type=float,
        default=0.01,
        help="低综合分桥接像素所需的最低 BiRefNet alpha",
    )
    parser.add_argument(
        "--gap-blur-max",
        type=float,
        default=0.35,
        help="允许桥接的最大 EDOF/BOK 虚化负证据",
    )
    parser.add_argument("--hair-prob-threshold", type=float, default=0.45, help="Hair 概率种子阈值")
    parser.add_argument("--matte-seed-threshold", type=float, default=0.50, help="Hair 全缺失时的 matte 种子阈值")
    parser.add_argument("--roi-threshold", type=float, default=0.05, help="生成 ROI 外接框的 mask 阈值")
    parser.add_argument("--roi-margin", type=int, default=-1, help="ROI 外扩下限；-1 表示自动覆盖所有滤波 halo")
    parser.add_argument("--no-roi", action="store_true", help="关闭裁剪，按完整分辨率处理")
    parser.add_argument("--low-score", type=float, default=0.12, help="双阈值连接的低阈值")
    parser.add_argument("--high-score", type=float, default=0.22, help="双阈值连接的高阈值")
    parser.add_argument("--component-score-min", type=float, default=0.38, help="连通域最低峰值置信度")
    parser.add_argument("--alpha-min", type=float, default=0.005, help="BiRefNet alpha 最低证据阈值")
    parser.add_argument("--min-seed-area", type=int, default=400, help="4K 参考尺度下的语义种子最小面积")
    parser.add_argument("--min-area", type=int, default=1, help="最终连通域最小面积")
    parser.add_argument("--min-extent", type=float, default=2.0, help="最终连通域外接框最小对角线")
    parser.add_argument("--blend-opacity", type=float, default=0.85, help="01 mask 红色叠加透明度")
    parser.add_argument(
        "--empty-policy",
        choices=("zero", "error"),
        default="zero",
        help="种子为空时输出全零结果或直接报错",
    )
    parser.add_argument("--no-debug", action="store_true", help="不保存数字编号调试图")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args, parser)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.batch:
        input_dir = args.input_dir.expanduser()
        if not input_dir.is_dir():
            parser.error(f"输入目录不存在：{input_dir}")
        prefixes = discover_prefixes(input_dir)
        if not prefixes:
            parser.error(f"目录内没有发现 *_bok 图像：{input_dir}")

        summary: dict[str, list[dict]] = {"success": [], "failed": []}
        for index, prefix in enumerate(prefixes, start=1):
            print(f"[批处理 {index}/{len(prefixes)}] {prefix}")
            try:
                sample = resolve_sample(input_dir, prefix, args, allow_explicit=False)
                metadata = process_one(sample, args.output / prefix, args)
                summary["success"].append(
                    {
                        "prefix": prefix,
                        "output": str(args.output / prefix),
                        "final_pixels": metadata["pixel_counts"]["final"],
                    }
                )
            except Exception as error:
                print(f"[{prefix}] 失败：{error}", file=sys.stderr)
                summary["failed"].append(
                    {
                        "prefix": prefix,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if args.fail_fast:
                    break

        (args.output / "batch_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"批处理结束：成功 {len(summary['success'])}，"
            f"失败 {len(summary['failed'])}"
        )
        if summary["failed"]:
            raise SystemExit(1)
        return

    try:
        input_dir, prefix = infer_single_job(args)
        sample = resolve_sample(input_dir, prefix, args, allow_explicit=True)
        process_one(sample, args.output, args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
