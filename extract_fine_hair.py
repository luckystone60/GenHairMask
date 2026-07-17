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

# 区域生长预设。高级参数仍可逐项覆盖，便于在不同数据集上做精度/召回权衡。
GROWTH_PRESETS = {
    "off": {
        "growth_radius": 0,
        "growth_radius_scale": 0.0,
        "growth_max_radius": 0,
        "growth_color_delta": 18.0,
        "growth_line_min": 0.055,
        "growth_score_min": 0.035,
        "growth_alpha_min": 0.001,
        "growth_blur_max": 0.38,
        "growth_coherence_min": 0.25,
        "growth_width_radius": 3,
        "growth_max_neighbors": 5,
        "growth_alpha_scale": 0.85,
    },
    "conservative": {
        "growth_radius": 18,
        "growth_radius_scale": 0.08,
        "growth_max_radius": 160,
        "growth_color_delta": 13.0,
        "growth_line_min": 0.075,
        "growth_score_min": 0.055,
        "growth_alpha_min": 0.003,
        "growth_blur_max": 0.35,
        "growth_coherence_min": 0.35,
        "growth_width_radius": 3,
        "growth_max_neighbors": 4,
        "growth_alpha_scale": 0.80,
    },
    "balanced": {
        "growth_radius": 32,
        "growth_radius_scale": 0.14,
        "growth_max_radius": 240,
        "growth_color_delta": 18.0,
        "growth_line_min": 0.055,
        "growth_score_min": 0.035,
        "growth_alpha_min": 0.001,
        "growth_blur_max": 0.38,
        "growth_coherence_min": 0.25,
        "growth_width_radius": 3,
        "growth_max_neighbors": 5,
        "growth_alpha_scale": 0.85,
    },
    "recall": {
        "growth_radius": 48,
        "growth_radius_scale": 0.22,
        "growth_max_radius": 360,
        "growth_color_delta": 23.0,
        "growth_line_min": 0.040,
        "growth_score_min": 0.020,
        "growth_alpha_min": 0.0005,
        "growth_blur_max": 0.42,
        "growth_coherence_min": 0.18,
        "growth_width_radius": 4,
        "growth_max_neighbors": 5,
        "growth_alpha_scale": 0.90,
    },
}

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


def line_coherence_map(bgr: np.ndarray, window_size: int = 7) -> np.ndarray:
    """用结构张量计算局部方向一致性，细长结构趋近 1，散乱纹理趋近 0。"""

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    kernel_size = (window_size, window_size)
    tensor_xx = cv2.boxFilter(
        gradient_x * gradient_x,
        cv2.CV_32F,
        kernel_size,
        normalize=True,
    )
    tensor_yy = cv2.boxFilter(
        gradient_y * gradient_y,
        cv2.CV_32F,
        kernel_size,
        normalize=True,
    )
    tensor_xy = cv2.boxFilter(
        gradient_x * gradient_y,
        cv2.CV_32F,
        kernel_size,
        normalize=True,
    )
    numerator = np.sqrt(
        np.maximum(
            (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy * tensor_xy,
            0.0,
        )
    )
    return np.clip(numerator / (tensor_xx + tensor_yy + 1e-6), 0.0, 1.0)


def nearest_seed_features(
    bgr: np.ndarray,
    seed: np.ndarray,
    seed_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算到最近确定发丝的距离、Lab 色差和可传播 alpha。"""

    if not seed.any():
        shape = seed.shape
        return (
            np.full(shape, np.inf, np.float32),
            np.full(shape, np.inf, np.float32),
            np.zeros(shape, np.float32),
        )

    lab_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = np.empty_like(lab_u8)
    lab[..., 0] = lab_u8[..., 0] * (100.0 / 255.0)
    lab[..., 1:] = lab_u8[..., 1:] - 128.0

    # DIST_LABEL_PIXEL 为每个确定发丝像素分配独立标签，可直接建立最近种子查找表。
    distance, nearest_labels = cv2.distanceTransformWithLabels(
        (~seed).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    maximum_label = int(nearest_labels.max(initial=0))
    color_lookup = np.zeros((maximum_label + 1, 3), np.float32)
    alpha_lookup = np.zeros(maximum_label + 1, np.float32)
    seed_labels = nearest_labels[seed]
    color_lookup[seed_labels] = lab[seed]
    alpha_lookup[seed_labels] = seed_alpha[seed]
    nearest_color = color_lookup[nearest_labels]
    color_delta = np.sqrt(np.sum((lab - nearest_color) ** 2, axis=2))
    nearest_alpha = alpha_lookup[nearest_labels]
    return distance, color_delta, nearest_alpha


def grow_thin_connected_region(
    seed: np.ndarray,
    allowed: np.ndarray,
    max_steps: int,
    width_radius: int,
    max_neighbors: int,
) -> np.ndarray:
    """从确定发丝逐像素生长，同时阻止横向变宽和大块区域灌入。"""

    if max_steps <= 0 or not seed.any() or not allowed.any():
        return seed.copy()

    grown = seed.copy()
    neighbor_kernel = np.ones((3, 3), np.uint8)
    for _ in range(max_steps):
        neighbor_count = cv2.filter2D(
            grown.astype(np.uint8),
            cv2.CV_16U,
            neighbor_kernel,
        )
        proposal = (
            cv2.dilate(grown.astype(np.uint8), neighbor_kernel).astype(bool)
            & allowed
            & ~grown
            & (neighbor_count <= max_neighbors)
        )
        if not proposal.any():
            break

        # 如果新区域能通过较大椭圆开运算，它更可能是块状轮廓而不是细长发丝。
        trial = grown | proposal
        broad = cv2.morphologyEx(
            trial.astype(np.uint8),
            cv2.MORPH_OPEN,
            ellipse(width_radius),
        ).astype(bool)
        proposal &= ~broad
        if not proposal.any():
            break
        grown |= proposal
    return grown


def prune_dense_growth(
    grown: np.ndarray,
    seed: np.ndarray,
    window_size: int,
    maximum_density: float,
) -> tuple[np.ndarray, np.ndarray]:
    """删除局部过密的网状新增区域，只保留仍与原始发丝连通的细长分支。"""

    added = grown & ~seed
    if not added.any() or window_size <= 1:
        return grown, np.zeros_like(seed)
    density = cv2.boxFilter(
        added.astype(np.float32),
        cv2.CV_32F,
        (window_size, window_size),
        normalize=True,
    )
    dense = added & (density > maximum_density)
    sparse = added & ~dense
    connected = keep_seed_connected(sparse, seed)
    final = seed | connected
    return final, grown & ~final


def grow_fine_hair_region(
    bok: np.ndarray,
    edof: np.ndarray,
    seed: np.ndarray,
    core_search: np.ndarray,
    growth_search: np.ndarray,
    person_core: np.ndarray,
    coarse_exclusion: np.ndarray,
    nonhair_exclusion: np.ndarray,
    bok_line: np.ndarray,
    edof_line: np.ndarray,
    blur_loss: np.ndarray,
    score: np.ndarray,
    alpha: np.ndarray,
    semantic: np.ndarray,
    effective_growth_radius: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """按颜色、方向一致性和细长几何约束扩展已确认发丝。"""

    zero_bool = np.zeros_like(seed)
    zero_float = np.zeros(seed.shape, np.float32)
    if effective_growth_radius <= 0 or not seed.any():
        return {
            "final": seed.copy(),
            "added": zero_bool,
            "allowed": zero_bool,
            "extension_allowed": zero_bool,
            "color_similarity": zero_float,
            "coherence": zero_float,
            "edof_rescue": zero_bool,
            "fused_line": zero_float,
            "dense_removed": zero_bool,
            "confidence": zero_float,
            "estimated_alpha": zero_float,
        }

    distance, color_delta, nearest_alpha = nearest_seed_features(bok, seed, alpha)
    bok_coherence = line_coherence_map(bok)
    edof_coherence = cv2.dilate(
        line_coherence_map(edof),
        ellipse(args.edof_tolerance_radius),
    )
    if args.growth_image == "bok":
        line = bok_line
        coherence = bok_coherence
    elif args.growth_image == "edof":
        line = edof_line
        coherence = edof_coherence
    else:
        shared_line = np.sqrt(np.clip(bok_line * edof_line, 0.0, 1.0))
        shared_coherence = np.sqrt(
            np.clip(bok_coherence * edof_coherence, 0.0, 1.0)
        )
        line = np.maximum(bok_line, args.edof_line_weight * shared_line)
        coherence = np.maximum(
            bok_coherence,
            args.edof_line_weight * shared_coherence,
        )
    color_similarity = np.clip(
        1.0 - color_delta / max(args.growth_color_delta, 1e-6),
        0.0,
        1.0,
    )
    # EDOF 中强且连续、BOK 中仍有微弱线索的细线可在 alpha/语义漏检时
    # 提供救援证据；颜色仍锚定 BOK，避免把两图几何差异硬写入最终 mask。
    strong_shared_geometry = (
        (bok_line >= max(args.edof_bok_support_min, 0.040))
        & (edof_line >= max(args.growth_line_min * 2.0, 0.12))
        & (bok_coherence >= 0.35)
        & (edof_coherence >= 0.45)
        & (color_delta <= args.growth_color_delta * 0.38)
    )
    edof_rescue = (
        (bok_line >= max(args.edof_bok_support_min, 0.040))
        & (edof_line >= max(args.growth_line_min * 1.8, 0.10))
        & (bok_coherence >= 0.20)
        & (edof_coherence >= max(args.growth_coherence_min + 0.15, 0.42))
        & (color_delta <= args.growth_color_delta * 0.48)
        & (
            (alpha >= max(args.growth_alpha_min, 0.001))
            | (semantic >= 0.04)
            | (distance <= 12.0)
            | strong_shared_geometry
        )
    )
    if args.growth_image == "bok":
        edof_rescue = zero_bool
    effective_blur_loss = blur_loss * (
        1.0 - args.edof_rescue_strength * edof_rescue.astype(np.float32)
    )

    # 常规区域仍遵守 Sapiens2 人体类别排除；只有紧邻发丝且证据很强时，
    # 才允许短距离跨过 Face/Apparel 等语义区域，以恢复贴脸或压在衣服上的发丝。
    hard_region = person_core | nonhair_exclusion
    hard_override = (
        (distance <= min(float(effective_growth_radius), 8.0))
        & (color_delta <= min(args.growth_color_delta, 12.0))
        & (line >= max(args.growth_line_min, 0.12))
        & (coherence >= max(args.growth_coherence_min, 0.35))
        & (alpha >= max(args.growth_alpha_min, 0.02))
    )

    strong_geometry = (
        (line >= max(args.growth_line_min * 1.8, 0.10))
        & (coherence >= max(args.growth_coherence_min, 0.30))
        & (color_delta <= args.growth_color_delta * 0.65)
    )
    evidence = (
        (line >= args.growth_line_min)
        & (coherence >= args.growth_coherence_min)
        & (effective_blur_loss <= args.growth_blur_max)
        & (
            (score >= args.growth_score_min)
            | (alpha >= args.growth_alpha_min)
            | (semantic >= 0.08)
            | strong_geometry
            | edof_rescue
        )
    )
    common_allowed = (
        growth_search
        & ~coarse_exclusion
        & (distance <= float(effective_growth_radius))
        & (color_delta <= args.growth_color_delta)
        & evidence
        & (~hard_region | hard_override)
    )

    # 超出常规搜索区后提高门槛，只允许颜色更接近、方向更稳定且仍有
    # alpha/Hair 证据的连续细线进入远距离延伸走廊。
    extension_evidence = (
        ~core_search
        & (color_delta <= args.growth_color_delta * 0.72)
        & (line >= max(args.growth_line_min * 1.25, 0.065))
        & (coherence >= min(args.growth_coherence_min + 0.10, 0.95))
        & (effective_blur_loss <= args.growth_blur_max * 0.90)
        & (
            (alpha >= max(args.growth_alpha_min * 2.0, 0.002))
            | (semantic >= 0.12)
            | edof_rescue
        )
        & ~hard_region
    )
    extension_allowed = common_allowed & extension_evidence
    allowed = common_allowed & (core_search | extension_evidence)

    grown = grow_thin_connected_region(
        seed,
        allowed,
        effective_growth_radius,
        args.growth_width_radius,
        args.growth_max_neighbors,
    )
    if args.growth_image == "dual":
        # 双图模式先完整保留 BOK 单图生长结果，只对 EDOF 额外贡献执行
        # 密度抑制，避免为了清理背景网纹反而删掉原本可靠的 BOK 发丝。
        bok_hard_override = (
            (distance <= min(float(effective_growth_radius), 8.0))
            & (color_delta <= min(args.growth_color_delta, 12.0))
            & (bok_line >= max(args.growth_line_min, 0.12))
            & (bok_coherence >= max(args.growth_coherence_min, 0.35))
            & (alpha >= max(args.growth_alpha_min, 0.02))
        )
        bok_strong_geometry = (
            (bok_line >= max(args.growth_line_min * 1.8, 0.10))
            & (bok_coherence >= max(args.growth_coherence_min, 0.30))
            & (color_delta <= args.growth_color_delta * 0.65)
        )
        bok_evidence = (
            (bok_line >= args.growth_line_min)
            & (bok_coherence >= args.growth_coherence_min)
            & (blur_loss <= args.growth_blur_max)
            & (
                (score >= args.growth_score_min)
                | (alpha >= args.growth_alpha_min)
                | (semantic >= 0.08)
                | bok_strong_geometry
            )
        )
        bok_common_allowed = (
            growth_search
            & ~coarse_exclusion
            & (distance <= float(effective_growth_radius))
            & (color_delta <= args.growth_color_delta)
            & bok_evidence
            & (~hard_region | bok_hard_override)
        )
        bok_extension_evidence = (
            ~core_search
            & (color_delta <= args.growth_color_delta * 0.72)
            & (bok_line >= max(args.growth_line_min * 1.25, 0.065))
            & (bok_coherence >= min(args.growth_coherence_min + 0.10, 0.95))
            & (blur_loss <= args.growth_blur_max * 0.90)
            & (
                (alpha >= max(args.growth_alpha_min * 2.0, 0.002))
                | (semantic >= 0.12)
            )
            & ~hard_region
        )
        bok_allowed = bok_common_allowed & (
            core_search | bok_extension_evidence
        )
        bok_grown = grow_thin_connected_region(
            seed,
            bok_allowed,
            effective_growth_radius,
            args.growth_width_radius,
            args.growth_max_neighbors,
        )
        grown, dense_removed = prune_dense_growth(
            grown | bok_grown,
            bok_grown,
            args.edof_density_window,
            args.edof_density_max,
        )
    elif args.growth_image == "edof":
        grown, dense_removed = prune_dense_growth(
            grown,
            seed,
            args.edof_density_window,
            args.edof_density_max,
        )
    else:
        dense_removed = zero_bool
    added = grown & ~seed

    line_confidence = np.clip(
        (line - args.growth_line_min) / max(0.25 - args.growth_line_min, 1e-6),
        0.0,
        1.0,
    )
    coherence_confidence = np.clip(
        (coherence - args.growth_coherence_min)
        / max(1.0 - args.growth_coherence_min, 1e-6),
        0.0,
        1.0,
    )
    distance_confidence = np.clip(
        1.0 - distance / max(float(effective_growth_radius), 1.0),
        0.0,
        1.0,
    )
    confidence = (
        0.40 * color_similarity
        + 0.25 * line_confidence
        + 0.20 * coherence_confidence
        + 0.15 * distance_confidence
    ) * added.astype(np.float32)
    estimated_alpha = np.clip(
        nearest_alpha
        * (0.35 + 0.65 * confidence)
        * args.growth_alpha_scale,
        0.0,
        1.0,
    ) * added.astype(np.float32)
    return {
        "final": grown,
        "added": added,
        "allowed": allowed,
        "extension_allowed": extension_allowed,
        "color_similarity": color_similarity,
        "coherence": coherence,
        "edof_rescue": edof_rescue,
        "fused_line": line,
        "dense_removed": dense_removed,
        "confidence": confidence,
        "estimated_alpha": estimated_alpha,
    }


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


def estimate_hair_extent(seed: np.ndarray) -> float:
    """估计主要 Hair 连通域尺度，避免人物间距或零散误检放大搜索半径。"""

    if not seed.any():
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        seed.astype(np.uint8),
        connectivity=8,
    )
    minimum_area = max(16, int(round(seed.size / REFERENCE_AREA * 100.0)))
    extents = [
        float(max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]))
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= minimum_area
    ]
    return max(extents, default=0.0)


def effective_search_radii(
    hair_seed: np.ndarray,
    args: argparse.Namespace,
    use_hair_scale: bool,
) -> tuple[int, int, float, float]:
    """按分辨率与主要 Hair 尺度计算常规搜索半径和远距离生长半径。"""

    resolution_scale = float(np.sqrt(hair_seed.size / REFERENCE_AREA))
    if args.search_mode == "fixed":
        return (
            args.outer_radius,
            args.growth_radius,
            estimate_hair_extent(hair_seed) if use_hair_scale else 0.0,
            resolution_scale,
        )

    hair_extent = estimate_hair_extent(hair_seed) if use_hair_scale else 0.0
    outer_minimum = int(round(args.outer_radius * resolution_scale))
    outer_from_hair = int(round(hair_extent * args.search_radius_scale))
    outer_cap = (
        args.search_max_radius
        if args.search_max_radius > 0
        else int(round(320 * resolution_scale))
    )
    effective_outer = min(max(outer_minimum, outer_from_hair), max(outer_cap, 0))

    if args.growth_radius <= 0:
        effective_growth = 0
    else:
        growth_minimum = int(round(args.growth_radius * resolution_scale))
        growth_from_hair = int(round(hair_extent * args.growth_radius_scale))
        growth_cap = (
            int(round(args.growth_max_radius * resolution_scale))
            if args.growth_max_radius > 0
            else 0
        )
        effective_growth = min(
            max(growth_minimum, growth_from_hair),
            max(growth_cap, 0),
        )
    return effective_outer, effective_growth, hair_extent, resolution_scale


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
    effective_outer_radius: int,
    effective_growth_radius: int,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """在已经裁剪的 ROI 内执行细碎边缘发丝检测。"""

    distance_to_support = cv2.distanceTransform(
        (~hair_seed).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    search = distance_to_support <= float(effective_outer_radius)
    growth_search = distance_to_support <= float(
        effective_outer_radius + effective_growth_radius
    )

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
    # 自适应模式额外计算延伸区响应；固定模式沿用旧行为，便于严格回归。
    growth_line = (
        median_line_response(bok, growth_search)
        if args.search_mode == "adaptive" and effective_growth_radius > 0
        else line
    )
    edof_growth_line = cv2.dilate(
        median_line_response(edof, growth_search),
        ellipse(args.edof_tolerance_radius),
    )
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

    pre_growth = final.copy()
    growth = grow_fine_hair_region(
        bok,
        edof,
        pre_growth,
        search,
        growth_search,
        person_core,
        coarse_exclusion,
        nonhair_exclusion,
        growth_line,
        edof_growth_line,
        blur_loss,
        score,
        alpha,
        semantic,
        effective_growth_radius,
        args,
    )
    final = growth["final"]

    # 已确认区域继续使用 BiRefNet 覆盖率；新增区域使用最近发丝 alpha 与
    # 生长置信度传播，避免直接套用脸/衣服上的整个人像 alpha。
    fine_alpha = alpha * pre_growth.astype(np.float32)
    fine_alpha[growth["added"]] = growth["estimated_alpha"][growth["added"]]
    final_score = np.maximum(score, growth["confidence"])
    return {
        "search": search,
        "growth_search": growth_search,
        "person_core": person_core,
        "coarse_exclusion": coarse_exclusion,
        "nonhair_exclusion": nonhair_exclusion,
        "fringe_zone": fringe_zone,
        "line": line,
        "growth_line": growth_line,
        "edof_growth_line": edof_growth_line,
        "fused_growth_line": growth["fused_line"],
        "blur_loss": blur_loss,
        "alpha_edge": alpha_edge,
        "score": score,
        "weak": weak,
        "strong": strong,
        "connected": hysteresis,
        "before_thin": before_thin,
        "broad": broad,
        "gap_bridge": gap_bridge,
        "pre_growth": pre_growth,
        "growth_allowed": growth["allowed"],
        "growth_extension_allowed": growth["extension_allowed"],
        "growth_extension_added": growth["added"] & ~search,
        "growth_color_similarity": growth["color_similarity"],
        "growth_coherence": growth["coherence"],
        "edof_rescue": growth["edof_rescue"],
        "edof_dense_removed": growth["dense_removed"],
        "growth_added": growth["added"],
        "growth_confidence": growth["confidence"],
        "final": final,
        "fine_alpha": fine_alpha,
        "final_score": final_score,
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
    save_binary_visual(debug_dir / "17_pre_growth_mask.png", stages["pre_growth"])
    save_u8(
        debug_dir / "18_growth_color_similarity.png",
        stages["growth_color_similarity"],
    )
    save_u8(
        debug_dir / "19_growth_line_coherence.png",
        stages["growth_coherence"],
    )
    save_binary_visual(
        debug_dir / "20_growth_allowed.png",
        stages["growth_allowed"],
    )
    save_binary_visual(
        debug_dir / "21_region_growth_added.png",
        stages["growth_added"],
    )
    save_u16(
        debug_dir / "22_growth_confidence_16bit.png",
        stages["growth_confidence"],
    )
    save_binary_visual(debug_dir / "23_final_mask_visual.png", stages["final"])
    save_binary_visual(
        debug_dir / "24_growth_search_region.png",
        stages["growth_search"],
    )
    save_binary_visual(
        debug_dir / "25_growth_extension_added.png",
        stages["growth_extension_added"],
    )
    save_binary_visual(
        debug_dir / "26_growth_extension_allowed.png",
        stages["growth_extension_allowed"],
    )
    save_u8(debug_dir / "27_growth_line_response.png", stages["growth_line"])
    save_u8(
        debug_dir / "28_edof_growth_line_response.png",
        stages["edof_growth_line"],
    )
    save_u8(
        debug_dir / "29_fused_growth_line_response.png",
        stages["fused_growth_line"],
    )
    save_binary_visual(
        debug_dir / "30_edof_rescue.png",
        stages["edof_rescue"],
    )
    save_binary_visual(
        debug_dir / "31_edof_dense_removed.png",
        stages["edof_dense_removed"],
    )


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

    use_hair_scale = not semantic_seed_source.startswith("matte_fallback")
    (
        effective_outer_radius,
        effective_growth_radius,
        estimated_hair_extent,
        resolution_scale,
    ) = effective_search_radii(hair_seed_full, args, use_hair_scale)
    print(
        f"[{sample.prefix}] 搜索半径：常规={effective_outer_radius}px，"
        f"远距离延伸={effective_growth_radius}px，模式={args.search_mode}"
    )

    filter_halo = max(
        24,
        max(MEDIAN_KERNELS) // 2 + 6,
        18 + args.nonhair_radius,
        args.inner_band + args.thin_radius + 4,
        args.gap_close_radius * 2 + 4,
        args.growth_width_radius + 8,
    )
    safe_roi_margin = (
        effective_outer_radius + effective_growth_radius + filter_halo
    )
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
            effective_outer_radius,
            effective_growth_radius,
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
            "growth_search": zero_bool,
            "person_core": zero_bool,
            "coarse_exclusion": zero_bool,
            "nonhair_exclusion": zero_bool,
            "fringe_zone": zero_bool,
            "line": zero_float,
            "growth_line": zero_float,
            "edof_growth_line": zero_float,
            "fused_growth_line": zero_float,
            "blur_loss": zero_float,
            "alpha_edge": zero_float,
            "score": zero_float,
            "weak": zero_bool,
            "strong": zero_bool,
            "connected": zero_bool,
            "before_thin": zero_bool,
            "broad": zero_bool,
            "gap_bridge": zero_bool,
            "pre_growth": zero_bool,
            "growth_allowed": zero_bool,
            "growth_extension_allowed": zero_bool,
            "growth_extension_added": zero_bool,
            "growth_color_similarity": zero_float,
            "growth_coherence": zero_float,
            "edof_rescue": zero_bool,
            "edof_dense_removed": zero_bool,
            "growth_added": zero_bool,
            "growth_confidence": zero_float,
            "final": zero_bool,
            "fine_alpha": zero_float,
            "final_score": zero_float,
        }
        empty_reason = "有效种子为空，按 empty-policy=zero 输出空结果"
        print(f"[{sample.prefix}] 警告：{empty_reason}")

    full_final = np.zeros((height, width), bool)
    full_alpha = np.zeros((height, width), np.float32)
    full_score = np.zeros((height, width), np.float32)
    full_final[roi] = stages["final"]
    full_alpha[roi] = stages["fine_alpha"]
    full_score[roi] = stages["final_score"]

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
            "受证据约束的四方向短缺口连接 + "
            "自适应双层搜索区 + BOK 颜色锚定 + "
            "EDOF 容差细线增强与过密纹理抑制的远距离区域生长"
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
            "search_mode": args.search_mode,
            "search_radius_scale": args.search_radius_scale,
            "search_max_radius": args.search_max_radius,
            "effective_outer_radius": effective_outer_radius,
            "estimated_hair_extent": estimated_hair_extent,
            "resolution_scale": resolution_scale,
            "inner_band": args.inner_band,
            "nonhair_radius": args.nonhair_radius,
            "thin_radius": args.thin_radius,
            "gap_close_radius": args.gap_close_radius,
            "gap_score_min": args.gap_score_min,
            "gap_line_min": args.gap_line_min,
            "gap_alpha_min": args.gap_alpha_min,
            "gap_blur_max": args.gap_blur_max,
            "growth_preset": args.growth_preset,
            "growth_image": args.growth_image,
            "edof_line_weight": args.edof_line_weight,
            "edof_tolerance_radius": args.edof_tolerance_radius,
            "edof_bok_support_min": args.edof_bok_support_min,
            "edof_rescue_strength": args.edof_rescue_strength,
            "edof_density_window": args.edof_density_window,
            "edof_density_max": args.edof_density_max,
            "growth_radius": args.growth_radius,
            "growth_radius_scale": args.growth_radius_scale,
            "growth_max_radius": args.growth_max_radius,
            "effective_growth_radius": effective_growth_radius,
            "growth_color_delta": args.growth_color_delta,
            "growth_line_min": args.growth_line_min,
            "growth_score_min": args.growth_score_min,
            "growth_alpha_min": args.growth_alpha_min,
            "growth_blur_max": args.growth_blur_max,
            "growth_coherence_min": args.growth_coherence_min,
            "growth_width_radius": args.growth_width_radius,
            "growth_max_neighbors": args.growth_max_neighbors,
            "growth_alpha_scale": args.growth_alpha_scale,
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
            "pre_growth": int(stages["pre_growth"].sum()),
            "region_growth_added": int(stages["growth_added"].sum()),
            "growth_extension_added": int(
                stages["growth_extension_added"].sum()
            ),
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
        args.search_max_radius,
        args.inner_band,
        args.nonhair_radius,
        args.thin_radius,
        args.gap_close_radius,
        args.growth_radius,
        args.growth_max_radius,
        args.growth_width_radius,
        args.edof_tolerance_radius,
    )
    if any(radius < 0 for radius in radii):
        parser.error("所有 radius 参数必须大于或等于 0")
    if args.roi_margin < -1:
        parser.error("--roi-margin 只能是 -1 或大于等于 0")
    if args.min_seed_area < 0 or args.min_area < 0 or args.min_extent < 0:
        parser.error("面积和尺度阈值必须大于或等于 0")
    if args.low_score > args.high_score:
        parser.error("--low-score 不能大于 --high-score")
    if args.growth_color_delta <= 0.0:
        parser.error("--growth-color-delta 必须大于 0")
    if args.search_radius_scale < 0.0 or args.growth_radius_scale < 0.0:
        parser.error("搜索和生长的 radius-scale 必须大于或等于 0")
    if not 1 <= args.growth_max_neighbors <= 8:
        parser.error("--growth-max-neighbors 必须位于 1～8")
    if args.edof_density_window <= 1 or args.edof_density_window % 2 == 0:
        parser.error("--edof-density-window 必须是大于 1 的奇数")
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
        "growth_line_min",
        "growth_score_min",
        "growth_alpha_min",
        "growth_blur_max",
        "growth_coherence_min",
        "growth_alpha_scale",
        "edof_line_weight",
        "edof_bok_support_min",
        "edof_rescue_strength",
        "edof_density_max",
        "blend_opacity",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} 必须位于 0～1")


def apply_growth_preset(args: argparse.Namespace) -> None:
    """应用区域生长预设；命令行显式给出的高级参数拥有更高优先级。"""

    preset = GROWTH_PRESETS[args.growth_preset]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


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

    parser.add_argument(
        "--outer-radius",
        type=int,
        default=120,
        help="4K 参考尺度下常规 Hair 搜索半径；adaptive 模式把它作为下限",
    )
    parser.add_argument(
        "--search-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="按分辨率/Hair 尺度自适应搜索（默认），或保持固定像素半径",
    )
    parser.add_argument(
        "--search-radius-scale",
        type=float,
        default=0.18,
        help="常规搜索半径相对主要 Hair 连通域最长边的比例",
    )
    parser.add_argument(
        "--search-max-radius",
        type=int,
        default=0,
        help="自适应常规搜索半径上限；0 表示按分辨率自动计算",
    )
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
    parser.add_argument(
        "--growth-preset",
        choices=tuple(GROWTH_PRESETS),
        default="balanced",
        help=(
            "区域生长档位：off 关闭；conservative 保守；"
            "balanced 平衡（默认）；recall 高召回"
        ),
    )
    parser.add_argument(
        "--growth-image",
        choices=("bok", "dual", "edof"),
        default="dual",
        help="区域生长细线证据来源；默认融合 BOK 与容差对齐后的 EDOF",
    )
    parser.add_argument(
        "--edof-line-weight",
        type=float,
        default=0.85,
        help="dual 模式下 EDOF 细线与方向证据权重",
    )
    parser.add_argument(
        "--edof-tolerance-radius",
        type=int,
        default=3,
        help="允许 EDOF/BOK 发丝几何偏移的局部最大值半径",
    )
    parser.add_argument(
        "--edof-bok-support-min",
        type=float,
        default=0.040,
        help="仅靠 EDOF 救援候选时，BOK 仍需保留的最低细线响应",
    )
    parser.add_argument(
        "--edof-rescue-strength",
        type=float,
        default=0.35,
        help="EDOF 强发丝证据对虚化负证据的最大抵消比例",
    )
    parser.add_argument(
        "--edof-density-window",
        type=int,
        default=15,
        help="抑制 EDOF 网状纹理误检的局部密度窗口，必须为大于 1 的奇数",
    )
    parser.add_argument(
        "--edof-density-max",
        type=float,
        default=0.18,
        help="EDOF 辅助生长允许的最大局部新增像素密度",
    )
    parser.add_argument(
        "--growth-radius",
        type=int,
        default=None,
        help="从已确认细发丝向外生长的最大像素距离",
    )
    parser.add_argument(
        "--growth-radius-scale",
        type=float,
        default=None,
        help="远距离延伸半径相对主要 Hair 连通域最长边的比例",
    )
    parser.add_argument(
        "--growth-max-radius",
        type=int,
        default=None,
        help="4K 参考尺度下远距离延伸半径上限",
    )
    parser.add_argument(
        "--growth-color-delta",
        type=float,
        default=None,
        help="生长像素与最近原始发丝种子的最大 Lab 色差",
    )
    parser.add_argument(
        "--growth-line-min",
        type=float,
        default=None,
        help="区域生长所需的最小细线响应",
    )
    parser.add_argument(
        "--growth-score-min",
        type=float,
        default=None,
        help="区域生长所需的最小原始综合分数",
    )
    parser.add_argument(
        "--growth-alpha-min",
        type=float,
        default=None,
        help="区域生长所需的最小 BiRefNet alpha 证据",
    )
    parser.add_argument(
        "--growth-blur-max",
        type=float,
        default=None,
        help="区域生长允许的最大 EDOF/BOK 虚化负证据",
    )
    parser.add_argument(
        "--growth-coherence-min",
        type=float,
        default=None,
        help="区域生长所需的最小局部方向一致性",
    )
    parser.add_argument(
        "--growth-width-radius",
        type=int,
        default=None,
        help="抑制生长结果变成宽块的形态学检测半径",
    )
    parser.add_argument(
        "--growth-max-neighbors",
        type=int,
        default=None,
        help="每轮生长允许的最大八邻域已生长像素数",
    )
    parser.add_argument(
        "--growth-alpha-scale",
        type=float,
        default=None,
        help="新增区域继承最近发丝 alpha 时的缩放系数",
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
    apply_growth_preset(args)
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
