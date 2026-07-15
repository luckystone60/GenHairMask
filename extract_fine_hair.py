from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_float_mask(path: Path) -> np.ndarray:
    src = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(path)
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if src.dtype == np.uint16:
        return src.astype(np.float32) / 65535.0
    if src.dtype == np.uint8:
        return src.astype(np.float32) / 255.0
    src = src.astype(np.float32)
    return np.clip(src / max(float(src.max()), 1.0), 0, 1)


def ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def robust_normalize(x: np.ndarray, mask: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    values = x[mask & (x > 0)]
    scale = float(np.percentile(values, percentile)) if values.size else 1.0
    return np.clip(x / max(scale, 1e-6), 0, 1)


def median_line_response(bgr: np.ndarray, search: np.ndarray, kernels=(7, 11, 17, 25, 35)) -> np.ndarray:
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    src = lum.astype(np.float32)
    response = np.zeros(src.shape, np.float32)
    for k in kernels:
        med = cv2.medianBlur(lum, k).astype(np.float32)
        response = np.maximum(response, np.abs(src - med))
    return robust_normalize(response, search, 99.25)


def blur_loss_map(bok: np.ndarray, edof: np.ndarray) -> np.ndarray:
    def gradient(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.magnitude(gx, gy)
        return cv2.dilate(g, ellipse(4))

    gb, ge = gradient(bok), gradient(edof)
    loss = np.clip((ge - 1.12 * gb) / (ge + 5.0), 0, 1)
    return loss * np.clip((ge - 5.0) / 25.0, 0, 1)


def keep_seed_connected(weak: np.ndarray, seed: np.ndarray) -> np.ndarray:
    joined = (weak | seed).astype(np.uint8)
    count, labels = cv2.connectedComponents(joined, connectivity=8)
    if count <= 1:
        return np.zeros_like(weak)
    ids = np.unique(labels[seed])
    ids = ids[ids != 0]
    return np.isin(labels, ids) & weak


def filter_components(mask: np.ndarray, score: np.ndarray, min_area: int, min_extent: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(mask.shape, bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        extent = float(np.hypot(width, height))
        pixels = labels == label
        if area >= min_area and extent >= min_extent and float(score[pixels].max()) >= 0.38:
            keep[pixels] = True
    return keep


def save_u8(path: Path, x: np.ndarray) -> None:
    cv2.imwrite(str(path), np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8))


def save_u16(path: Path, x: np.ndarray) -> None:
    cv2.imwrite(str(path), np.rint(np.clip(x, 0, 1) * 65535).astype(np.uint16))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract only thin/wispy boundary hair at full BOK resolution.")
    ap.add_argument("--bok", type=Path, required=True)
    ap.add_argument("--edof", type=Path, required=True)
    ap.add_argument("--hair", type=Path, required=True)
    ap.add_argument("--hair-probability", type=Path)
    ap.add_argument("--sapiens2-labels", type=Path)
    ap.add_argument("--matte", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--outer-radius", type=int, default=120)
    ap.add_argument("--inner-band", type=int, default=12, help="Pixels retained inside the coarse Hair boundary.")
    ap.add_argument("--nonhair-radius", type=int, default=3, help="Exclusion margin around non-Hair Sapiens2 classes.")
    ap.add_argument("--thin-radius", type=int, default=6, help="Opening radius used to remove broad boundary bands.")
    ap.add_argument("--hair-prob-threshold", type=float, default=0.45)
    ap.add_argument("--low-score", type=float, default=0.12)
    ap.add_argument("--high-score", type=float, default=0.22)
    ap.add_argument("--alpha-min", type=float, default=0.005)
    ap.add_argument("--min-area", type=int, default=1)
    ap.add_argument("--min-extent", type=float, default=2.0)
    args = ap.parse_args()

    bok = cv2.imread(str(args.bok), cv2.IMREAD_COLOR)
    edof = cv2.imread(str(args.edof), cv2.IMREAD_COLOR)
    if bok is None or edof is None:
        raise FileNotFoundError("BOK or EDOF could not be read")
    alpha = read_float_mask(args.matte)
    coarse = read_float_mask(args.hair)
    probability = read_float_mask(args.hair_probability) if args.hair_probability else coarse
    labels = cv2.imread(str(args.sapiens2_labels), cv2.IMREAD_GRAYSCALE) if args.sapiens2_labels else None
    h, w = alpha.shape
    checked = (bok, edof, coarse, probability) + ((labels,) if labels is not None else ())
    if any(x.shape[:2] != (h, w) for x in checked):
        raise ValueError("All inputs must already share the BOK resolution")

    # Sapiens2 is deliberately used as a soft semantic prior. A low-probability
    # support catches pale/transparent strands, while the 0.45 mask remains the seed.
    hair_seed = ((labels == 4) if labels is not None else (probability >= args.hair_prob_threshold)) | (coarse >= 0.5)
    # Remove tiny isolated semantic mistakes before growing the hair search region.
    n, cc, stats, _ = cv2.connectedComponentsWithStats(hair_seed.astype(np.uint8), 8)
    clean_seed = np.zeros_like(hair_seed)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= 400:
            clean_seed[cc == idx] = True
    hair_seed = clean_seed
    distance_to_support = cv2.distanceTransform((~hair_seed).astype(np.uint8), cv2.DIST_L2, 5)
    search = distance_to_support <= float(args.outer_radius)

    # We only want fringe material, never the large opaque interior of a person or hair mass.
    person_core = cv2.erode((alpha >= 0.72).astype(np.uint8), ellipse(10)).astype(bool)
    coarse_exclusion = cv2.erode(hair_seed.astype(np.uint8), ellipse(args.inner_band)).astype(bool)
    if labels is not None:
        nonhair_person = (labels != 0) & (labels != 4)
        nonhair_exclusion = cv2.dilate(nonhair_person.astype(np.uint8), ellipse(args.nonhair_radius)).astype(bool)
    else:
        nonhair_exclusion = np.zeros_like(search)
    fringe_zone = search & ~person_core & ~coarse_exclusion & ~nonhair_exclusion

    line = median_line_response(bok, search)
    blur_loss = blur_loss_map(bok, edof)
    semantic = cv2.dilate(probability, ellipse(18))
    alpha_edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, ellipse(2))
    alpha_edge = robust_normalize(alpha_edge, search, 99.0)

    score = (
        0.47 * np.sqrt(np.clip(alpha, 0, 1))
        + 0.27 * line
        + 0.12 * np.sqrt(np.clip(semantic, 0, 1))
        + 0.14 * alpha_edge
        - 0.24 * blur_loss
    )
    score = np.clip(score, 0, 1) * fringe_zone
    evidence = (alpha >= args.alpha_min) & ((line >= 0.055) | (alpha_edge >= 0.10) | (alpha >= 0.22))
    weak = fringe_zone & evidence & (score >= args.low_score)
    strong = fringe_zone & evidence & ((score >= args.high_score) | ((alpha >= 0.45) & (line >= 0.08)))

    # Permit a one-pixel bridge to the semantic hair seed, then remove the seed itself.
    weak_bridge = cv2.dilate(weak.astype(np.uint8), ellipse(1)).astype(bool)
    connected = keep_seed_connected(weak_bridge, cv2.dilate(hair_seed.astype(np.uint8), ellipse(3)).astype(bool))
    connected &= weak
    hysteresis = keep_seed_connected(connected, strong)
    before_thin = filter_components(hysteresis, score, args.min_area, args.min_extent)
    if args.thin_radius > 0:
        broad = cv2.morphologyEx(before_thin.astype(np.uint8), cv2.MORPH_OPEN, ellipse(args.thin_radius)).astype(bool)
        final = before_thin & ~broad
        final = filter_components(final, score, args.min_area, args.min_extent)
    else:
        final = before_thin
    # Preserve BiRefNet's calibrated soft coverage; score is confidence, not opacity.
    fine_alpha = alpha * final.astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    save_u8(args.output / "fine_hair_mask.png", final.astype(np.float32))
    save_u16(args.output / "fine_hair_alpha_16bit.png", fine_alpha)
    save_u16(args.output / "fine_hair_score_16bit.png", score)
    save_u8(args.output / "debug_search.png", search.astype(np.float32))
    save_u8(args.output / "debug_fringe_zone.png", fringe_zone.astype(np.float32))
    save_u8(args.output / "debug_nonhair_exclusion.png", nonhair_exclusion.astype(np.float32))
    save_u8(args.output / "debug_line_response.png", line)
    save_u8(args.output / "debug_blur_loss.png", blur_loss)
    save_u8(args.output / "debug_candidate.png", weak.astype(np.float32))
    save_u8(args.output / "debug_before_thin_filter.png", before_thin.astype(np.float32))

    rgb = cv2.cvtColor(bok, cv2.COLOR_BGR2RGB).astype(np.float32)
    tint = np.array([255, 32, 32], np.float32)
    a = (final.astype(np.float32) * 0.85)[..., None]
    preview = cv2.cvtColor(np.clip(rgb * (1 - a) + tint * a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.output / "fine_hair_overlay.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])

    meta = {
        "resolution_wh": [w, h],
        "algorithm": "BiRefNet alpha fringe + Sapiens2 soft hair support + multiscale median line response + EDOF blur rejection",
        "parameters": vars(args) | {"bok": str(args.bok), "edof": str(args.edof), "hair": str(args.hair), "matte": str(args.matte), "hair_probability": str(args.hair_probability)},
        "pixel_counts": {"hair_seed": int(hair_seed.sum()), "weak_candidate": int(weak.sum()), "final": int(final.sum())},
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
