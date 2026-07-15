from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_edof_to_bok(edof: np.ndarray, bok: np.ndarray, work_scale: float = 0.35) -> tuple[np.ndarray, dict]:
    """Resize EDOF to BOK and measure (but do not force) the residual global transform.

    The AIGC BOK image is not pixel-identical to EDOF. Applying a transform fitted to
    one changed region can make another region worse. The robust median match offset
    is sub-pixel for the supplied pair, so resize-only is the safer canonical image.
    """
    h, w = bok.shape[:2]
    resized = cv2.resize(edof, (w, h), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(bok, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    small_e = cv2.resize(resized, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=6000)
    kb, db = sift.detectAndCompute(cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY), None)
    ke, de = sift.detectAndCompute(cv2.cvtColor(small_e, cv2.COLOR_BGR2GRAY), None)
    if db is None or de is None:
        raise RuntimeError("SIFT could not find enough features for EDOF/BOK registration")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(de, db, k=2)
    good = [a for a, b in pairs if a.distance < 0.70 * b.distance]
    if len(good) < 20:
        raise RuntimeError(f"Only {len(good)} reliable matches; registration is unsafe")
    src = np.float32([ke[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000, confidence=0.999
    )
    if affine is None or inliers is None or int(inliers.sum()) < 15:
        raise RuntimeError("RANSAC registration failed")
    affine = affine.astype(np.float64)
    affine[:, 2] /= work_scale
    info = {
        "method": "resize_to_bok; SIFT_RANSAC_similarity_is_diagnostic_only",
        "affine_applied": False,
        "matches": len(good),
        "inliers": int(inliers.sum()),
        "affine_resized_edof_to_bok": affine.tolist(),
    }
    return resized, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the four same-resolution base images in BOK coordinates.")
    ap.add_argument("--bok", type=Path, required=True)
    ap.add_argument("--edof", type=Path, required=True)
    ap.add_argument("--hair", type=Path, required=True)
    ap.add_argument("--matte", type=Path, required=True)
    ap.add_argument("--hair-probability", type=Path)
    ap.add_argument("--sapiens2-labels", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prefix", default="2p")
    args = ap.parse_args()

    bok = cv2.imread(str(args.bok), cv2.IMREAD_COLOR)
    edof = cv2.imread(str(args.edof), cv2.IMREAD_COLOR)
    hair = cv2.imread(str(args.hair), cv2.IMREAD_UNCHANGED)
    matte = cv2.imread(str(args.matte), cv2.IMREAD_UNCHANGED)
    if any(x is None for x in (bok, edof, hair, matte)):
        raise FileNotFoundError("One or more inputs could not be read")
    h, w = bok.shape[:2]
    for name, image in (("hair", hair), ("matte", matte)):
        if image.shape[:2] != (h, w):
            raise ValueError(f"{name} is {image.shape[1]}x{image.shape[0]}, expected {w}x{h}")

    aligned_edof, registration = estimate_edof_to_bok(edof, bok)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bok": args.output / f"{args.prefix}_bok.png",
        "edof": args.output / f"{args.prefix}_edof.png",
        "hair": args.output / f"{args.prefix}_bok_hair.png",
        "matte": args.output / f"{args.prefix}_bok_mat4k.png",
    }
    cv2.imwrite(str(outputs["bok"]), bok, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(outputs["edof"]), aligned_edof, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(outputs["hair"]), hair)
    cv2.imwrite(str(outputs["matte"]), matte)
    if args.hair_probability:
        prob = cv2.imread(str(args.hair_probability), cv2.IMREAD_UNCHANGED)
        if prob is None or prob.shape[:2] != (h, w):
            raise ValueError("hair probability cannot be read or has the wrong size")
        cv2.imwrite(str(args.output / f"{args.prefix}_bok_hair_probability_16bit.png"), prob)
    if args.sapiens2_labels:
        labels = cv2.imread(str(args.sapiens2_labels), cv2.IMREAD_UNCHANGED)
        if labels is None or labels.shape[:2] != (h, w):
            raise ValueError("Sapiens2 labels cannot be read or have the wrong size")
        cv2.imwrite(str(args.output / f"{args.prefix}_bok_sapiens2_labels.png"), labels)

    metadata = {
        "coordinate_system": "BOK native pixels",
        "target_size_wh": [w, h],
        "source_bok_size_wh": [bok.shape[1], bok.shape[0]],
        "source_edof_size_wh": [edof.shape[1], edof.shape[0]],
        "registration": registration,
        "outputs": {k: str(v.resolve()) for k, v in outputs.items()},
    }
    (args.output / "base_images_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
