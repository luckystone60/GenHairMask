from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageSegmentation,
    Sapiens2ForSemanticSegmentation,
)


BIREF_MODEL = "ZhengPeng7/BiRefNet_HR-matting"
SAPIENS2_MODEL = "facebook/sapiens2-seg-0.4b"
BIREF_REVISION = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"
SAPIENS2_REVISION = "449b3c5335e6722bb94990abdd1aa6e612432f22"
HAIR_CLASS_ID = 4


def save_gray(path: Path, array: np.ndarray, bit16: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(array, 0.0, 1.0)
    scale, dtype = (65535.0, np.uint16) if bit16 else (255.0, np.uint8)
    if not cv2.imwrite(str(path), np.rint(array * scale).astype(dtype)):
        raise OSError(f"Failed to write {path}")


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = (np.clip(mask, 0, 1) * 0.55)[..., None]
    out = rgb * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def final_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "logits"):
        return value.logits
    if isinstance(value, dict):
        for key in ("logits", "pred", "out"):
            if key in value:
                return final_tensor(value[key])
    if isinstance(value, (tuple, list)):
        return final_tensor(value[-1])
    raise TypeError(f"Unsupported model output: {type(value)!r}")


@torch.inference_mode()
def run_birefnet(image: Image.Image, device: torch.device, cache: Path) -> np.ndarray:
    model = AutoModelForImageSegmentation.from_pretrained(
        BIREF_MODEL,
        trust_remote_code=True,
        cache_dir=cache,
        revision=BIREF_REVISION,
    ).eval().to(device)
    if device.type == "cuda":
        model = model.half()

    transform = Compose([
        Resize((2048, 2048)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    if device.type == "cuda":
        x = x.half()
    logits = final_tensor(model(x))
    logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    matte = logits[:, :1].sigmoid()[0, 0].float().cpu().numpy()
    del model, x, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matte


@torch.inference_mode()
def run_sapiens2(image: Image.Image, device: torch.device, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    processor = AutoImageProcessor.from_pretrained(SAPIENS2_MODEL, cache_dir=cache, revision=SAPIENS2_REVISION)
    model = Sapiens2ForSemanticSegmentation.from_pretrained(
        SAPIENS2_MODEL,
        cache_dir=cache,
        revision=SAPIENS2_REVISION,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).eval().to(device)
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device.type == "cuda" and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].half()
    logits = model(**inputs).logits.float()
    logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    probs = logits.softmax(dim=1)
    hair_prob = probs[0, HAIR_CLASS_ID].cpu().numpy()
    labels = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
    return hair_prob, labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate BiRefNet portrait matte and Sapiens2 coarse hair mask.")
    ap.add_argument("--bok", type=Path, required=True)
    ap.add_argument("--edof", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(__file__).parent / "models")
    ap.add_argument("--hair-threshold", type=float, default=0.45)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    bok = Image.open(args.bok).convert("RGB")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.bok.stem

    print(f"[1/2] BiRefNet HR matting on {device} ...", flush=True)
    matte = run_birefnet(bok, device, args.cache)
    save_gray(args.output / f"{stem}_biref_bok_mat4k.png", matte, bit16=True)
    save_gray(args.output / f"{stem}_biref_bok_mat4k_8bit.png", matte)

    print(f"[2/2] Sapiens2 0.4B segmentation on {device} ...", flush=True)
    hair_prob, labels = run_sapiens2(bok, device, args.cache)
    hair_mask = hair_prob >= args.hair_threshold
    save_gray(args.output / f"{stem}_sapiens2_hair_probability_16bit.png", hair_prob, bit16=True)
    save_gray(args.output / f"{stem}_hair.png", hair_mask.astype(np.float32))
    cv2.imwrite(str(args.output / f"{stem}_sapiens2_labels.png"), labels)
    overlay(bok, hair_prob, (255, 0, 255)).save(args.output / f"{stem}_hair_overlay.jpg", quality=95)

    metadata = {
        "bok": str(args.bok.resolve()),
        "bok_size_wh": list(bok.size),
        "edof": str(args.edof.resolve()) if args.edof else None,
        "edof_size_wh": list(Image.open(args.edof).size) if args.edof else None,
        "output_coordinate_system": "bok",
        "birefnet_model": BIREF_MODEL,
        "birefnet_revision": BIREF_REVISION,
        "sapiens2_model": SAPIENS2_MODEL,
        "sapiens2_revision": SAPIENS2_REVISION,
        "sapiens2_hair_class_id": HAIR_CLASS_ID,
        "hair_threshold": args.hair_threshold,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {args.output.resolve()}")


if __name__ == "__main__":
    main()
