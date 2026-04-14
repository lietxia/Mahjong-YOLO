#!/usr/bin/env python3
"""Generate Python reference baselines for the current web ONNX route.

Recommended invocation:

uv run --python 3.12 --with 'numpy<2' --with onnxruntime --with pillow --with pyyaml \
  python scripts/export_web_python_references.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/nano/mahjong-yolon-best.onnx"
CLASS_SOURCE = ROOT / "notebooks/data.yaml"
IMAGE_DIR = ROOT / "inference_validation"
OUTPUT_PATH = ROOT / "web_app/public/model/python-baselines.json"

CONFIDENCE = 0.3
IOU = 0.5
INPUT_SIZE = 640
PADDING_VALUE = 114 / 255


def load_classes() -> list[str]:
    data = yaml.safe_load(CLASS_SOURCE.read_text(encoding="utf-8"))
    classes = data["names"]
    if len(classes) != 37:
        raise ValueError(f"Expected 37 classes from {CLASS_SOURCE}, got {len(classes)}")
    return classes


def compute_letterbox_placement(
    width: int, height: int, input_size: int
) -> tuple[float, int, int, int, int]:
    scale = min(input_size / width, input_size / height)
    draw_width = round(width * scale)
    draw_height = round(height * scale)
    delta_width = input_size - draw_width
    delta_height = input_size - draw_height
    pad_x = round(delta_width / 2 - 0.1)
    pad_y = round(delta_height / 2 - 0.1)
    return scale, draw_width, draw_height, pad_x, pad_y


def map_target_to_source(
    target_index: int, target_length: int, source_length: int
) -> float:
    mapped = ((target_index + 0.5) * source_length) / target_length - 0.5
    return min(max(mapped, 0), source_length - 1)


def bilinear_channel(
    image: np.ndarray,
    source_x: float,
    source_y: float,
    channel: int,
) -> float:
    x0 = int(np.floor(source_x))
    y0 = int(np.floor(source_y))
    x1 = min(x0 + 1, image.shape[1] - 1)
    y1 = min(y0 + 1, image.shape[0] - 1)
    x_weight = source_x - x0
    y_weight = source_y - y0
    top = (
        float(image[y0, x0, channel]) * (1 - x_weight)
        + float(image[y0, x1, channel]) * x_weight
    )
    bottom = (
        float(image[y1, x0, channel]) * (1 - x_weight)
        + float(image[y1, x1, channel]) * x_weight
    )
    return (top * (1 - y_weight) + bottom * y_weight) / 255.0


def build_letterboxed_tensor(
    image_path: Path,
) -> tuple[np.ndarray, dict[str, float | int]]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    original_height, original_width = image.shape[:2]
    scale, draw_width, draw_height, pad_x, pad_y = compute_letterbox_placement(
        original_width, original_height, INPUT_SIZE
    )
    plane_size = INPUT_SIZE * INPUT_SIZE
    tensor = np.full((3, plane_size), PADDING_VALUE, dtype=np.float32)

    for target_y in range(draw_height):
        source_y = map_target_to_source(target_y, draw_height, original_height)
        for target_x in range(draw_width):
            source_x = map_target_to_source(target_x, draw_width, original_width)
            canvas_x = pad_x + target_x
            canvas_y = pad_y + target_y
            pixel_index = canvas_y * INPUT_SIZE + canvas_x
            tensor[0, pixel_index] = bilinear_channel(image, source_x, source_y, 0)
            tensor[1, pixel_index] = bilinear_channel(image, source_x, source_y, 1)
            tensor[2, pixel_index] = bilinear_channel(image, source_x, source_y, 2)

    return tensor.reshape(1, 3, INPUT_SIZE, INPUT_SIZE), {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "original_width": original_width,
        "original_height": original_height,
    }


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> list[float]:
    x, y, width, height = box
    return [x - width / 2, y - height / 2, x + width / 2, y + height / 2]


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def predict_ordered_tiles(
    session: ort.InferenceSession, classes: list[str], image_path: Path
) -> list[str]:
    tensor, meta = build_letterboxed_tensor(image_path)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    channels, anchors = output.shape[1], output.shape[2]
    flat = output.reshape(-1).tolist()

    detections = []
    for anchor_index in range(anchors):
        x = flat[anchor_index]
        y = flat[anchors + anchor_index]
        width = flat[anchors * 2 + anchor_index]
        height = flat[anchors * 3 + anchor_index]
        best_class = -1
        best_score = float("-inf")

        for class_offset in range(4, channels):
            candidate = flat[anchors * class_offset + anchor_index]
            if candidate > best_score:
                best_score = candidate
                best_class = class_offset - 4

        if best_score >= CONFIDENCE:
            detections.append(
                {
                    "class_id": best_class,
                    "score": best_score,
                    "bbox": xywh_to_xyxy((x, y, width, height)),
                }
            )

    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(
            existing["class_id"] == detection["class_id"]
            and compute_iou(existing["bbox"], detection["bbox"]) > IOU
            for existing in kept
        ):
            continue
        kept.append(detection)

    ordered = []
    for detection in kept:
        x1, y1, x2, y2 = detection["bbox"]
        restored = [
            min(meta["original_width"], max(0.0, (x1 - meta["pad_x"]) / meta["scale"])),
            min(
                meta["original_height"], max(0.0, (y1 - meta["pad_y"]) / meta["scale"])
            ),
            min(meta["original_width"], max(0.0, (x2 - meta["pad_x"]) / meta["scale"])),
            min(
                meta["original_height"], max(0.0, (y2 - meta["pad_y"]) / meta["scale"])
            ),
        ]
        ordered.append(
            (
                ((restored[0] + restored[2]) / 2),
                ((restored[1] + restored[3]) / 2),
                classes[detection["class_id"]],
            )
        )

    ordered.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ordered]


def main() -> int:
    classes = load_classes()
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    samples = []
    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        ordered_tiles = predict_ordered_tiles(session, classes, image_path)
        samples.append(
            {
                "imageName": image_path.name,
                "expectedOrderedTiles": ordered_tiles,
                "notes": [
                    f"Generated by scripts/export_web_python_references.py from {MODEL_PATH.relative_to(ROOT)}",
                    f"Class source: {CLASS_SOURCE.relative_to(ROOT)}",
                    f"Thresholds: conf={CONFIDENCE}, iou={IOU}, imgsz={INPUT_SIZE}",
                    "Route: onnxruntime + deterministic bilinear letterbox + current web-compatible decode/NMS",
                ],
                "source": "python-onnxruntime-web-route",
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(samples)} samples to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
