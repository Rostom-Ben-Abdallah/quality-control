"""Public industrial quality-control implementation.

This refactors the original prototype into a configurable pipeline for checking
laser-marked labels using OCR, geometric position tolerances and CIEDE2000 colour
difference. Production/customer configuration and private images are not included.

Config format (JSON):
{
  "expected_rgb": [235, 235, 235],
  "delta_e_max": 12.0,
  "position_tolerance_px": 18,
  "labels": {
    "Anmelden": [184, 142, 443, 209],
    "Telefon 1": [493, 145, 721, 209]
  }
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import easyocr
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


@dataclass
class Expectation:
    text: str
    box: np.ndarray  # [x1, y1, x2, y2]


def canonical(text: str) -> str:
    return " ".join(text.lower().strip().split())


def easyocr_box_to_xyxy(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def position_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Mean absolute corner displacement in pixels."""
    return float(np.mean(np.abs(actual - expected)))


def inset_roi(image: np.ndarray, box: np.ndarray, inset: float = 0.12) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    dx, dy = int(w * inset), int(h * inset)
    x1, x2 = max(0, x1 + dx), min(image.shape[1], x2 - dx)
    y1, y2 = max(0, y1 + dy), min(image.shape[0], y2 - dy)
    return image[y1:y2, x1:x2]


def median_rgb(roi_bgr: np.ndarray) -> np.ndarray:
    if roi_bgr.size == 0:
        return np.array([0.0, 0.0, 0.0])
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3)
    return np.median(pixels, axis=0).astype(np.float32)


def delta_e_2000(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    a = rgb2lab((rgb_a / 255.0).reshape(1, 1, 3))
    b = rgb2lab((rgb_b / 255.0).reshape(1, 1, 3))
    return float(deltaE_ciede2000(a, b)[0, 0])


def load_config(path: str):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_rgb = np.asarray(cfg["expected_rgb"], dtype=np.float32)
    labels: Dict[str, Expectation] = {}
    for text, box in cfg["labels"].items():
        labels[canonical(text)] = Expectation(text=text, box=np.asarray(box, dtype=np.float32))
    return cfg, expected_rgb, labels


def inspect(image: np.ndarray, config_path: str, gpu: bool = False):
    cfg, expected_rgb, expectations = load_config(config_path)
    position_tol = float(cfg.get("position_tolerance_px", 18))
    delta_e_max = float(cfg.get("delta_e_max", 12.0))
    ocr_min_conf = float(cfg.get("ocr_min_confidence", 0.25))

    reader = easyocr.Reader(["en"], gpu=gpu)
    results = reader.readtext(image)

    observed = {}
    for points, text, confidence in results:
        if confidence < ocr_min_conf:
            continue
        key = canonical(text)
        if key in expectations and (key not in observed or confidence > observed[key][1]):
            observed[key] = (easyocr_box_to_xyxy(points), float(confidence), text)

    annotated = image.copy()
    rows = []
    piece_ok = True

    for key, expected in expectations.items():
        detected = observed.get(key)
        if detected is None:
            piece_ok = False
            rows.append({"label": expected.text, "status": "MISSING"})
            x1, y1, x2, y2 = expected.box.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, f"MISSING: {expected.text}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            continue

        actual_box, confidence, raw_text = detected
        pos_err = position_error(actual_box, expected.box)
        roi = inset_roi(image, actual_box)
        measured_rgb = median_rgb(roi)
        de = delta_e_2000(measured_rgb, expected_rgb)

        pos_ok = pos_err <= position_tol
        colour_ok = de <= delta_e_max
        item_ok = pos_ok and colour_ok
        piece_ok &= item_ok

        colour = (0, 190, 0) if item_ok else (0, 0, 230)
        x1, y1, x2, y2 = actual_box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(
            annotated,
            f"{expected.text} | pos:{pos_err:.1f}px | dE00:{de:.1f}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
            cv2.LINE_AA,
        )

        rows.append(
            {
                "label": expected.text,
                "ocr_text": raw_text,
                "ocr_confidence": confidence,
                "position_error_px": round(pos_err, 2),
                "delta_e_2000": round(de, 2),
                "measured_rgb": measured_rgb.round(1).tolist(),
                "position_ok": pos_ok,
                "colour_ok": colour_ok,
                "status": "OK" if item_ok else "NG",
            }
        )

    status = "PASS" if piece_ok else "FAIL"
    banner_colour = (0, 160, 0) if piece_ok else (0, 0, 220)
    cv2.rectangle(annotated, (0, 0), (image.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(annotated, f"QUALITY CONTROL: {status}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_colour, 2, cv2.LINE_AA)
    return piece_ok, rows, annotated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/inspection_result.jpg")
    parser.add_argument("--report", default="outputs/inspection_report.json")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    passed, report, annotated = inspect(image, args.config, gpu=args.gpu)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, annotated)
    Path(args.report).write_text(json.dumps({"passed": passed, "checks": report}, indent=2), encoding="utf-8")
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
