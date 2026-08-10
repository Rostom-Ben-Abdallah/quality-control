"""Bottle segmentation, counting and cap-quality inspection.

Public portfolio implementation of the vision logic demonstrated in the project video.
It requires compatible custom Ultralytics segmentation weights; private datasets and
trained production weights are intentionally not included.

Example:
    python bottle_cap_inspection.py --model weights/bottle_seg.pt --source demo.mp4

The script demonstrates:
- YOLO instance segmentation + persistent tracking
- one-time line-crossing counts by track ID
- bottle/cap association
- OK/NG classification (cap present / expected cap colour)
- live throughput, latency and yield overlays
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackState:
    last_x: Optional[float] = None
    counted: bool = False


def parse_source(value: str):
    """Accept a camera index (e.g. 0) or video path."""
    return int(value) if value.isdigit() else value


def normalize(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def contains_any(name: str, terms: Iterable[str]) -> bool:
    n = normalize(name)
    return any(term in n for term in terms)


def center(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box.astype(float)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def cap_belongs_to_bottle(cap_box: np.ndarray, bottle_box: np.ndarray) -> bool:
    """Associate a cap with the upper part of a bottle using geometry only."""
    cx, cy = center(cap_box)
    x1, y1, x2, y2 = bottle_box.astype(float)
    width = x2 - x1
    height = y2 - y1
    return (
        x1 - 0.08 * width <= cx <= x2 + 0.08 * width
        and y1 - 0.12 * height <= cy <= y1 + 0.38 * height
    )


def status_for_bottle(
    bottle_label: str,
    bottle_box: np.ndarray,
    detections: list[dict],
    expected_cap_colour: str,
) -> Tuple[str, Optional[dict]]:
    """Return (OK/NG/UNKNOWN, associated_cap)."""
    n = normalize(bottle_label)
    if "no cap" in n or "without cap" in n or "missing cap" in n:
        return "NG", None

    caps = [
        d
        for d in detections
        if contains_any(d["label"], ("cap", "closure"))
        and cap_belongs_to_bottle(d["box"], bottle_box)
    ]

    if not caps:
        # If the custom model directly encodes OK/NG in the bottle class, respect it.
        if " ok" in f" {n}" or n.endswith("ok"):
            return "OK", None
        if " ng" in f" {n}" or "defect" in n:
            return "NG", None
        return "UNKNOWN", None

    cap = max(caps, key=lambda d: d["conf"])
    cap_name = normalize(cap["label"])
    if expected_cap_colour and expected_cap_colour.lower() not in cap_name:
        return "NG", cap
    return "OK", cap


def overlay_mask(frame: np.ndarray, mask: np.ndarray, ok: bool, alpha: float = 0.28) -> None:
    colour = np.array((40, 190, 40) if ok else (40, 40, 210), dtype=np.uint8)
    binary = mask > 0.5
    if not np.any(binary):
        return
    frame[binary] = (
        frame[binary].astype(np.float32) * (1.0 - alpha)
        + colour.astype(np.float32) * alpha
    ).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Custom YOLO segmentation weights")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--line-x", type=float, default=0.50,
                        help="Counting line as fraction of image width")
    parser.add_argument("--expected-cap-colour", default="yellow")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--output", default="outputs/bottle_inspection.mp4")
    args = parser.parse_args()

    model = YOLO(args.model)
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_x = int(np.clip(args.line_x, 0.05, 0.95) * width)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height),
    )

    states: Dict[int, TrackState] = {}
    total = ok_count = ng_count = 0
    started = time.perf_counter()
    ema_latency_ms: Optional[float] = None

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        t0 = time.perf_counter()
        result = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            verbose=False,
        )[0]

        detections: list[dict] = []
        boxes = result.boxes
        names = result.names
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.int().cpu().numpy()
            ids = (
                boxes.id.int().cpu().numpy()
                if boxes.id is not None
                else np.arange(len(xyxy), dtype=int)
            )

            for i, (box, conf, cls_id, track_id) in enumerate(zip(xyxy, confs, classes, ids)):
                detections.append(
                    {
                        "box": box,
                        "conf": float(conf),
                        "label": names[int(cls_id)],
                        "track_id": int(track_id),
                        "mask": None if masks is None else masks[i],
                    }
                )

        bottle_dets = [d for d in detections if contains_any(d["label"], ("bottle", "vial"))]

        # Draw all detections first.
        for d in detections:
            x1, y1, x2, y2 = d["box"].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 60), 2)
            cv2.putText(
                frame,
                f'{d["label"]} {d["conf"]:.2f}',
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )

        # Evaluate and count bottles.
        for d in bottle_dets:
            tid = d["track_id"]
            bx = d["box"]
            cx, _ = center(bx)
            status, associated_cap = status_for_bottle(
                d["label"], bx, detections, args.expected_cap_colour
            )
            state = states.setdefault(tid, TrackState())

            crossed = (
                state.last_x is not None
                and state.last_x < line_x <= cx
                and not state.counted
            )
            if crossed:
                state.counted = True
                total += 1
                if status == "OK":
                    ok_count += 1
                else:
                    # UNKNOWN is conservative for production quality control.
                    ng_count += 1

            state.last_x = cx
            x1, y1, x2, y2 = bx.astype(int)
            is_ok = status == "OK"
            colour = (0, 210, 0) if is_ok else (0, 0, 230)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(
                frame,
                f"BOTTLE ID:{tid} {status}",
                (x1, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                colour,
                2,
                cv2.LINE_AA,
            )
            if d["mask"] is not None:
                mask = cv2.resize(d["mask"], (width, height), interpolation=cv2.INTER_NEAREST)
                overlay_mask(frame, mask, is_ok)

            if associated_cap is not None:
                cx1, cy1, cx2, cy2 = associated_cap["box"].astype(int)
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), colour, 2)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        ema_latency_ms = latency_ms if ema_latency_ms is None else 0.9 * ema_latency_ms + 0.1 * latency_ms
        elapsed_min = max((time.perf_counter() - started) / 60.0, 1e-6)
        cadence = total / elapsed_min
        yield_pct = 100.0 * ok_count / total if total else 0.0

        cv2.line(frame, (line_x, 0), (line_x, height), (255, 220, 0), 2)
        cv2.rectangle(frame, (0, 0), (width, 54), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"TOTAL:{total}  OK:{ok_count}  NG:{ng_count}  "
            f"Cadence:{cadence:.1f}/min  Yield:{yield_pct:.1f}%  "
            f"Latency:{ema_latency_ms:.1f}ms",
            (10, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        cv2.imshow("Bottle Segmentation / Counting / Quality Control", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
