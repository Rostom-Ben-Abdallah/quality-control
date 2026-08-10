# Bottle Segmentation, Counting & Cap Quality Inspection

A real-time industrial computer-vision demo combining **instance segmentation**, **persistent tracking**, **one-time counting**, and **cap-quality validation**.

The demonstrated pipeline identifies bottles as they move through the camera view, segments each bottle, associates the cap region, classifies the result as **OK / NG**, and maintains production-style counters for total pieces, accepted pieces, rejected pieces, throughput and yield.

## Demo

https://github.com/user-attachments/assets/8f01de0a-532a-40ac-932f-2ff271ecea43

The video is embedded using GitHub's native attachment player, so it can be played directly in the README without downloading it first.

## Pipeline

```text
Camera / video
     |
     v
YOLO instance segmentation + tracking
     |
     +----> bottle mask / track ID
     |
     +----> cap detection / association
                    |
                    v
           cap present + colour check
                    |
              +-----+-----+
              |           |
             OK          NG
              |           |
              +-----+-----+
                    |
           line-crossing counter
                    |
                    v
       Total / OK / NG / cadence / yield
```

## Public code

[`bottle_cap_inspection.py`](bottle_cap_inspection.py) is a clean portfolio implementation of the demonstrated logic. It supports:

- Ultralytics YOLO segmentation models
- BoT-SORT persistent track IDs
- segmentation-mask visualization
- bottle/cap geometric association
- expected cap-colour checking
- count-once line crossing
- conservative UNKNOWN → NG production logic
- latency, cadence and yield overlays
- annotated video export

The custom model weights and private training data are **not** published. Supply compatible segmentation weights with `--model`.

```bash
python bottle_cap_inspection.py \
  --model weights/bottle_seg.pt \
  --source input.mp4 \
  --expected-cap-colour yellow
```

## Technologies

`Python` · `OpenCV` · `Ultralytics YOLO` · `Instance Segmentation` · `BoT-SORT` · `Real-Time Tracking` · `Industrial Vision`

## Portfolio scope

This repository version is intended to demonstrate the engineering approach without exposing proprietary datasets, production weights, customer configuration or deployment-specific assets.
