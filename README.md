# Industrial Computer Vision — Quality Control, Segmentation & Counting

A portfolio repository for applied **real-time computer vision in industrial environments**. It groups three representative systems I worked on: laser-marking quality inspection, bottle/cap segmentation and quality checking, and segmentation-based medication counting.

> **Portfolio / confidentiality note:** public code is limited to material that can be shared safely. Customer production data, private datasets, trained production weights, PLC configuration, credentials, and confidential deployment assets are not included.

## Featured projects

### 1. Bottle Segmentation, Counting & Cap Quality Inspection

Real-time instance segmentation and tracking with bottle/cap association, OK/NG classification, cap-colour checking, one-time line-crossing counts, cadence and yield monitoring.

**[Project page + public code](projects/bottle-cap-inspection/README.md)**

https://github.com/user-attachments/assets/8f01de0a-532a-40ac-932f-2ff271ecea43

### 2. Laser-Marking Quality Control

Industrial inspection combining OCR, geometric position checking, colour comparison, visual validation and production-oriented PASS/FAIL decision logic.

**[Project page + public colour/position code](projects/laser-marking-quality-control/README.md)**

https://github.com/user-attachments/assets/8aa7c5ec-2791-41a1-9837-2bae320c3ea0

### 3. Medication Package Segmentation & Counting

YOLOv11-seg + ByteTrack pipeline for automated medication dispensing, with ROI/hysteresis logic and track-ID filtering to reduce double counts.

**[Project page](projects/medication-counting/README.md)**

https://github.com/user-attachments/assets/c11e8a28-19eb-49b8-8467-d3547293a64f

The medication project is documented and demonstrated publicly, while its original deployment source and production assets remain private.

## Industrial vision workflow

```text
Camera / video
      |
      v
Detection / instance segmentation
      |
      +----> tracking / persistent IDs
      +----> OCR / readability
      +----> position / geometry
      +----> colour / appearance
      |
      v
Temporal + ROI logic
      |
      v
Count / PASS / FAIL / alert
      |
      v
Operator UI / logs / production control
```

## Public code highlights

- [`projects/bottle-cap-inspection/bottle_cap_inspection.py`](projects/bottle-cap-inspection/bottle_cap_inspection.py) — segmentation, tracking, bottle/cap association, cap-quality checking and one-time counting.
- [`projects/laser-marking-quality-control/color_position_inspection.py`](projects/laser-marking-quality-control/color_position_inspection.py) — OCR, expected-position checks, CIEDE2000 colour comparison and structured PASS/FAIL reporting.
- Legacy prototype scripts are retained at the repository root as engineering history; the project folders above are the recommended portfolio entry points.

## Technologies

**Vision / AI:** Python · OpenCV · PyTorch · Ultralytics YOLO · YOLOv11-seg · EasyOCR · instance segmentation  
**Tracking:** ByteTrack · BoT-SORT · ROI / hysteresis / ID filtering  
**Quality:** CIEDE2000 · position validation · OCR · appearance checks  
**Deployment:** CUDA/cuDNN · PyQt · Excel logging · Modbus TCP / PLC integration

## Engineering focus

The important challenge in these projects is not simply achieving a detection. The systems must produce repeatable decisions in continuous video while handling motion, overlap, track changes, timing constraints and real production workflow requirements.

## Author

**Rostom Ben Abdallah**  
Industrial Computer Engineering · Computer Vision / Visual AI  
Mitacs Research Intern — Université de Moncton  
[GitHub Profile](https://github.com/Rostom-Ben-Abdallah) · [LinkedIn](https://www.linkedin.com/in/rostom-ben-abdallah-77bb441a1/)
