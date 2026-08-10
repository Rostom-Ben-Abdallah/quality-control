# Laser-Marking Quality Control

Real-time visual quality-control work for laser-marked industrial parts. The project combines **OCR**, **position validation**, **colour-difference checks**, and production-style PASS/FAIL decision logic.

This is the project described on my CV as a real-time laser-marking quality-control system using two specialized YOLO models, OpenCV validation, DeltaE2000, OCR and PLC/Modbus integration.

## Demo

▶ **[Watch the short repository demo](../../assets/demos/laser_marking_quality_control.mp4)**

The clip shows the inspection setup and visual validation interface in operation. The repository video is intentionally compressed for quick viewing.

## Public code

[`color_position_inspection.py`](color_position_inspection.py) is a cleaned public version of the colour/position/OCR validation logic. It demonstrates:

- EasyOCR label detection
- expected-position checking
- median colour measurement in the detected ROI
- CIEDE2000 colour difference
- configurable position and colour tolerances
- missing-label detection
- PASS / FAIL overlay
- structured JSON report export

An example configuration is provided in [`config_example.json`](config_example.json).

```bash
python color_position_inspection.py \
  --image sample_part.jpg \
  --config config_example.json
```

## Full industrial pipeline

```text
Industrial camera
      |
      v
Detection / localization
      |
      +----> OCR / readability
      +----> position validation
      +----> colour / DeltaE2000
      +----> opacity / appearance checks
      |
      v
PASS / FAIL decision
      |
      v
PLC / Modbus production logic
```

Production images, customer configuration, trained production weights and PLC assets are not published.

## Technologies

`Python` · `OpenCV` · `EasyOCR` · `YOLO` · `DeltaE2000` · `CUDA/cuDNN` · `Modbus TCP` · `PLC Integration`
