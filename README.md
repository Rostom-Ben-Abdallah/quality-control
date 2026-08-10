# Real-Time Industrial Vision Quality Control

A computer-vision quality-control prototype for inspecting laser-marked industrial parts in real time. The project combines deep-learning detection with deterministic OpenCV checks so that visual defects can be detected before a part continues through production.

> **Portfolio note:** this repository is a public, non-confidential representation of the engineering work. Proprietary production data, customer assets, trained production weights, PLC configuration, and internal deployment details are intentionally not published.

## What the system demonstrates

- Real-time image acquisition and visual inspection
- YOLO-based object/region detection
- Position and geometry validation with OpenCV
- Color-difference checking using DeltaE-style measurements
- Opacity / visual-quality checks
- OCR-oriented readability validation
- GPU-oriented inference workflow
- Integration mindset for industrial decision logic and line control

## Engineering context

The original project was developed as an end-of-year industrial computer engineering project. The complete system connected an industrial camera and AI/vision pipeline to production decision logic, including safe stop / acknowledgement / restart behavior through industrial communication.

The main challenge was not simply detecting an object: it was producing a repeatable **PASS / FAIL decision under real production constraints**, including varying appearance, timing requirements, and multiple quality criteria.

## High-level pipeline

```text
Industrial camera
      |
      v
Frame acquisition
      |
      v
YOLO detection / ROI localization
      |
      +-------------------+
      |                   |
      v                   v
Geometry / position     Visual checks
OpenCV                  color / opacity / OCR
      |                   |
      +---------+---------+
                |
                v
          Quality decision
          PASS / FAIL
                |
                v
       Industrial control layer
```

## Technologies

`Python` · `OpenCV` · `YOLO` · `PyTorch` · `EasyOCR` · `CUDA/cuDNN` · `Computer Vision` · `Industrial Automation`

## Why this project matters

This project reflects the part of computer vision that interests me most: taking a model beyond an offline notebook and integrating it into a complete real-time system where reliability, latency, repeatability, and failure handling matter.

## Repository structure

The repository contains prototype Python scripts, requirements, and example data/assets used for experimentation. Some files reflect iterative development rather than a packaged production release.

## Related research interests

I am currently interested in graduate research involving:

- object detection and segmentation
- multi-object tracking and re-identification
- video understanding and action recognition
- robust visual perception under real-world conditions
- intelligent robotics and industrial vision
- computer vision for animal and human behavior analysis

## Author

**Rostom Ben Abdallah**  
Industrial Computer Engineering · Computer Vision / Visual AI  
Mitacs Research Intern, Université de Moncton  
[LinkedIn](https://www.linkedin.com/in/rostom-ben-abdallah-77bb441a1/) · [GitHub](https://github.com/Rostom-Ben-Abdallah)
