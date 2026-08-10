# Medication Package Segmentation & Counting

A real-time computer-vision project for automated medication dispensing and package counting. The system combines **instance segmentation**, **tracking**, **ROI logic**, and ID filtering so that each object is counted once even when detections overlap or re-enter the region of interest.

This project corresponds to my SAC Marquage internship work on automated medication dispensing.

## Demo

▶ **[Watch the short repository demo](../../assets/demos/medication_segmentation_counting.mp4)**

The demo shows segmentation masks and a running total while medication packages pass through the counting area.

## Project pipeline

```text
Camera / recorded production video
        |
        v
YOLOv11-seg instance segmentation
        |
        v
ByteTrack persistent tracking
        |
        v
ROI + hysteresis + ID filtering
        |
        v
One-time object count
        |
        +----> operator status
        +----> annotated clips
        +----> Excel / structured logs
```

## Engineering work

- collected on-site video data
- polygon-annotated objects for instance segmentation
- fine-tuned YOLOv11-seg
- integrated ByteTrack for stable IDs
- implemented ROI and hysteresis logic to prevent double counting
- developed operator-oriented visualization and logging workflows

## Technologies

`Python` · `OpenCV` · `YOLOv11-seg` · `ByteTrack` · `Roboflow` · `PyTorch` · `PyQt` · `Excel`

## Source-code note

The demonstration is published for portfolio purposes. The original project source, customer-specific configuration, trained weights and production data are not included here.
