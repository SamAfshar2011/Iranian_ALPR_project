```markdown
# Iranian License Plate Recognition (ALPR)

A full multi-stage Iranian Automatic License Plate Recognition (ALPR) pipeline for detecting license plates, segmenting plate characters, and recognizing the final plate text from images or video streams.

This project is built as a three-model pipeline and an inference workflow that connects all stages into a single end-to-end system. It supports multiple plates per frame, works on both images and videos, and can track detected plates over time to assign a persistent ID in video mode.

## Overview

The system is designed as a cascaded recognition pipeline:

1. **Model 1 — Whole Plate Detection**  
   Detects every license plate in the input image/frame and crops each detected plate region.

2. **Model 2 — Character Region Extraction**  
   Receives each cropped plate image and extracts the eight character regions of an Iranian plate.  
   This model does **not** classify the characters. Its role is only to localize and crop them in order.

3. **Model 3 — Character Recognition**  
   Receives the eight cropped character images and predicts the identity of each character.

The final output is a structured Iranian plate string, for example:

`20 879 B 90`

## Pipeline Architecture

### Stage 1 — Plate Detection
The first model is responsible for detecting all plate instances in a full image or video frame.  
Its output is one or more bounding boxes corresponding to full plates.

- **Model type:** YOLO
- **Backbone / checkpoint:** `yolo26s`
- **Primary responsibility:** find every plate in the scene
- **Output:** cropped plate images, one crop per detected plate

This stage is designed for multi-plate scenarios, meaning it preserves all valid plate detections instead of assuming only a single target vehicle or plate per image.

### Stage 2 — Character Localization and Ordering
The second model takes a cropped plate image and extracts all eight character regions.

Important detail: this model does **not** decide what the characters are.  
It only identifies where the characters are and returns the eight crops in reading order:

`C1 C2 C3 C4 C5 C6 C7 C8`

This explicit separation between **localization** and **classification** makes the pipeline easier to debug and more modular.

- **Model type:** YOLO
- **Backbone / checkpoint:** `yolo26l`
- **Input:** cropped plate image from Model 1
- **Output:** eight ordered character crops

### Stage 3 — Character Classification
The third model takes the eight character crops and classifies each one independently.

- **Model type:** ConvNeXt Small
- **Input:** ordered character crops from Model 2
- **Output:** predicted character identity for each crop

The final recognized plate text is reconstructed from the ordered predictions.

## Inference Workflow

The full pipeline is connected in `inference.ipynb`.

At inference time, the process is:

1. Read an input image or video
2. Detect all plates in the frame
3. Crop each plate region
4. Extract and sort the eight character crops for each plate
5. Classify each cropped character
6. Reconstruct the final plate string
7. Render the result on the output frame

For videos, the system also supports **tracking**, allowing each detected plate to receive a unique ID and be followed across consecutive frames.

## Project Structure

A typical project layout looks like this:

```text
.
├── inference.ipynb
├── model_1.ipynb
├── model_2.ipynb
├── model_3.ipynb
├── Datasets/
│   ├── DS_model_1/
│   ├── DS_model_2/
│   └── DS_model_3/
├── Saved_models/
│   ├── model_1/
│   ├── model_2/
│   └── model_3/
└── yolo models/
```

## Model Breakdown

### Model 1 — Whole Plate Detector
`model_1.ipynb`

This notebook trains the first-stage detector that finds complete plate regions in full images.

#### Key characteristics
- Uses full scene / vehicle images as input
- Uses XML annotations placed beside the corresponding images
- Trains a single-class detector:
  - `class 0 = plate`
- Preserves multiple plate instances per image
- Skips invalid samples instead of silently corrupting labels
- Reads image dimensions directly from image files instead of trusting XML metadata

#### Annotation handling
The training logic is intentionally strict.

Only objects whose label is exactly:

`کل ناحیه پلاک`

are used as valid whole-plate boxes for Model 1.

Character-level boxes are ignored for this model by default.

This is a very good design choice, by the way — because mixing full-plate detection labels with character-level annotations in the same detector is how people accidentally teach their model chaos and then act surprised when it predicts nonsense.

#### Data safety rules
The notebook enforces a conservative data contract:

- no train/validation mixing
- no automatic test split creation
- no physical deletion of invalid files
- no blind reliance on XML image size metadata
- no uncontrolled external augmentation
- no silent acceptance of malformed boxes
- out-of-bound boxes are validated and handled carefully

This makes the preprocessing pipeline reproducible and easier to audit.

## Why the Three-Stage Design?

Instead of trying to solve everything with one giant model, this project splits the task into specialized stages:

- **Detection** answers: *Where is the plate?*
- **Segmentation/localization** answers: *Where are the characters?*
- **Recognition** answers: *What is each character?*

This decomposition has several practical advantages:

- easier training and debugging
- cleaner failure analysis
- more modular experimentation
- the ability to replace one stage without retraining the entire stack
- better control over Iranian plate formatting constraints

In other words: less magical thinking, more engineering.

## Supported Inputs

The inference workflow supports:

- **Images**
- **Videos**

For videos, detected plates can be tracked over time and assigned a stable identity.

## Example Output

A typical recognition result is formatted like:

`20 879 B 90`

If multiple plates are present, the system can detect and process all of them independently.

## Training Notes

Each notebook is responsible for one stage of the pipeline:

- `model_1.ipynb` — whole plate detection
- `model_2.ipynb` — character localization / extraction
- `model_3.ipynb` — character classification
- `inference.ipynb` — full end-to-end inference

The training and inference logic are intentionally separated, which keeps experimentation cleaner and deployment logic easier to maintain.

## Requirements

The exact environment depends on the notebook contents, but the project is built around Python and common deep learning / computer vision libraries.

Typical dependencies include:

- Python 3.x
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Jupyter Notebook

If you want this repository to be easy for others to run, it is strongly recommended to add a `requirements.txt` or `environment.yml`.

## Recommended Repository Additions

To make the project more complete on GitHub, consider adding:

- `requirements.txt`
- `README` sample predictions
- model weight download instructions
- dataset preparation notes
- a small `configs/` directory for paths and thresholds
- a Python inference script version in addition to the notebook
- example input/output media

## Limitations

Like any ALPR system, performance can be affected by:

- motion blur
- low resolution plates
- strong perspective distortion
- occlusion
- poor lighting
- rare plate styles or damaged plates
- detection errors propagating into downstream recognition stages

Since the pipeline is sequential, an early-stage failure can affect later stages. That tradeoff is normal for cascaded systems and should be considered during evaluation.

## Future Improvements

Possible next steps for the project:

- export inference from notebook form into a production-ready Python package
- add evaluation metrics for all three stages
- benchmark runtime on image and video workloads
- add confidence reporting for plate-level predictions
- improve temporal fusion in video tracking mode
- support batch inference and structured JSON outputs
- add deployment endpoints or a lightweight demo app

## Summary

This repository implements a full Iranian ALPR pipeline with:

- multi-plate detection
- ordered character extraction
- character-level recognition
- image and video inference
- plate tracking with IDs in video mode

The project is organized around three specialized models:

- **YOLO26s** for whole plate detection
- **YOLO26l** for ordered character extraction
- **ConvNeXt Small** for character classification

Together, they form a modular and practical end-to-end license plate recognition system.
```
