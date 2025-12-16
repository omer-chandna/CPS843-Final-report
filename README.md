# Optical Flow Based Navigation for GPS-Denied Environments

This repository contains the implementation and experimental results for speed estimation using optical flow algorithms, developed as part of CPS843 Final Report research on navigation in GPS-denied environments.

## Overview

In GPS-denied environments (e.g., indoor spaces, tunnels, urban canyons), traditional navigation systems fail. This project explores the use of optical flow algorithms to estimate vehicle/robot speed from video sequences, providing an alternative navigation solution that relies solely on visual information.

## Methodology

The project implements and compares two optical flow algorithms:

1. **Sparse Optical Flow (Lucas-Kanade)**: Tracks a limited set of feature points across frames
2. **Dense Optical Flow (Farneback)**: Computes motion vectors for every pixel in the frame

Both algorithms compute motion magnitude (pixels/frame) from video sequences, which is then calibrated to real-world speed (km/h) using ground-truth measurements through linear regression.

### Algorithm Parameters

#### Sparse Optical Flow (Lucas-Kanade)
- **Feature Detection**:
  - `maxCorners`: 500
  - `qualityLevel`: 0.01
  - `minDistance`: 7 pixels
  - `blockSize`: 7
- **Lucas-Kanade Tracking**:
  - `winSize`: (21, 21)
  - `maxLevel`: 3 (pyramid levels)
  - `criteria`: 30 iterations, 0.01 epsilon threshold

#### Dense Optical Flow (Farneback)
- `pyr_scale`: 0.5
- `levels`: 3
- `winsize`: 15
- `iterations`: 3
- `poly_n`: 5
- `poly_sigma`: 1.2

## Dataset

The dataset consists of 9 video sequences captured under different conditions:
- **Indoor environments**: slow, medium, fast speeds
- **Outdoor bright conditions**: slow, medium, fast speeds  
- **Outdoor dark conditions**: slow, medium, fast speeds

All videos are stored in the `videos/` directory.

## Installation

### Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib (for plotting, optional)

### Setup

```bash
# Install dependencies
pip install opencv-python numpy matplotlib seaborn pandas
```

## Usage

### Running the Analysis

```bash
python main.py
```

This will:
1. Process all videos in the `videos/` directory
2. Compute motion magnitudes using both sparse and dense optical flow
3. Calibrate motion magnitude to speed using ground-truth measurements
4. Estimate speeds for all videos and compute errors
5. Generate `speed_estimation_results.csv` with detailed results

### Output Files

- **`speed_estimation_results.csv`**: Detailed results for each video including:
  - Ground-truth speeds
  - Motion magnitudes (sparse and dense)
  - Predicted speeds
  - Absolute errors

- **`results/` directory**: Contains visualization plots:
  - `gt_vs_predicted.png`: Ground truth vs predicted speeds scatter plots
  - `error_comparison.png`: Error comparison between methods
  - `calibration_curves.png`: Motion magnitude vs speed calibration curves
  - `per_video_results.png`: Comprehensive per-video analysis
  - `summary_statistics.png`: Overall method comparison

### Performance Metrics
- **Sparse Optical Flow MAE**: 0.625 km/h
- **Dense Optical Flow MAE**: 0.307 km/h

The dense optical flow method demonstrates superior accuracy, achieving approximately 50% lower mean absolute error compared to sparse optical flow.

## Project Structure

```
CPS843-Final-report/
├── main.py                          # Main analysis script
├── README.md                        # This file
├── videos/                          # Input video sequences
│   ├── indoors_slow.mp4
│   ├── indoors_med.mp4
│   ├── indoors_fast.mp4
│   ├── outdoor_bright_slow.mp4
│   ├── outdoors_bright_med.mp4
│   ├── outdoors_bright_fast.mp4
│   ├── outdoors_dark_slow.mp4
│   ├── outdoors_dark_med.mp4
│   └── outdoors_dark_fast.mp4
└── results/                         # Generated results and plots
    ├── speed_estimation_results.csv
    ├── gt_vs_predicted.png
    ├── error_comparison.png
    ├── calibration_curves.png
    ├── per_video_results.png
    └── summary_statistics.png
```

## Configuration

Edit `main.py` to modify:
- Video paths (`VIDEOS` dictionary)
- Ground-truth speeds (`TRUE_SPEEDS` dictionary)
- Algorithm parameters (in respective function definitions)
- Maximum frames to process (`max_frames` parameter)

## Software Versions

The code prints OpenCV and NumPy versions on execution. To check manually:

```python
import cv2
import numpy as np
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
```

## Future Work

Potential extensions:
- Handling of more challenging scenarios (occlusions, lighting variations)
- Multi-camera fusion for depth estimation

## Authors

Omer Chandna,
Haarish Mansur,
Rejab Rehan