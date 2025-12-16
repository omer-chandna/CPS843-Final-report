#!/usr/bin/env python3
"""
Speed estimation from premade optical flow algorithms (sparse + dense).

Steps:
1. For each video, compute average motion magnitude from:
   - Sparse optical flow (Lucas–Kanade)
   - Dense optical flow (Farneback)
2. Fit a linear mapping from motion magnitude (px/frame) to speed (e.g., km/h)
   using your ground-truth speeds.
3. Estimate speed for each video and compute MAE for both algorithms.

Usage:
    python speed_from_optical_flow.py
"""

import cv2
import numpy as np
import csv
from typing import Dict, Optional, Tuple

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# ----------------------------------------------------------------------
# 1. config: videos + ground-truth speeds
# ----------------------------------------------------------------------

VIDEOS: Dict[str, str] = {
    "indoors_slow":  "videos/indoors_slow.mp4",
    "indoors_med":   "videos/indoors_med.mp4",
    "indoors_fast":  "videos/indoors_fast.mp4",
    "outdoors_bright_slow": "videos/outdoor_bright_slow.mp4",
    "outdoors_bright_med":  "videos/outdoors_bright_med.mp4",
    "outdoors_bright_fast": "videos/outdoors_bright_fast.mp4",
    "outdoors_dark_slow": "videos/outdoors_dark_slow.mp4",
    "outdoors_dark_med":  "videos/outdoors_dark_med.mp4",
    "outdoors_dark_fast": "videos/outdoors_dark_fast.mp4"
}

# ground-truth average speed (e.g., km/h) for each video.
TRUE_SPEEDS: Dict[str, float] = {
    "indoors_slow": 1.1,
    "indoors_med":  2.3,
    "indoors_fast": 3.1,
    "outdoors_bright_slow": 1.2,
    "outdoors_bright_med":  2.2,
    "outdoors_bright_fast": 3.2,
    "outdoors_dark_slow": 1.3,
    "outdoors_dark_med":  2.3,
    "outdoors_dark_fast": 3.1,
}


CALIBRATION_KEYS = [
# will calibrate off all videos that have TRUE_SPEEDS
]


# ----------------------------------------------------------------------
# 2. sparse optical flow (lucas–kanade)
# ----------------------------------------------------------------------

def analyze_sparse_flow(
    video_path: str,
    max_frames: int = 300,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.5,
) -> Optional[float]:
    """
    compute average per-frame motion magnitude (px/frame)
    using sparse Lucas–Kanade optical flow.

    args:
        blur_kernel_size: size of gaussian blur kernel (must be odd)
        blur_sigma: standard deviation for gaussian blur

    returns:
        mean_motion_mag_over_video (float) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print(f"[SPARSE] Could not read first frame from {video_path}")
        cap.release()
        return None

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # apply gaussian blur preprocessing
    old_gray = cv2.GaussianBlur(old_gray, (blur_kernel_size, blur_kernel_size), blur_sigma)

    # detect corners
    feature_params = dict(
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    if p0 is None or len(p0) == 0:
        print(f"[SPARSE] No features found in first frame for {video_path}")
        cap.release()
        return None

    # lucas–kanade params
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    frame_idx = 0
    frame_magnitudes = []

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur preprocessing
        frame_gray = cv2.GaussianBlur(frame_gray, (blur_kernel_size, blur_kernel_size), blur_sigma)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is None or st is None:
            break

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_old) == 0:
            break

        # motion vectors
        motion = good_new - good_old
        mags = np.linalg.norm(motion, axis=1)

        frame_magnitudes.append(float(np.mean(mags)))

        # prepare for next frame
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        frame_idx += 1

    cap.release()

    if len(frame_magnitudes) == 0:
        return None

    return float(np.mean(frame_magnitudes))


# ----------------------------------------------------------------------
# 3. dense optical flow (farneback)
# ----------------------------------------------------------------------

def analyze_dense_flow(
    video_path: str,
    max_frames: int = 300,
    roi: Optional[Tuple[int, int, int, int]] = None,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.5,
) -> Optional[float]:
    """
    compute average per-frame motion magnitude (px/frame)
    using dense Farneback optical flow.

    args:
        roi: Optional (x, y, w, h) region of interest in the frame.
             if provided, magnitude is averaged only over this region.
        blur_kernel_size: size of gaussian blur kernel (must be odd)
        blur_sigma: standard deviation for gaussian blur

    returns:
        mean_motion_mag_over_video (float) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"[DENSE] Could not read first frame from {video_path}")
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # apply gaussian blur preprocessing
    prev_gray = cv2.GaussianBlur(prev_gray, (blur_kernel_size, blur_kernel_size), blur_sigma)
    frame_idx = 0
    frame_magnitudes = []

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur preprocessing
        gray = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), blur_sigma)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if roi is not None:
            x, y, w, h = roi
            mag_roi = mag[y:y + h, x:x + w]
            frame_magnitudes.append(float(np.mean(mag_roi)))
        else:
            frame_magnitudes.append(float(np.mean(mag)))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if len(frame_magnitudes) == 0:
        return None

    return float(np.mean(frame_magnitudes))


# ----------------------------------------------------------------------
# 4. linear calibration: motion (px/frame) -> speed (e.g., km/h)
# ----------------------------------------------------------------------

def fit_linear_mapping(
    mags: np.ndarray,
    true_speeds: np.ndarray,
) -> Tuple[float, float]:
    """
    fit v ≈ a * M + b using least-squares (numpy.polyfit).

    returns:
        a, b
    """
    # polyfit with degree 1 returns [a, b] (already lowercase)
    a, b = np.polyfit(mags, true_speeds, 1)
    return float(a), float(b)


# ----------------------------------------------------------------------
# 5. main pipeline
# ----------------------------------------------------------------------

def main():
    # check that we have GT for at least some videos
    common_keys = [k for k in VIDEOS.keys() if k in TRUE_SPEEDS]
    if not common_keys:
        print("ERROR: TRUE_SPEEDS is empty or does not match VIDEOS.")
        print("Please fill TRUE_SPEEDS with ground-truth speeds for your clips.")
        return

    print("Videos with ground-truth speeds:", common_keys)

    if CALIBRATION_KEYS:
        calib_keys = [k for k in CALIBRATION_KEYS if k in common_keys]
    else:
        calib_keys = common_keys

    if not calib_keys:
        print("ERROR: No valid calibration keys. Check CALIBRATION_KEYS.")
        return

    print("Using these videos for calibration:", calib_keys)

    # store results
    results = []

    sparse_mags = {}
    dense_mags = {}

    roi = None  # set to (x, y, w, h) if you want to restrict region of interest

    # 1) compute magnitudes for all videos
    for name, path in VIDEOS.items():
        print(f"\nProcessing video: {name} -> {path}")

        sparse_mag = analyze_sparse_flow(path)
        dense_mag = analyze_dense_flow(path, roi=roi)

        if sparse_mag is None or dense_mag is None:
            print(f"Skipping {name} because one of the analyses failed.")
            continue

        sparse_mags[name] = sparse_mag
        dense_mags[name] = dense_mag

        gt_speed = TRUE_SPEEDS.get(name, None)

        results.append({
            "name": name,
            "video_path": path,
            "gt_speed": gt_speed,
            "sparse_mag": sparse_mag,
            "dense_mag": dense_mag,
        })

    # filter to only videos that have both magnitudes and ground-truth speeds
    valid_for_eval = [
        r for r in results
        if (r["gt_speed"] is not None)
    ]

    if not valid_for_eval:
        print("No videos with both optical flow results and ground-truth speeds.")
        return

    # 2) build calibration sets from videos that have both magnitudes and ground-truth speeds
    sparse_mags_calib = []
    dense_mags_calib = []
    speeds_calib = []

    for name in calib_keys:
        if name not in sparse_mags or name not in dense_mags:
            print(f"[CALIB] Warning: {name} missing mags; skipping from calibration.")
            continue
        if name not in TRUE_SPEEDS:
            print(f"[CALIB] Warning: {name} missing GT speed; skipping from calibration.")
            continue

        sparse_mags_calib.append(sparse_mags[name])
        dense_mags_calib.append(dense_mags[name])
        speeds_calib.append(TRUE_SPEEDS[name])

    sparse_mags_calib = np.array(sparse_mags_calib, dtype=float)
    dense_mags_calib = np.array(dense_mags_calib, dtype=float)
    speeds_calib = np.array(speeds_calib, dtype=float)

    if len(speeds_calib) < 2:
        print("Not enough calibration samples to fit a line (need at least 2).")
        return

    # 3) fit v ≈ a * M + b for sparse and dense
    a_s, b_s = fit_linear_mapping(sparse_mags_calib, speeds_calib)
    a_d, b_d = fit_linear_mapping(dense_mags_calib, speeds_calib)

    print("\n=== Linear Calibration ===")
    print(f"Sparse: v ≈ {a_s:.4f} * M_sparse + {b_s:.4f}")
    print(f"Dense : v ≈ {a_d:.4f} * M_dense  + {b_d:.4f}")

    # 4) estimate speeds for all valid videos and compute errors
    mae_sparse = []
    mae_dense = []

    for r in valid_for_eval:
        name = r["name"]
        gt = float(r["gt_speed"])
        Ms = float(r["sparse_mag"])
        Md = float(r["dense_mag"])

        v_hat_s = a_s * Ms + b_s
        v_hat_d = a_d * Md + b_d

        err_s = abs(v_hat_s - gt)
        err_d = abs(v_hat_d - gt)

        r["v_hat_sparse"] = v_hat_s
        r["v_hat_dense"] = v_hat_d
        r["err_sparse"] = err_s
        r["err_dense"] = err_d

        mae_sparse.append(err_s)
        mae_dense.append(err_d)

    mean_mae_sparse = float(np.mean(mae_sparse)) if mae_sparse else None
    mean_mae_dense = float(np.mean(mae_dense)) if mae_dense else None

    # 5) print summary
    print("\n=== Per-video Results ===")
    header = (
        f"{'Name':25s} | {'GT':>6s} | "
        f"{'Sparse_hat':>10s} | {'Sparse_err':>10s} | "
        f"{'Dense_hat':>10s} | {'Dense_err':>10s}"
    )
    print(header)
    print("-" * len(header))

    for r in valid_for_eval:
        print(
            f"{r['name']:25s} | "
            f"{r['gt_speed']:6.3f} | "
            f"{r['v_hat_sparse']:10.3f} | {r['err_sparse']:10.3f} | "
            f"{r['v_hat_dense']:10.3f} | {r['err_dense']:10.3f}"
        )

    print("\n=== Mean Absolute Error (MAE) ===")
    if mean_mae_sparse is not None:
        print(f"Sparse MAE: {mean_mae_sparse:.3f}")
    if mean_mae_dense is not None:
        print(f"Dense  MAE: {mean_mae_dense:.3f}")

    # 6) write CSV 
    csv_path = "speed_estimation_results.csv"
    fieldnames = [
        "name",
        "video_path",
        "gt_speed",
        "sparse_mag",
        "dense_mag",
        "v_hat_sparse",
        "v_hat_dense",
        "err_sparse",
        "err_dense",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in valid_for_eval:
            writer.writerow(r)

    print(f"\nSaved detailed results to: {csv_path}")


if __name__ == "__main__":
    main()
