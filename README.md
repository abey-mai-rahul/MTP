
# Football Motion Analyzer — v1

A Streamlit prototype based on the uploaded `MTP.ipynb` notebook.

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Modes

### Single camera — 2D
- MOG2 background subtraction and contour detection, following notebook cell 4.
- Adjustable contour thresholds and region of interest.
- Linear X(t), quadratic Y(t) fit, and pixel-coordinate exports.
- Annotated video and CSV downloads.

### Two cameras — 3D
- YOLOv8s COCO class 32 (sports ball) detection, following notebook cells 45–51.
- One-to-one timestamp matching, with default 20 ms tolerance, following cell 58.
- Approximate phone intrinsics based on the notebook's 960×540 focal estimates:
  - iPhone 13: fx=fy=661.9 px
  - Motorola Edge 50 Neo: fx=fy=611 px
- Essential matrix / relative pose estimation, triangulation, quadratic fits and motion metrics.
- CSV exports and optional side-by-side tracking video.

## Important limitations

This is a first working application, not a calibrated measurement system.

1. The notebook uses approximate intrinsics and does not know the physical camera baseline.
2. The stereo pose is estimated from ball trajectory correspondences in this app, which is not a robust substitute for calibration from static scene features or a calibration target. A moving ball's correspondences can produce unstable geometry.
3. Translation from essential-matrix recovery has unit length only; reconstructed XYZ, speed and acceleration are in relative units.
4. The optional metre estimate assumes a 0.22 m ball diameter and uses detected box size. It is a rough estimate only.
5. Single-camera MOG2 can mistake other moving objects for the ball. Adjust the ROI and contour thresholds if needed.
6. Dual-camera recordings should be simultaneous, stationary, same action, with no zoom changes. Audio/clap sync or a visible sync event helps.
7. The first implementation uses the highest-confidence sports-ball detection in each sampled frame. Crowded scenes or multiple balls may confuse it.
8. For longer videos, processing can be slow and memory-intensive. Use the sampled-frame setting.

## Suggested next calibration upgrade

For dependable 3D results, calibrate each phone camera (intrinsics and distortion) and estimate stereo extrinsics from a checkerboard/Charuco board or many static background feature correspondences. Measure the camera baseline in metres. Only then report metric XYZ, speed and acceleration.


## v1.1 display improvements
- Dual-camera upload area previews both original videos side by side.
- Results show separate YOLO-annotated review videos for each camera.
- Dedicated X(t), Y(t), Z(t) plots, plus speed(t) and acceleration(t).
- Original and annotated videos are downloadable.


## v1.2 — Red trajectory trail videos
- Single-camera mode retains the original detection/bounding-box video and adds a second full-length video with a persistent red polyline tracing the ball-centre path.
- Dual-camera mode retains the separate YOLO detection videos and adds a red-line trajectory video for each phone.
- The optional combined side-by-side video also draws red trajectory trails in both views.
- Red trails are drawn from the cleaned trajectory detections. Missing detections do not create fabricated points; the trail joins successive available detections.


## v1.3 — Side-on 2D projectile analysis
- Adds a dedicated **2D projectile view** tab in dual-camera results.
- Projects the reconstructed 3D trajectory onto a side-on vertical plane. The horizontal axis follows the dominant horizontal travel direction in the reconstructed X-Z plane; the vertical axis uses reconstructed Y.
- Displays the projected 2D path, horizontal and vertical position vs time, fitted parametric equations x(t), y(t), and a direct y(x) quadratic fit.
- Exports the projected 2D trajectory as CSV.
- Important: this is a projection of the existing estimated 3D reconstruction, not an independent calibrated 2D measurement. Its vertical/world-axis interpretation depends on camera geometry and calibration.


## v1.6 — Automatic ground-contact origin (heuristic)
- The 2D projectile view estimates a ground-contact reference by selecting the lowest vertical point in the projected reconstructed trajectory.
- The selected point is shifted to (0, 0); positive horizontal is along the estimated direction of travel and positive vertical is above the estimated contact height.
- 2D position-time equations use time relative to the estimated contact moment.
- This cannot guarantee actual ball-ground contact when the clip does not show the landing/bounce, or when the vertical axis / stereo reconstruction is inaccurate. A calibrated ground plane or visible contact/bounce evidence is needed for reliable automatic contact detection.
