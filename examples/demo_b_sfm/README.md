# Demo B: structure from motion around a water-filled tube

```sh
python examples/demo_b_sfm/demo_b.py                 # renders, matches and reconstructs: about 15 minutes
python examples/demo_b_sfm/demo_b.py --models colmap,colmap_air
python examples/demo_b_sfm/figures.py
```

It needs `pip install .[demo]` plus `opencv-python` and `pycolmap`. It writes `demo_b_out.txt`, `results.json` and
`summary.png` beside the script. Renders and intermediate results go to `_cache/`, which is not committed.

## The set-up

- **The tube:** the paper's reference port, acrylic (radius 100 mm, 6 mm wall, n = 1.49) filled with water
  (n = 1.33). Inside, textured stones stand on a gravel floor, several of them within 10 mm of the wall.
- **The camera:** a robot-held camera photographs the tube from 60 poses on three rings, 0.34–0.40 m from its axis.
  Each image is 1280 × 1024 px with f = 1500 px.
- **The images** are ray-traced through the tube by `refractive_camera.scene`, which uses no closed form of the
  projection.
- **What the robot knows:**
  - its poses roughly, as from its kinematics: errors of 1.5 mm and 0.2° per axis, or 2.1 mm and 0.29° at the median;
  - the tube's place and size, and the camera's intrinsics, as after a calibration in air.

## The pipeline

1. **Features:** SIFT (OpenCV), about 850 per view, kept where the pixel's ray enters the water.
2. **Matches:** between views up to 50° apart around the tube, with a ratio test and mutual nearest neighbours.
   A match is kept if the two keypoints' rays, traced through the window with the prior poses, pass within 8 mm of
   each other in the water: 38,678 of 47,352.
3. **Tracks** of three or more views: 2,558.
4. **Triangulation** with the prior poses.
5. **Bundle adjustment** of all poses and points, with a Huber loss and the pose priors:
   - drop the observations the model cannot explain;
   - refine again.

The same tracks and the same bundle adjustment (Levenberg–Marquardt on the Schur complement) are run with four
camera models:

| Model | How a point's pixel is predicted |
|---|---|
| **refractive (ours)** | `project()`: every image; each observation takes its nearest. Analytic Jacobians (`image_jacobians`) |
| **pinhole + AFRT** | the same window, one AFRT start per point (Belden 2013); the analytic Jacobian at the ray it finds |
| **AFRT alone** (`--models afrt_pure`) | one AFRT start per point, its Jacobian by central differences of AFRT (13 runs per observation): no formula of the refraction model |
| **pinhole + distortion** | the window ignored; the intrinsics and OpenCV distortion refined with everything else |

**COLMAP's own pipeline** (pycolmap 4.2, OPENCV camera started at the true focal length) runs on the same images as
it comes, and on a control: the same scene and cameras with the tube's refraction switched off.

**Scoring:**
- **Ground truth:** what each keypoint's ray meets, from the renderer. A track is scored only if all its keypoints
  meet one surface point (within 0.5 mm); 1,461 tracks qualify. The rest hold at least one wrong match.
- **Frames:** points are compared after the similarity that best aligns the camera centres, the usual SfM metric.
  The absolute errors are in `results.json`.

## Results (`demo_b_out.txt`)

![Demo B](summary.png)

| | Points | 3-D error, median | 95th pct. | Camera centres | Rotations | Time |
|---|---:|---:|---:|---:|---:|---:|
| robot prior | | | | 2.1 mm | 0.29° | |
| **refractive (ours)** | 2,222 | **0.24 mm** | 0.93 mm | 0.59 mm | 0.076° | 227 s |
| pinhole + AFRT | 2,220 | 0.24 mm | 0.93 mm | 0.57 mm | 0.073° | 188 s |
| AFRT alone | 2,221 | 0.24 mm | 0.93 mm | 0.58 mm | 0.075° | 841 s\* |
| pinhole + distortion | 2,350 | 35.7 mm | 45.7 mm | 6.3 mm | 0.90° | 41 s |
| COLMAP, through the tube | registers 2 of 60 views | | | | | |
| COLMAP, same scene without the tube | 3,753 | 0.14 mm\* | 1.1 mm\* | 0.13 mm | 0.034° | |

\* AFRT alone ran on 6 worker processes, ours on 8 threads; this PC's times vary by 10–20 % between runs.
For COLMAP: the distance of its points to the true surface (it has no tracks of ours to score). Ours by the same
measure: 0.18 mm.

**What the numbers say:**
- **Through the tube, the refractive model reconstructs the scene about as well as COLMAP does in air.** It does so
  near the wall too: the median error is 0.2–0.3 mm at every distance from the axis.
- **Bundle adjustment through the tube needs a model of the tube.**
  - The pinhole + distortion camera fits the tracks to 1.8 px, but its scene is 36 mm off at the median. Its error
    grows from 21 mm near the axis to 45 mm at the wall.
  - To absorb the tube, it shrinks the scene and inflates its focal length, to fx = 3,737 and fy = 2,764 px from
    1,500, with k2 = 59 and k3 = −2,447.
  - COLMAP, which has no tube model, verifies 652 image pairs but cannot chain more than two views into one
    reconstruction. On the same scene without the tube it reconstructs all 60 views.
- **Pinhole + AFRT does as well as `project()` here.** Almost every point that SIFT matches across views is seen
  once and away from the shadow's edge, which is where a single AFRT start succeeds.
  - **AFRT on its own, with no formula of the refraction model, ties too,** at about four times the time: its
    derivatives come from 13 AFRT runs per observation, where ours are one closed form.
  - Where a point is seen twice, its two images are mirror images of each other, so SIFT does not match them.
  - Where the model has to predict every image — tracking (Demo A), splatting (Demo C) — the difference shows.

**Limits of the demo:**
- **The poses start from robot priors.** The tube fixes the scale and four of the six rigid freedoms, but not its own
  symmetries: rotation about its axis and shift along it. The priors hold those, so absolute errors partly reflect the
  priors.
- **The tube's place is taken as known.**
- **The images are synthetic,** with no noise, blur or brightness change at the interfaces.
