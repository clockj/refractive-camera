# Demo D: a robot arm catches a ball in a water tank, seen through the wall

```sh
python examples/demo_d_arm/demo_d.py --kinds refractive,afrt,pinhole   # 40 ball paths x 3 perceptions, ~15 min
python examples/demo_d_arm/fold_zone.py                                # parts 2-3: tracking next to the wall
python examples/demo_d_arm/precision.py --workers 5                     # part 4: a needle at the ball's centre, ~4 min
python examples/demo_d_arm/video.py --seed 9                           # catch.mp4 and catch.gif
python examples/demo_d_arm/figures.py                                  # summary.png, visibility.png, fold_zone.png
```

It needs `pip install .[robot]` (MuJoCo, imageio). On first run the Franka Panda model is downloaded from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) (Apache-2.0)
into `_cache/`, which is not committed (set `REFCAM_CACHE` to put the caches elsewhere).

![catch](catch.gif)

## The task

- **The tank:** an acrylic cylinder 36 cm across (8 mm wall, n = 1.49), filled with water (n = 1.33) to 30 cm.
- **The ball:** an orange ball 16 mm across swims inside. It starts near the axis and spirals out, on each of 40
  random paths, to its circle:
  - 20 paths stay in the open water, 40–120 mm from the axis;
  - 20 swim out to 136–160 mm, where the wall shows a point twice or hides it (beyond 180 mm / 1.33 = 135 mm).
- **The arm:** a Franka Panda on a 25 cm pedestal holds a **slim long-reach gripper**:
  - a 14 mm shaft 30 cm long, and two thin fingers (14 mm wide, 50 mm long) that open to 40 mm and close along the
    tank's tangent;
  - the Panda's own hand is 208 mm across its closing direction, so it fits inside this tank only around points within
    93 mm of the axis. The slim tool reaches the whole tank.
- **The tank is solid:** its wall, floor and stones are collision geometry, so a touch shows as a contact and counts as a
  failed catch.
- **The catch:** the tool waits in the water above the axis while the cameras pick the ball up. After 4.5 s it moves over
  the ball, goes down to it, and closes. From 30 mm away it stops using the cameras, whose view its own fingers would
  block, and closes on the filter's prediction.
- **A catch counts** if, as the fingers reach the ball, the ball lies between them (within 12 mm along the closing
  direction, 9 mm across it and 17 mm along the fingers), and the robot never touched the tank. The ball's motion is
  scripted (MuJoCo has no buoyancy), and it has no contact with the fingers.

**What the arm knows** comes only from three cameras outside the tank (640 × 512 px, f = 800 px, 0.7 m from its
axis). Their frames are **ray-traced through the curved wall** by `refractive_camera.scene`, with the ball, the tool
and the stones in the water.
- **Detection:** the ball is found in each frame as an orange blob, coarse to fine as in Demo A. The tool can hide it.
- **Tracking:** each perception feeds a constant-velocity Kalman filter at 20 Hz. Its prediction guides the tool.

| Perception | How it locates the ball |
|---|---|
| **refraction-aware (ours)** | `project()` predicts every image of the ball in every camera, so a blob is associated even where a camera sees the ball twice, and no camera waits for a ball it cannot see. The blobs are triangulated through the wall (`triangulate`). |
| **pinhole + AFRT** | the refraction-aware perception, but each camera's predicted image comes from one AFRT start: at most one image, none where AFRT fails. Triangulation from the blobs' traced rays. |
| **pinhole + distortion** | each camera's own OpenCV model, calibrated through the tank on a 3-D target in its middle (r ≤ 80 mm; fits of 1.3 px rms), as a user unaware of refraction would do. Straight-ray triangulation. |

**Calibration.**
- **The two refraction-aware perceptions** (ours and pinhole + AFRT) are given the cameras' and the tank's true
  parameters, as a perfect calibration through the tank would give them. So no calibration error enters them here.
  Demo E calibrates the refractive model from images instead.
- **Pinhole + distortion** has no such parameters to be given. Each camera's model is fitted through the tank to
  3,000 points of its middle, with 0.1 px of noise: its best case.

**Nothing is learned.** The arm is driven by damped least-squares inverse kinematics toward the filter's prediction,
through a scripted sequence (wait, approach, descend, close, lift). The thresholds of the sequence were set by hand.

![what each camera sees](visibility.png)

## Part 1: catches (`demo_d_out.txt`)

| Perception | Caught | Open water | Next to the wall | Estimate's error as the fingers close (median, max) | Touched the tank |
|---|---:|---:|---:|---:|---:|
| **refraction-aware (ours)** | **40 of 40** | 20 of 20 | **20 of 20** | 1.1 mm, 6.9 mm | 0 |
| pinhole + AFRT | 40 of 40 | 20 of 20 | 20 of 20 | 1.4 mm, 8.6 mm | 0 |
| pinhole + distortion | 28 of 40 | 20 of 20 | 8 of 20 | 4.0 mm, 19.4 mm | 0 |

![summary](summary.png)

- **In the open water all three catch every ball.** There the tank is a mild lens, and the pinhole model was
  calibrated there.
- **Next to the wall the pinhole model misses 12 of 20.** Its estimate is off by 12 mm at the median as the fingers
  close, against 2 mm for ours, and its straight rays do not know where the wall shows the ball twice or hides it.
- **Pinhole + AFRT ties with ours,** and that is expected: any image of the ball, traced back through the wall, gives a
  ray through the ball, so a triangulation does not need the second image AFRT misses. Next to the wall the cameras hid
  the ball in 9 % of their frames before the catch and showed it twice in 2 %; one AFRT start found every visible ball
  (checked at 6 radii × 720 azimuths × 3 cameras). Part 3 shows where it falls short.
- **No trial touched the tank.** The tool centre point is kept 10 mm inside the wall, and the open fingers clear it.

## Part 2: tracking where the wall hides or doubles the ball (`fold_zone.py`)

On 20 paths of 16 s the ball swims 140–162 mm from the axis, and the perceptions track it. There is no arm: this
measures what each perception knows. Each camera loses the ball in the wall's shadow in 22 % of its frames, sees it
twice in 4.5 %, and once otherwise; one AFRT start found the visible ball in every camera-frame.

| Perception | 3-D measurement error, median | 95th pct. | Kalman estimate, median | Kalman estimate, max |
|---|---:|---:|---:|---:|
| **ours, whole-ball fit (part 3)** | **0.03 mm** | **0.52 mm** | **0.12 mm** | 30 mm |
| ours, blob centroids | 1.37 mm | 3.76 mm | 1.49 mm | 16.5 mm |
| pinhole + AFRT | 1.42 mm | 3.72 mm | 1.50 mm | 16.5 mm |
| pinhole + distortion | 15 mm | 60 mm | 15 mm | 169 mm |

Every perception measures the ball in the same 95 % of frames. In the other 5 %, at least two cameras lose the ball in
the wall's shadow, nothing can be triangulated, and each Kalman filter coasts on its constant-velocity prediction.

![fold zone](fold_zone.png)

- **From blob centroids, ours and AFRT tie** at about 1.4 mm. Any image of the ball, traced back, gives a ray through
  it; AFRT misses only second images, and one image is enough. Their error grows from 0.4 mm in the open water to
  1.4 mm because a blob's centroid is no longer the image of the ball's centre (below).
- **The pinhole model is off by 15 mm at the median** and up to 17 cm: the ball swims about twice as far from the axis
  as its calibration target reached (80 mm).
- **The filters coast through the gaps.** The longest last 1.5 s, while only one camera sees the ball. On a circle a
  constant-velocity prediction drifts outwards by about ½ r ω² t², up to 30 mm here, and the remaining camera sees
  that drift nearly along its ray. That is the filter, not the perception: the estimate snaps back within a frame of
  the next measurement (0.2 mm for the whole-ball fit). The centroid filters drifted less on these paths by chance:
  their velocity, biased by the stretched blobs, happened to point inwards.

## Part 3: the whole ball, and one camera

Next to the wall a camera sees the ball stretched, clipped at the shadow's edge, or twice, and two copies may merge into
one blob; a blob's centroid is then not the image of the ball's centre. `ball_image.py` predicts the blobs from the
model instead:
- points spread over the ball's surface are projected by `project()`, which gives every image of each;
- an image counts if its ray meets the surface from outside;
- it is weighted by the area it covers in the image, from `image_jacobians`, and by how the detector weighs its colour;
- the ball's centre is fitted so that the predicted blob centroids match the measured ones (Gauss-Newton, with the
  centroids' analytic derivative).

**Results** (`fold_zone_out.txt`): the fit measures the ball to **0.03 mm** at the median next to the wall, 40 times
better than the centroids (1.37 mm for ours, 1.42 mm for AFRT; table above).
- It needs every image of the ball's surface (where the wall shows part of it twice, or clips it) and the Jacobian of
  each. One AFRT start gives one image per surface point and no derivative; with finite differences of its own
  iterations it can make the same fit, at about 14 times the cost (part 4).
- A ray tracer could predict the same blobs by rendering each guess of the ball's position, at the cost of rendering
  frames inside a fit.

**One camera** is then enough where it sees the ball twice: its two rays cross at the ball, 7–22° apart in the water,
so the camera is a stereo pair.
- **Ours** pairs the two blobs with one point and fits it. In the 267 camera-frames of part 2 where a camera showed the
  ball as two blobs, the fit was accepted in 82 %, with an error of **0.50 mm** at the median (90th percentile
  1.7 mm), from one camera. The two rays' crossing alone is 17 mm off: the stretched blobs' centroids are not the
  images of the ball's centre, and two rays from one camera cross at a shallow angle.
- **Pinhole + AFRT:** one AFRT start knows one image, so the camera gives a ray, not a point.
- **Pinhole + distortion:** two straight lines from one centre meet only there.

## Part 4: putting a needle at the ball's centre (`precision.py`)

The ball rests at a point of the tank, and the arm, holding a needle on the slim tool's shaft, must put the needle's
tip at the ball's centre.
- **What the arm sees:** the three tank cameras' frames, taken with the needle parked above the water.
- **How it moves:** it brings the tip onto the perception's estimate and servos it there with its own joint angles,
  to 0.001 mm, so the perception alone decides.
- **Success:** the tip ends within 0.5 mm of the true centre, and the robot never touches the tank.
- **Placements:** 20 in the open water (40–120 mm from the axis) and 20 next to the wall (136–160 mm).

| Perception | How it locates the ball | The window's closed forms |
|---|---|---|
| **ours, whole ball** | part 3's fit: every image of the ball's surface by `project()`, weighted with `image_jacobians` | used |
| AFRT alone, whole ball | the same fit with AFRT only: each surface point's image from one AFRT start, every derivative by finite differences of AFRT (`afrt_ball.py`) | none |
| AFRT, blob centroids | parts 1–2's AFRT perception | none |
| pinhole + distortion | parts 1–2 | none |

**Results** (`precision_out.txt`):

| Perception | Open water: succeed, median error | Next to the wall: succeed, median error | Time per estimate |
|---|---:|---:|---:|
| **ours, whole ball** | **20 of 20**, 0.016 mm | **19 of 20**, 0.027 mm | **0.9 s** |
| AFRT alone, whole ball | 20 of 20, 0.025 mm | 18 of 20, 0.023 mm | 13 s |
| AFRT, blob centroids | 20 of 20, 0.35 mm | 5 of 20, 1.0 mm | 0.005 s |
| pinhole + distortion | 2 of 20, 1.1 mm | 0 of 20, 12 mm | 0.001 s |

![the needle task: success against tolerance](precision.png)

- **Fitting the whole ball is what lands the needle next to the wall.** From blob centroids AFRT succeeds in 5 of 20
  placements there; fitted to the whole ball, in 18.
- **AFRT alone can make that fit, at 14 times the cost.** With finite differences of its own iterations it is about
  as accurate as ours here.
  - It lacks the second images of the parts of the ball the wall shows twice. It leaves that blob out of the fit,
    which costs little in this test.
  - The exception is where the ball's centre itself is seen twice: in one such placement ours lands 0.30 mm from the
    centre and AFRT 0.86 mm.
- **Ours predicts every blob and uses it,** in under a second on one core, with the derivative in closed form. In the
  open water it is also the most precise: every tip within 0.03 mm.
- **The detector.** Part 4 samples its coarse pass every 2 × 2 pixels (parts 1–3: 4 × 4). At 4 × 4 it missed a small
  second copy of the ball (17 px of 430, 53 px from the main blob), exactly a part that AFRT does not predict, and
  that biased ours' fit in one trial.

**Limits:**
- the frames are synthetic, with no noise, blur or reflections;
- the ball's motion is scripted, and it does not touch the fingers;
- the arm moves as if in air.

[Isaac Sim](../../docs/isaac_validation.md), with RTX path-traced frames, is the next step, on RTX hardware.
