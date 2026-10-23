# Demo C: Gaussian splatting through a water-filled tube

```sh
python examples/demo_c_splatting/demo_c.py              # ours, pinhole, pinhole + distortion: about 1.5 hours on a 4 GB GPU
python examples/demo_c_splatting/demo_c.py --models afrt
python examples/demo_c_splatting/demo_c.py --models afrt_pure   # AFRT alone, no formula of ours: about 4.4 hours
python examples/demo_c_splatting/demo_c.py --models refractive,afrt --footprint hybrid   # the footprint next to caustics
python examples/demo_c_splatting/demo_c.py --models refractive_noisy,refractive_refined   # from a robot's poses
python examples/demo_c_splatting/demo_c.py --no-repair --no-water-mask   # as the demo was first trained
python examples/demo_c_splatting/derivatives.py         # the Jacobian AFRT would need, and the images it misses
python examples/demo_c_splatting/figures.py             # the figures below, and orbit.gif
```

It needs `pip install .[splat]` (PyTorch; CUDA optional). The rasterizer is plain PyTorch, so nothing has to be
compiled. Renders, models and intermediate results go to `_cache/`, which is not committed.

![orbit](orbit.gif)

## Why splatting needs a forward projection, and its derivative

3-D Gaussian splatting draws each Gaussian where its centre projects, with a footprint `J Σ Jᵀ` from the
projection's Jacobian `J`, and its gradients flow back through the same projection. That is what makes it fast: it
never traces a ray.
- **Ray-marching methods** (NeRF) need only back-projection, which through a window is easy tracing.
- **A rasterizer** needs the forward projection, and through a curved window that has more than one answer:
  - near the wall a Gaussian must be drawn **twice**, in the band where the tube shows a point twice;
  - or **not at all**, in the wall's shadow.

  A pinhole camera, with any distortion, has exactly one answer per point.
- **It also needs the derivative** of that projection, for every image of every Gaussian at every step. Ours is the
  implicit-function derivative of the closed form: one 2 × 2 solve on terms the projection has already evaluated.

## The set-up

- **The scene:** the stones of Demo B in the paper's reference tube (acrylic, water).
- **The views:** 48 training views on three rings around the tube, and 12 test views between them; 320 × 256 px.
- **Ground truth:** ray-traced by `refractive_camera.scene`, which contains no Gaussians.
- **The cameras** train the same small 3D Gaussian splatting (`splat.py`). They share the rasterizer, the
  initialisation (10,000 random Gaussians in the water), the schedule (4,000 steps, densification from step 300 to
  2,500) and the losses (L1 and SSIM).

| Camera | Where a Gaussian is drawn |
|---|---|
| **refractive (ours)** | at every image `project()` returns. The footprint comes from the analytic Jacobian of that image, and the depth order from the ray's length. |
| AFRT alone | each Gaussian's image from one AFRT start at every step, its footprint from central differences of AFRT: no formula of ours (section below) |
| AFRT's images + our Jacobian | ours, but each Gaussian's images come from one AFRT start: at most one image, none where AFRT fails. Our Newton tracking and Jacobian do the rest, and its repair runs AFRT again, so it differs from ours only in which images it knows. |
| pinhole | the true camera, the window ignored |
| pinhole + distortion | each camera's own OpenCV model, fitted through the tube to points of the scene volume: the best a pinhole can do |

**How ours trains:**
- The images of each Gaussian are found by `project()` (complete) for each view, every 500 steps.
- In between, each image's momenta (Λ, C_z) follow its Gaussian by Newton's method from their last value. A Gaussian
  whose image Newton's method loses is searched again at once (`--repair`, on by default), so ours draws in training
  the images it is scored on.
- Each view's pixels whose central ray reaches the water are traced once (`splat.water_mask`). A Gaussian in the water
  lights only those pixels (`--water-mask`, on by default), so a model shows exactly the background outside the water.
  Both options apply to every camera that models the tube.
- Gradients flow through one Newton step taken with autograd. That is the implicit-function derivative of the
  projection; it was checked against the analytic Jacobian to 1e-14.
- New Gaussians from densification start from their parents' images.
- Splats next to a caustic, where the Jacobian blows up, are left out and counted.
- With `--footprint hybrid`, an image whose linear footprint is wider than 6 px gets the unscented footprint instead
  (see "The streaks next to the caustics"). The results below use the linear footprint.

## Results (`results.json`; `demo_c_out.txt` logs every run, the latest last)

Two sets of test views:
- **between the training views:** 12 views on the training rings;
- **from new poses:** 12 views closer and higher, or farther and lower, than any training view. This is where a robot
  that moves its camera would look from.

PSNR is given over the whole image, and over the pixels where the camera sees a stone within 15 mm of the wall.

| Camera | Between the training views | ... near the wall | From new poses | ... near the wall | SSIM (between, new) | Opaque Gaussians to the true surface | Training |
|---|---:|---:|---:|---:|---:|---:|---:|
| **refractive (ours)** | **33.2 dB** | **24.9 dB** | 30.1 dB | **23.0 dB** | 0.985, 0.971 | 0.87 mm median (61 % within 1 mm) | 36 min |
| AFRT alone | 33.1 dB | 24.6 dB | 30.1 dB | 22.5 dB | 0.985, 0.970 | 0.88 mm (61 %) | 4.4 h |
| AFRT's images + our Jacobian | 33.1 dB | 24.5 dB | 29.9 dB | 22.6 dB | 0.985, 0.969 | 0.86 mm (62 %) | 44 min |
| pinhole | 23.3 dB | 17.5 dB | 21.4 dB | 15.9 dB | 0.899, 0.824 | 3.9 mm (18 %) | 29 min |
| pinhole + distortion | 24.0 dB | 15.8 dB | 22.2 dB | 14.5 dB | 0.904, 0.860 | 2.1 mm (18 %) | 29 min |

Every camera trains with the demo's defaults: 4,000 steps, the linear footprint, and the repair and the water mask
for the cameras that model the tube. The pinholes do not model it, so neither option applies to them. The times are
on a GTX 1630 (4 GB); ours ends with 20,997 Gaussians.

![metrics](metrics.png)

**What the numbers say:**
- **The refractive camera is 9–10 dB ahead of the pinholes between the training views, and 8–9 dB from new
  poses.** About 1.5 dB of that lead comes from the two options, which a pinhole cannot have. Ours' Gaussians sit on
  the stones: 61 % of the opaque ones are within 1 mm of the true surface.
- **The pinhole cameras learn a different scene,** one that looks right from close to the training views. From new
  poses it falls further behind, and its Gaussians lie 2–4 mm from the stones at the median.
  - Per-camera distortion, fitted through the tube to 2.8 px rms, helps a little overall.
  - It does worse next to the wall, where a polynomial cannot follow the tube.
- **The three cameras that model the tube end close together over the whole image. Ours is ahead next to the
  wall,** where it draws the second images:
  - **AFRT alone,** which uses no formula of ours, is level with ours over the whole image (0.12 and 0.01 dB
    behind) and 0.4 dB behind next to the wall. It trains 7.4 times as long (section below).
  - **AFRT's images, given our Jacobian and our repair,** are 0.14 and 0.17 dB behind over the whole image, and
    0.4–0.5 dB behind next to the wall.
  - **Training ours again with another seed** moved a single view by up to 3.7 dB in the development campaign. It
    moved the average over the whole image by 0.1–0.3 dB, but next to the wall by only 0.06–0.12 dB, a third or less
    of ours' lead there.
  - All three put their Gaussians equally close to the stones: 0.86–0.88 mm at the median.

Trained with the hybrid footprint (`--footprint hybrid`), ours scored 33.2 (24.7) / 29.8 (22.7) dB and the AFRT
model 32.8 (24.3) / 29.6 (22.4) dB. Each pair is between the training views (near the wall) / from new poses
(near the wall). With the repair and the mask the unscented footprint no longer helps (see "The streaks next to the
caustics").

## Take the tube away

Every image above is a view through the tube, distortion and all. The Gaussians are not: ours sit where the stones
are. Drawn by a plain camera with the tube taken away, the same trained models compare with the stones ray-traced
with no tube (`without_tube.json`, 3 test views):

| Model, trained through the tube | Drawn without the tube |
|---|---:|
| **refractive (ours)** | **32.3 dB** |
| pinhole + distortion | 21.8 dB |
| pinhole | 19.1 dB |

![without the tube](without_tube.png)

Ours shows the stones as they are, nearly as sharp as its views through the tube. The pinhole models built the tube's
distortion into the stones themselves. One reconstruction gives a robot both: what its camera will see through the
wall, and the scene behind it.

## The images AFRT misses, and the derivative it lacks (`derivatives.py`)

Measured on ours' Gaussians, trained with the demo's defaults, in the 12 test views, on the same GPU, with a batched
PyTorch AFRT (`afrt_torch.py`, the library's AFRT to 2e-16):

| | |
|---|---|
| **Pixels that show an image AFRT does not find** | 1,280 of 188,774 stone pixels, in 10 of 12 views |
| PSNR there | ours 15.7 dB, AFRT's images 12.7 dB (both 26.7 dB on the other stone pixels) |
| **The Jacobian `d(uv)/dX`**, which AFRT does not return | central differences of AFRT, or autograd through its iterations: both within ~1e-7 of ours at the median (1e-5 to 5e-3 next to a caustic) |
| Time per view, 20,997 Gaussians | **ours, a training step: 53 ms** · AFRT, pixels only: 0.74 s · AFRT + differences: 5.3 s · AFRT + autograd: 24.4 s |

AFRT cannot tell a hidden point from a slow one, so a batch runs to its iteration cap (300).

Every image AFRT converged to is one of ours (232,190 of 232,228 Gaussian-views), except those of 5 Gaussians that
ended up in training just past the water, in the acrylic wall. `project()` reports these as outside its model, whose
points are in the water, and the splat camera draws them nowhere. AFRT does not check where a point is, and returns a
ray for them.

![derivatives](derivatives.png)

## The streaks next to the caustics

As first trained, with neither of the demo's options, ours' renders showed short streaks next to the tube's caustics,
many of them reaching past the water's edge. With the water mask no light falls outside the water any more (the left
crop above). The streaks are not the double image. They come from the footprint:
- `J Σ Jᵀ` assumes the projection is linear across the Gaussian.
- Next to a caustic, where a point's two images merge and then vanish into the shadow, the projection is a square-root
  fold. `J` grows without bound, while the true image of a Gaussian stays short and ends at the caustic.
- The linear footprint is long and symmetric, so it smears across the caustic.

The double images lie in the same band next to the caustic, so the two go together. In the models first trained,
60–66 % of the footprints wider than 16 px belong to Gaussians a view shows twice, against 3 % of all images.

Rendering ours' own Gaussians, trained with the linear footprint and neither option, four ways (development
campaign, T18) shows the trade:

| ours' linear-trained Gaussians, rendered with | between views | new poses | next to the wall (between / new) | streak pixels (between / new) |
|---|---:|---:|---:|---:|
| every image, footprints capped at 48 px (as trained) | 31.74 | 28.69 | 24.66 / 22.46 | 24.47 / 20.15 |
| footprints capped at 16 px | 31.83 | 28.75 | 24.64 / 22.49 | 24.91 / 20.47 |
| one image per Gaussian, AFRT's | 31.86 | 28.89 | 24.35 / 22.26 | 25.12 / 20.82 |
| the unscented footprint where the linear one is wider than 6 px (`splat.render_ut`) | 31.86 | 28.79 | 24.63 / 22.46 | 24.94 / 20.74 |

- **Dropping the second images helps the whole image,** because some carry streaks, and **hurts next to the wall,**
  where they are real content. That is why a model that never draws them can come out ahead on some views.
- **The unscented footprint** projects six sigma points of the Gaussian on the image's own branch, so the points
  beyond the caustic have no image and the footprint stays on the side where the Gaussian is seen. It needs every
  image and each one's branch. Used only where the linear footprint is wider than 6 px, it removes the long streaks
  at render time, without retraining.

**Training with it** (`--footprint hybrid`): ours and the AFRT model were retrained, still with neither option, with
the unscented footprint wherever the linear one is wider than 6 px. Each model is drawn with the footprint it was
trained with, and all four are scored on the same streak pixels: those where the linear-trained model of ours draws
footprints wider than 16 px (development campaign, T21).

| Model, footprint | between views | new poses | next to the wall (between / new) | streak pixels (between / new) |
|---|---:|---:|---:|---:|
| ours, linear | 31.74 | 28.69 | 24.66 / 22.46 | 24.47 / 20.15 |
| **ours, hybrid** | **31.90** | 28.83 | 24.59 / **22.53** | **25.19** / 20.99 |
| AFRT's images + our Jacobian, linear | 31.66 | 28.80 | 24.34 / 22.40 | 24.45 / 20.49 |
| AFRT's images + our Jacobian, hybrid | 31.82 | **28.92** | 24.31 / 22.45 | 25.10 / **21.18** |

- **Training with the unscented footprint helps both models the same way:** 0.7–0.8 dB on the streak pixels and
  0.1–0.2 dB over the whole image.
- **Ours and the AFRT model still tie over the whole image.** Ours keeps its lead next to the wall between the views
  (0.3 dB), where the second images are.

**With the demo's options, the unscented footprint no longer helps:**

| Trained with the repair and the mask | Between the training views (near the wall) | From new poses (near the wall) |
|---|---|---|
| **ours, linear (the demo's default)** | **33.2 (24.9) dB** | **30.1 (23.0) dB** |
| ours, hybrid | 33.2 (24.7) dB | 29.8 (22.7) dB |
| AFRT's images + our Jacobian, linear | 33.1 (24.5) dB | 29.9 (22.6) dB |
| AFRT's images + our Jacobian, hybrid | 32.8 (24.3) dB | 29.6 (22.4) dB |

The results above therefore use the linear footprint, and `--footprint hybrid` stays an option.

## Training with AFRT alone (`--models afrt_pure`)

The AFRT model above is given our Jacobian and our Newton tracking. `splat.AFRTPureSplatCamera` uses neither: at
every step each Gaussian's image comes from one AFRT start and its footprint from central differences of AFRT, so no
formula of the refraction model enters. It trains on the same terms as ours:
- **both draw in training the images they are scored on:** AFRT alone runs AFRT for every Gaussian at every step, and
  ours searches again at once any image Newton's method loses;
- **both light only the pixels whose ray reaches the water;**
- **both use the linear footprint:** the unscented one needs each image's branch, which AFRT does not know.

| Linear footprint | Between the training views (near the wall) | From new poses (near the wall) | SSIM (between, new) | Opaque Gaussians to the true surface | Training |
|---|---|---|---|---|---:|
| **ours** | **33.2 (24.9) dB** | **30.1 (23.0) dB** | 0.985 / 0.971 | **0.87 mm** (61 % within 1 mm) | **36 min** |
| AFRT alone | 33.1 (24.6) dB | 30.1 (22.5) dB | 0.985 / 0.970 | 0.88 mm (61 %) | 4.4 h |

- **Next to the wall ours is ahead by 0.4 dB,** between the training views and from new poses. There it draws the
  second images that AFRT never finds. Retrained with other seeds, ours moved by only 0.06–0.12 dB there.
- **Over the whole image the two are level.** Between the training views ours is 0.12 dB ahead on average, but one
  view gives it 1.4 dB and it is ahead in only 5 of the 12. From new poses it is 0.01 dB ahead, in 8 of the 12 views.
  Two training runs that differ only in their seed differ by up to 3.7 dB in a single view, and by 0.1–0.3 dB on
  average.
- **Ours trains 7.4 times as fast:** 0.53 s per step against 3.9 s.

**The first comparison, and why it differed.** The demo first trained ours with neither option
(`--no-repair --no-water-mask`). Trained that way, AFRT alone led by 0.6 dB over the whole image and trailed by
0.3–0.4 dB next to the wall, at 7.4 times the time. The development campaign found two causes, both in how the demo
trained ours:
- **Ours did not draw in training every image it draws when scored.** The demo searches a view's images every 500
  steps and follows them by two Newton steps in between. An image whose two steps did not converge was skipped until
  the view's next search. A probe that searched the trained view afresh found 11 % of the images undrawn on average,
  and up to 58 % in one view.
- **The second images' linear footprints spill past the water's edge.** A Gaussian seen twice has its second image at
  the water's edge in the view, where its footprint is long. The part beyond the edge lights pixels whose rays never
  reach the water. AFRT alone never finds those images, so it put half as much stray light there.

The repair removes the first cause: at each step, the Gaussians whose image Newton's method lost are searched again at
once, by `project()`, while each view is still searched in full every `--refresh` steps. No image is skipped any more.
The only images a fresh search adds are second images gained since the view's last search (0.2 %). The water mask
removes the second cause.

| Ours, linear footprint, seed 0 | Between the training views (near the wall) | From new poses (near the wall) | Opaque Gaussians to the true surface | Training |
|---|---|---|---|---:|
| neither option, as first trained (`--no-repair --no-water-mask`) | 31.7 (24.7) dB | 28.7 (22.5) dB | 0.99 mm | 37 min |
| the repair only (`--no-water-mask`) | 32.1 (24.4) dB | 29.0 (22.4) dB | 0.88 mm | 38 min |
| every view searched at every step instead (`--refresh 1 --no-repair --no-water-mask`) | 32.0 (24.5) dB | 28.8 (22.4) dB | 0.89 mm | 75 min |
| the same, with the mask (`--refresh 1 --no-repair`) | 33.3 (24.8) dB | 30.1 (22.9) dB | 0.87 mm | 65 min |
| **both options: the defaults** | **33.2 (24.9) dB** | **30.1 (23.0) dB** | **0.87 mm** | **36 min** |

- The repair scores like searching every step, at the demo's own training time.
- The mask gains ours 1.1 dB and AFRT alone 0.8 dB, since ours drew the second images that spilled. It also helps
  inside the water: the Gaussians no longer offset the spill.

## Training from a robot's poses (`--models refractive_noisy,refractive_refined`)

A robot knows where its camera is only to a millimetre or two. Here the 48 training poses are off by 1.8 mm and 0.29°
(rms), as Demo B's prior.
- `refractive_noisy` trains from those poses as they are;
- `refractive_refined` refines them with the Gaussians (`splat.PoseSplatCamera`). The gradient in the camera's pose
  comes from the same implicit-function step: the camera's position enters the closed form through r_P, φ_P and z_P,
  and its rotation turns the ray into a pixel. It equals `image_jacobians`' pose derivative to 1e-13. AFRT would need
  twelve more runs per Gaussian, per view, per step for it.

Both train with the demo's defaults. Each camera's water mask comes from its pose as reported, and is not updated as
the pose is refined.

| Training poses | Between the training views (near the wall) | From new poses (near the wall) | SSIM (between, new) | Opaque Gaussians to the true surface |
|---|---|---|---|---|
| true (ours with the demo's defaults) | 33.2 (24.9) dB | 30.1 (23.0) dB | 0.985 / 0.971 | 0.87 mm |
| a robot's, as they are | 25.5 (17.7) dB | 24.4 (17.3) dB | 0.914 / 0.886 | 1.38 mm |
| **a robot's, refined with the Gaussians** | **31.4 (23.3) dB** | **29.1 (21.8) dB** | 0.978 / 0.962 | **1.00 mm** |

- **Refining the poses wins back 5.9 of the 7.8 dB** that a robot's poses cost between the views, and 4.7 of the
  5.6 dB from new poses. It brings the Gaussians to within 1.00 mm of the stones at the median.
- **The poses themselves move from 1.80 mm and 0.29° to 1.39 mm and 0.21°** (rms). The test views are rendered from
  their true poses, so the scores above measure the Gaussians where they truly are.

![test views](test_views.png)

## Checks

In the development campaign:
- **The splat camera against `project()`:** the same 264 images, pixels within 3e-12 px, Jacobians within 5e-14.
- **Gradients:** its autograd gradient equals the analytic Jacobian (1e-14); its pose gradient equals
  `image_jacobians` (1e-13) and central differences of `project()` (2e-8).
- **Warm starts:** Newton from the previous momenta agrees with a fresh `project()` after a 0.2 mm move (4e-12 px).
- **The rasterizer** passes `gradcheck`, and matches a brute-force per-pixel sum (1e-16).
- **A footprint against Monte-Carlo:** a 1 mm Gaussian's drawn mean and covariance agree with 20,000 samples
  projected by `project()`. The mean is within 0.02 px and the covariance within about 1 %.
- **The batched AFRT** equals the library's AFRT: same convergence on 2,000 point-camera pairs, directions within
  2.2e-16.
- **AFRT alone's camera** calls no formula of the refraction model. Where it has ours' image, its pixels agree with
  ours within 3e-6 px, and its Jacobians within 8e-8 at the median.
- **The repair:** with it, a probe that searched each trained view afresh found no image skipped. The AFRT model's
  repair gives each Gaussian exactly the image a fresh AFRT start finds, never a second one. The refined camera's
  repair equals a fresh search at its current pose (3e-12 px).
- **The water mask** equals the ground truth's record of which pixels' central rays reach the water, in all 24 test
  views. Drawn with it, a model shows exactly the background outside it.
- **Both options off** (`--no-repair --no-water-mask`), every model trained before they existed draws its test views as
  before, bit for bit. AFRT alone's model was trained by `demo_c.train` with the demo's settings in the development
  campaign; `--eval-only` scores it exactly as the campaign did.

**Limits:**
- the images are synthetic and view-independent (no reflections, no Fresnel);
- the linear footprint's Jacobian is held constant in the backward pass;
- the splats are clamped next to caustics.
