# refractive-camera

Camera models that see through curved windows. For a 3-D point behind a cylindrical or spherical window (a
water-filled tube, a pipe, a spherical tank), `project()` finds **every** pixel where the point appears: none when
the wall hides it, one, or several. Flat windows (tank walls, flat ports) are covered too.

![A dot moving inside a water-filled tube, seen by three cameras](examples/demo_a_localization/animation.gif)

> **Status: early development (0.1.0.dev0).** The projection is complete and validated (below); the API may still
> change. Robotics integration (NVIDIA Isaac Sim) is being prepared: [docs/isaac_validation.md](docs/isaac_validation.md).

## Why

A tank, a tube or a dome bends every ray that passes through its wall. A point inside can appear displaced, stretched,
**twice**, or **not at all**, and where it appears depends on the camera's pose as well as on the point. A pinhole
camera with lens distortion cannot represent that, even when it is calibrated through the window.
- **The pinhole camera.** In the demo below it is off by about a millimetre even where it was calibrated, and by 8 mm
  (up to 22 mm) near the wall.
- **The refractive model** is good to a micrometre wherever the target's image is undistorted.

Iterative ray-shooting methods find one image from one starting guess, so a point with two images comes back with
one. When they return nothing, the point may be hidden, or the search may simply have failed (see the benchmark).
`project()` returns the complete list of images for every point, from closed-form ray equations and a certified
search.

## Install

```sh
pip install .            # from a checkout; builds the compiled core if a C++17 compiler is available
```

- **Requirements.** Python 3.10 or later, NumPy and mpmath.
- **Without a compiler** the package installs as pure Python. That is the same search, about 100 times slower.
- **Check.** `refractive_camera.available_backends()` says which backends you have.
- **Optional dependencies.** The examples need `pip install .[demo]` (matplotlib, scipy, Pillow), and the tests
  `pip install .[test]`.

## Quick start

```python
import numpy as np
import refractive_camera as rc

# An acrylic tube (outer radius 100 mm, wall 6 mm) filled with water, seen from air; lengths in metres
tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))
cam = rc.Camera.look_at(position=(0.35, 0.0, 0.0), target=(0.0, 0.0, 0.0), f=2000.0, size=(1280, 1024))

points = np.array([[0.020, 0.030, 0.010],          # an ordinary point
                   [-0.0621, 0.0610, 0.0],         # a point the camera sees twice
                   [0.0038, 0.0869, 0.0]])         # a point hidden by the wall
images = rc.project(points, cam, tube)
print(images.counts())          # [1 2 0]: images per point
print(images.uv)                # one pixel per image; images.point says whose
```

Run `python examples/quickstart.py` for the full example.

`project()` handles any number of coaxial cylinders, concentric spheres or parallel planes
(`CylindricalWindow`, `SphericalWindow`, `FlatWindow`), in any position and orientation. Each result row carries:
- the pixel and whether it falls on the sensor;
- the branch: the ray meets the point before or after its closest approach to the axis;
- the ray's conserved momenta;
- its direction at the camera.

Also in the package:

| | |
|---|---|
| `trace()`, `back_project()` | Vector-Snell ray tracing through the window: from pixels to the line each one sees inside |
| `triangulate()` | A point from pixels in two or more cameras: back-projected lines, then reprojection refinement |
| `render_dots()` | A small ray-traced renderer of dots behind the window, for synthetic data |
| `tracing.search_images()` | A brute-force image search by tracing alone, used to validate `project()` |
| `baselines.afrt()` | A published iterative method (AFRT, Belden 2013), for comparison |
| `isaac` | Conversions between Isaac Sim / USD cameras and `Camera` (no Isaac Sim import) |

Conventions (OpenCV cameras, media numbered from the camera inward) are in [docs/conventions.md](docs/conventions.md).

## Demo: locating a target through a water-filled tube

`python examples/demo_a_localization/demo_a.py` (about 6 minutes; [details](examples/demo_a_localization/README.md))

- **The scene.** Three cameras watch a 1 mm dot inside the reference tube. The dot spirals out to the wall, then
  circles it.
- **The frames** are rendered by tracing rays through the tube, without the projection's equations.
- **Localisation.** The dot is located from the detected blobs in two ways: with the refractive model, and with a
  pinhole camera with lens distortion calibrated through the tube on a dot grid near the axis.

| 3-D error | Inside the pinhole's calibration volume | Outside it |
|---|---:|---:|
| Refractive model (`project()`, `triangulate()`) | 0.001 mm rms | 0.09 mm rms |
| Pinhole + distortion, calibrated through the tube | 0.94 mm rms | 8.1 mm rms (up to 22 mm) |

**The dot hidden or seen twice.** Each camera loses the dot in the wall's shadow in about 30 of the 150 frames, and
sees it twice in 5.
- `project()` predicts both events.
- AFRT returns one of the two images.
- The pinhole camera knows neither.

**Near the wall.** The refractive model's remaining error there comes from the dot's size, not the model. A finite
dot's centroid is not the image of its centre where the magnification changes across it.

![Demo A summary](examples/demo_a_localization/summary.png)

## Validation and speed

The benchmark ([benchmarks/](benchmarks/README.md)) runs at the paper's reference port, with 550 points per window
(cylinder and sphere): uniform, two-image, edge of the shadow, steep and near-axis. Every answer is scored against a
brute-force search that uses tracing alone.

| Points | `project()` | AFRT, one start |
|---|---:|---:|
| Uniform in the water: complete and correct | 100 % | 97–100 % |
| Difficult sets pooled: complete and correct | 100 % | 63 % |
| Two-image points: both images returned | 100 % | 0 % |

**Speed.** The compiled core takes about 0.13 ms per point at the median on one thread. A batch runs at 0.02–0.03 ms
per point on all cores of a desktop CPU. Both were timed on a busy machine; see the benchmark.

**Against AFRT.** On easy points one AFRT start is faster. What it cannot do is return every image, or tell a
hidden point from a failed search.

The checks are listed in [docs/validation.md](docs/validation.md):
- every ray `project()` returns is traced back through the window;
- every image list is compared with a brute-force search that uses tracing alone;
- the two backends are compared on thousands of points;
- results are checked to be the same in any length unit and frame.

## Scope

Supported now:
- A camera **outside** coaxial cylindrical or concentric spherical interfaces, looking at points inside the
  innermost one: tubes, pipes, tanks, flasks.
- A camera on one side of parallel planar interfaces, looking at points on the other: tank walls, flat ports. A flat
  window gives every point exactly one image.
- Any number of interfaces and any positive refractive indices.

Not yet:
- A camera *inside* a dome or cylindrical port, looking out.
- Non-concentric or deforming interfaces.
- Brightness: the model gives where an image is, not how bright it is.

Rounding can lose a ray that grazes an interface to within about 5e-14 of the grazing limit: its image sits on the
very edge of the wall's shadow.

## Citation

The method:

S. Zhong, S. Tan, X. Zhang, Y. Li and R. Ni, "A complete forward projection algorithm through cylindrical and
spherical refractive windows", [arXiv:2610.17413](https://arxiv.org/abs/2610.17413) (2026).

## License and contact

MIT (see [LICENSE](LICENSE)). Shijie Zhong (szhong12@jhu.edu), Ni Research Lab, Johns Hopkins University.
