# Conventions

What every quantity in `refractive_camera` means. Where a choice could be made two ways, it is written down here.

## Units and frames

- **Lengths.** Any one unit, used throughout: metres, millimetres, anything.
  - The search divides every length by a power of two near the camera's distance from the axis (or centre). Its
    tolerances are therefore dimensionless, and a rig gives the same pixels in metres and in millimetres. The
    tests check this.
  - Division by a power of two is exact, so the problem solved is the one given, to the last bit.
- **World frame.** Right-handed, and otherwise free. Windows can sit anywhere, in any orientation.

## Camera

`Camera(K, R, t, dist, size)` follows OpenCV.

| Quantity | Convention |
|---|---|
| Camera frame | x right, y down, z forward (the optical axis) |
| Pose | `X_cam = R @ X_world + t`; the centre (pinhole) is `-R.T @ t` |
| Intrinsics | `K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]`, in pixels |
| Pixels | (u, v): u to the right, v down; the centre of the top-left pixel is (0, 0) |
| On the sensor | `-0.5 <= u < W - 0.5` and `-0.5 <= v < H - 0.5` |
| Distortion | OpenCV's (k1, k2, p1, p2, k3) on the normalised coordinates (x/z, y/z) |

**Distortion describes the lens, never the window.** The window's refraction is computed from its geometry and
indices. A distortion term that also tried to model the window would count the refraction twice.

A direction outside the range where the radial polynomial still increases gets no pixel. Beyond that range the
polynomial folds back, and a direction outside the lens's field could otherwise land inside the image.

`Camera.look_at(position, target, up, f=..., size=...)` builds a camera whose optical axis passes through `target`,
with `up` pointing up in the image. Its principal point is the image centre, `((W - 1)/2, (H - 1)/2)`.

## Windows

The media are numbered from the camera inward, as in the paper: medium 1 holds the camera and medium N the points.

| Argument | Meaning |
|---|---|
| `radii` | The N − 1 interface radii, outermost first, strictly decreasing |
| `indices` | The N refractive indices, from the camera's medium inward |
| `axis_point`, `axis_direction` | A point on a cylinder's axis and its direction (`CylindricalWindow`) |
| `center` | The common centre (`SphericalWindow`) |

A water-filled acrylic tube seen from air is `CylindricalWindow((R_outer, R_inner), (1.0, 1.49, 1.33))`.

**Where the model applies.** The camera must be strictly outside the outermost interface. A point is projected only
when it lies strictly inside the innermost one.
- A camera elsewhere raises `ValueError`.
- A point elsewhere gets the status `"outside_model"`. That means *not searched*, never *not visible*.

## Images

`project(points, camera, window)` returns one row per image, in an `Images` result, and one status per point.

- **A point can have no image, one or several.** The rows cover:
  - the near branch, which meets the point before its closest approach to the axis or centre;
  - the far branch, which meets it after;
  - both senses of rotation about the axis;
  - rays that sweep more than half a turn.
- **Order.** Rows are grouped by point in input order. Within a point their order means nothing.
- **Invariants.** `lam` (Λ = n r dφ/ds, signed by the sense of rotation) and `c_z` (C_z = n dz/ds, zero for spheres)
  are the ray's conserved momenta. They are the same in every medium.
- **Direction.** `direction` is the ray's unit direction at the camera centre, in the world frame.
- **Pixels.**
  - A ray that leaves the camera backwards gets `in_front = False` and no pixel (NaN).
  - So does a ray beyond the lens model's range.
  - `in_image` also requires the pixel to fall on the sensor.
- **Rings.** For a sphere, a point on the line through the centre and the camera can be imaged as a whole circle (an
  axial ring). Such a row has `branch == "ring"` and no direction or pixel.
- **Status per point.**
  - `"ok"`: every image was searched for.
  - `"outside_model"`: see above.
  - `"capped"`: the far-branch search reached its box limit and is not certified complete. The search has not hit
    this limit at any point tested so far.

## Accuracy and its limits

- **Accepted rays.** A ray is accepted when its residual on the dimensionless sweep is below `tol = 1e-9`.
  - Traced back through the window, the rays the tests accept pass within about 1e-15 of the camera's distance from
    the axis or centre.
  - The tests require 1e-9.
- **Rounding next to grazing.** Within about 5e-14 of grazing (1 − |Λ|/Λmax below that), rounding alone can lose a
  ray. Its image sits on the very edge of the shadow. This is a property of double-precision arithmetic, not of the
  method.
- **Independent of the closed forms.** The tracer (`trace`, `back_project`, `render_dots`, `search_images`) uses
  only intersections and the vector form of Snell's law, so it gives an independent check of every result.
