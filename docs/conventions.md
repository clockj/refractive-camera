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

A curved window describes its stack of media as it is, whichever side the camera is on.

| Argument | Meaning |
|---|---|
| `radii` | The N − 1 interface radii, outermost first, strictly decreasing |
| `indices` | The N refractive indices, from the outermost medium inward |
| `axis_point`, `axis_direction` | A point on a cylinder's axis and its direction (`CylindricalWindow`) |
| `center` | The common centre (`SphericalWindow`) |

- **A water-filled acrylic tube in air** is `CylindricalWindow((R_outer, R_inner), (1.0, 1.49, 1.33))`. The same
  object serves a camera outside looking in and a camera in the water looking out.
- **An air-filled housing or a dome port under water** is `(1.33, 1.49, 1.0)`, with the camera in the air inside.
- **The paper's numbering.** For a camera outside, medium 1 (the outermost) holds the camera and medium N the points.

`FlatWindow(normal, offsets, indices, origin)` lists its media from the camera's side. `normal` points from the
camera's side to the points' side, the interfaces' `offsets` along it increase, and `indices` start with the camera's
medium.

**Where the model applies.** `project()` finds which side of a curved window the camera is on.
- **A camera outside** (strictly beyond the outermost interface) images the points strictly inside the innermost one.
- **A camera inside** (strictly inside the innermost interface) images the points strictly outside the outermost one.
- **Through a flat window**, the camera is on the first interface's side and the points beyond the last.
- **Elsewhere:**
  - A camera elsewhere (on an interface, or in a wall) raises `ValueError`.
  - A point elsewhere gets the status `"outside_model"`. That means *not searched*, never *not visible*.

## Images

`project(points, camera, window)` returns one row per image, in an `Images` result, and one status per point.

- **A point can have no image, one or several.** The rows cover:
  - the near branch, which meets the point before its closest approach to the axis or centre;
  - the far branch, which meets it after;
  - from a camera inside: near rays leave the camera moving away from the axis or centre, while far rays pass their
    closest approach first;
  - both senses of rotation about the axis;
  - rays that sweep more than half a turn.
- **Order.** Rows are grouped by point in input order. Within a point their order means nothing.
- **Invariants.** `lam` (Λ = n r dφ/ds, signed by the sense of rotation) and `c_z` (C_z = n dz/ds, zero for spheres)
  are the ray's conserved momenta, as it leaves the camera. They are the same in every medium.
- **A camera inside** sees each outside point a number of times that depends on where it sits:
  - **in its least dense medium** (air in a housing, or behind a dome port under water), exactly once;
  - **in a denser medium** (a camera in a water-filled tube), none, once or twice. Rays that meet the wall too
    obliquely are reflected back (total internal reflection), so the wall hides part of the outside. Next to that
    part, rays bent by different amounts reach the same point twice.
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

## Derivatives

`image_jacobians(points, camera, window, images)` gives two arrays for every row of `images`:
- **`d_point`** (K, 2, 3): d(u, v)/dX, how the pixel moves with its point (world frame);
- **`d_pose`** (K, 2, 6): d(u, v)/d(ω, c).
  - ω turns the camera: `R → expm([ω]×) R`, a rotation in the camera frame.
  - c moves its centre, in the world frame.

**Coverage.** They exist for every window, and for both sides of a curved one. Lens distortion is included.

**Method.** Each is the implicit-function derivative of the projection's own equation, so a row has none where
that equation is singular:
- **on a caustic,** where two images merge (det J = 0), the row is infinite;
- **for rays without a pixel and for a sphere's ring,** it is NaN;
- **for a camera exactly on a cylinder's axis or at a sphere's centre,** the centre's motion off the axis (or the
  centre) is not computed: those columns are NaN.

## Accuracy and its limits

- **Accepted rays.** A ray is accepted when its residual on the dimensionless sweep is below `tol = 1e-9`.
  - Traced back through the window, the rays the tests accept pass within about 1e-15 of the camera's distance from
    the axis or centre.
  - The tests require 1e-9.
- **A camera inside.** Its rays are the rays of the outside problem with the camera and the point exchanged, found to
  the same tolerance and traced to the camera. Traced from the camera, they pass within 1e-10 of their points'
  distance from the axis or centre in the tests.
- **Rounding next to grazing.** Within about 5e-14 of grazing (1 − |Λ|/Λmax below that), rounding alone can lose a
  ray. Its image sits on the very edge of the shadow. This is a property of double-precision arithmetic, not of the
  method.
- **Independent of the closed forms.** The tracer (`trace`, `back_project`, `render_dots`, `search_images`) uses
  only intersections and the vector form of Snell's law, so it gives an independent check of every result.
