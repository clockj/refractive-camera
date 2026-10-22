# Validation

How `refractive_camera` is checked, and against what. The principle: nothing is checked only against itself.
- Rays are verified by a tracer that uses none of the projection's equations.
- Image lists are verified by a search that uses only that tracer.
- The two backends are compared with each other, and with the implementation in the paper.

## What the tests check (`pytest`, every commit)

| Check | How | Bound |
|---|---|---|
| Every returned ray is a real ray | Each ray `project()` returns is traced from the camera through the window by vector Snell's law (`trace`), and its distance to the point measured | < 1e-9 of the camera's distance from the axis or centre (typically 1e-16) |
| Pixels lead back to their points | Each image pixel is back-projected through the window (`back_project`) | < 1e-9, same scale |
| No image is missed or invented | Image lists compared with `tracing.search_images`, a brute-force search: a 600 × 400 grid of traced rays, refined by Gauss–Newton (from a camera inside, 720 × 360 directions all round). Test points include a shadow point (no image) and a two-image point | Same count and same rays (1e-7 rad) at every point |
| Both backends agree | Compiled core against the pure-Python reference | Same image counts; pixels within 1e-6 px |
| Any unit, any frame | The same rig in metres and in millimetres, then moved by a random rigid motion | Same counts; pixels within 1e-6 px |
| The no-refraction limit | Equal indices everywhere give the plain pinhole projection | 1e-7 px |
| Snell's law conserves the invariants | Along traced rays, C_z and Λ (cylinder) or n x × d (sphere) are the same in every medium | 1e-12 |
| Special points | On the axis, opposite the camera (Δφ = π), a sphere's axial ring; inputs outside the model are flagged | — |
| A camera inside | Housings, dome ports and submerged cameras (cylinders and spheres). Rays traced from the camera reach their points. A camera in the least dense medium sees each point once. The submerged sphere's shadow and two-image bands and the dome's bending match reference values. Rays leave exactly inside the escape window. Cameras on the axis or at the centre are covered | 1e-10 of the point's distance; 0.002° |
| Derivatives | `image_jacobians` against central differences of `project()`: cylinders, spheres and flat windows, cameras outside and inside, with lens distortion | 1e-6 (relative) |
| Flat windows | Random stacks and poses give one image per point, traced to it. The solver stays inside its domain up to 30,000 depths to the side | 1e-9; f to 1e-12 |
| Triangulation | Exact pixels give the point, and refinement never increases the reprojection error | 1e-9 m |
| Rendering | A small dot's rendered centroid lies where `project()` puts it | 0.1 px |
| Isaac Sim / USD cameras | Conversions round-trip, and an independently written USD projection matches `Camera` | 1e-9 px |

The suite runs in about 35 s with the compiled core, and about 4 min on pure Python.

## Checks made during development

- **The compiled core is OpenLPT's.** It is OpenLPT's curved-window projection (`RefractCurved.cpp`), copied
  unchanged. On 600 points at the paper's reference port (uniform, steep, two-image band, edge of the shadow; cylinder
  and sphere), this package's build and OpenLPT's own build return bit-identical rays.
- **The Python backend is the paper's reference implementation**, copied function by function. It returns
  bit-identical results to the original on the same 600 points.
- **The two backends agree** on 3,844 points. Image counts are identical everywhere, Λ agrees within 1e-14 of its
  range, and C_z within 3e-16. Only the order in which the rays of a point are listed can differ.
- **The derivations the package adds** were checked symbolically with SymPy: the Jacobian of OpenCV's distortion used
  to undistort, the lens's monotone range, the triangulation's normal equations, and the flat window's derivative.
  None failed.
- **The AFRT baseline** (`baselines.afrt`) gives bit-identical status, iterations and rays to the implementation
  used in the paper, on 600 points.
- **The image Jacobians** (`image_jacobians`) give how each image's pixel moves with its point and with the camera,
  for every window and both sides of a curved one.
  - **SymPy:** every partial derivative in their chains was checked step by step: 54 steps for cylinders seen from
    outside, all proved, and 52 for flat windows, spheres, cameras inside and the limit at the axis or centre: 47
    proved, 2 checked on samples only (two inverse-trigonometric identities on a range of angles), 3 recorded as
    not expressible in SymPy and checked numerically (a camera inside sees along the reversed rays of the outside
    problem; a sphere's sweep is a cylinder's at C_z = 0; the expansions' remainders do not reach the derivative).
  - **Against `project()`, cylinders from outside:** compared with central differences of `project()` itself, the
    median relative error is 3e-10 and the largest 1e-6. That is over 457 images away from caustics, seen from four
    camera poses, on both branches.
  - **Against `project()`, every other case:**
    - **The rigs:** flat windows (one with lens distortion), spheres from outside, housings and dome ports,
      submerged cameras, and cameras on the axis or at the centre.
    - **The result:** over 9 rigs and 500 images, the median relative error is 1e-10 to 1e-9 per rig and the
      largest 1e-6.
  - **Against the tracer alone:** inverting the Jacobian of (u, v, s) → X taken by finite differences of the tracer,
    the median error is 1e-9. The worst images, next to the shadow's edge, converge as the step shrinks.
- **Cameras inside** (`project()` from inside a cylinder or a sphere):
  - **Against a brute-force search from the camera.** `search_images` traces rays in every direction from the camera
    and refines them by Gauss–Newton, using no closed form. The stacks were random, of 3–5 media: cylinders and
    spheres, with the camera in the least dense medium or not. On 192 points over 16 stacks, the image lists are
    the same and the rays agree within 3e-13.
  - **Against a separate implementation of the inside case's closed forms.** On 960 points over 24 random cylinder
    stacks, the image lists are the same and the rays agree within 5e-13. Traced from the camera, every ray passes
    within 8e-11 of its point's distance.
  - **A camera in the least dense medium.** Each of 240 points (12 stacks) has one image, equal to a nested bisection
    on the closed forms within 1e-13.
  - **The escape window.** Of 36,000 random rays from 12 cameras, the tracer lets out exactly those inside it.
  - **Reference values:**
    - the submerged sphere's shadow and two-image bands, within 0.01°;
    - the dome port's largest bending: 0.772°, 2.337°, 4.824° and 11.232° for offsets of 5, 15, 30 and 60 mm.
- **The scene renderer** (`refractive_camera.scene`) renders textured objects behind a window.
  - With every index set to 1, its hits equal straight-ray intersections to 1e-12.
  - Every point it hits inside the tube projects back to its own pixel by `project()`, to within 1e-6 px.

## The paper's evidence

The paper (arXiv:2610.17413, Section 9.2.1) compares the same search, in C++, with three published iterative
projection methods on 4,000 points.
- **Ours.** It gave the correct image list, including the empty list at every unseen point, at 3,999 points. The
  exception is a ray within about 1e-14 of grazing, which rounding loses.
- **The iterative methods.** Even restarted 256 times, they returned every image at only 57–96 % of the difficult
  points.

## Benchmark and demo

- [benchmarks/](../benchmarks/README.md): time per point and image-list correctness at the reference port, for both
  backends and AFRT, scored against the brute-force truth.
- [examples/demo_a_localization/](../examples/demo_a_localization/README.md): a target located through a
  water-filled tube from rendered frames, by the refractive model and by a calibrated pinhole camera.

## Known limits

- **Rounding next to grazing.** Rounding can lose a ray within about 5e-14 of grazing an interface (1 − |Λ|/Λmax).
  Its image lies on the very edge of the wall's shadow.
- **What the brute-force search can miss.** `search_images` is a sampled search. It can miss a ray next to a
  caustic, where two images merge, or very close to grazing. It serves as an independent check of `project()`
  away from those places, not as a reference there.
- **Finite targets.** A finite target's image centroid is not the image of its centre, wherever the window's
  magnification varies. Demo A measures this bias for a 1 mm dot near the wall.
