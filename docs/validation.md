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
| No image is missed or invented | Image lists compared with `tracing.search_images`, a brute-force search: a 600 × 400 grid of traced rays, refined by Gauss–Newton. Test points include a shadow point (no image) and a two-image point | Same count and same rays (1e-7 rad) at every point |
| Both backends agree | Compiled core against the pure-Python reference | Same image counts; pixels within 1e-6 px |
| Any unit, any frame | The same rig in metres and in millimetres, then moved by a random rigid motion | Same counts; pixels within 1e-6 px |
| The no-refraction limit | Equal indices everywhere give the plain pinhole projection | 1e-7 px |
| Snell's law conserves the invariants | Along traced rays, C_z and Λ (cylinder) or n x × d (sphere) are the same in every medium | 1e-12 |
| Special points | On the axis, opposite the camera (Δφ = π), a sphere's axial ring; inputs outside the model are flagged | — |
| Flat windows | Random stacks and poses give one image per point, traced to it. The solver stays inside its domain up to 30,000 depths to the side | 1e-9; f to 1e-12 |
| Triangulation | Exact pixels give the point, and refinement never increases the reprojection error | 1e-9 m |
| Rendering | A small dot's rendered centroid lies where `project()` puts it | 0.1 px |
| Isaac Sim / USD cameras | Conversions round-trip, and an independently written USD projection matches `Camera` | 1e-9 px |

The suite runs in about 30 s with the compiled core, and about 2.5 min on pure Python.

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
