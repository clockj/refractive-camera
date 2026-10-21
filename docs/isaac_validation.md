# Validating Isaac Sim's refraction against `refractive_camera`

**Status: protocol, not results.** Nothing below has been run yet. It needs an RTX GPU (see Requirements).

## Why

Isaac Sim renders refraction through glass and water with its RTX path tracer. `refractive_camera.project()`
predicts exactly where a point behind a cylindrical or spherical window appears: every image, or none. Before
rendered frames are used as observations, for calibration, reconstruction or tracking through a tube, this protocol
measures how far the renderer's images are from the exact geometry. It also shows where the difference comes from:
camera conventions, sampling, tessellation, or how the scene represents the glass–water contact.

## Requirements

- **GPU.** An NVIDIA RTX GPU with RT cores and at least 16 GB of VRAM. The minimum listed is an RTX 4080, and
  GPUs without RT cores are not supported
  ([requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)). Every
  rendering mode is a mode of the RTX renderer
  ([rendering modes](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/reference_material/rendering_modes.html)).
- **Operating system.** Windows 11 or Ubuntu 22.04/24.04.
- **Cloud.** Alternatives are an L40S instance on Brev or AWS (g6e) or an L4 on GCP
  ([cloud](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_cloud.html),
  [Brev](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_advanced_cloud_setup_brev.html),
  [AWS](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_advanced_cloud_setup_aws.html)). The
  container runs Python apps headless
  ([container](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_container.html)).
- **Version pin.** Isaac Sim 6.2.0 on Python 3.12. Install the matching PyTorch first, then
  `pip install isaacsim[all,extscache]==6.2.0.0 --extra-index-url https://pypi.nvidia.com`
  ([pip install](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html)). Record the
  driver, the GPU and every setting below with each dataset.
- **License.** On first use Isaac Sim asks you to accept the NVIDIA Omniverse License Agreement. That is your
  decision as the user; `OMNI_KIT_ACCEPT_EULA=YES` accepts it non-interactively. The Isaac Sim repository is
  Apache-2.0, and its Kit SDK and assets carry NVIDIA's additional terms
  ([license](https://github.com/isaac-sim/IsaacSim/blob/main/LICENSE)).

## Renderer settings for geometry

Render for positions, not looks. Use the documented defaults as starting points and raise them until the results
stop changing. Confirm the setting names in the installed version.

- **Mode.** Interactive path tracing (`PathTracing`), not the real-time modes. NVIDIA warns that RTX Real-Time 2.0
  results may diverge from path tracing, and its transmission bounces default to 3
  ([real-time](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html)).
- **Samples.** `/rtx/pathtracing/totalSpp` well above its default of 512, with the OptiX denoiser off (it is on by
  default). No DLSS or other upscaling, and no temporal accumulation across a moving scene
  ([path tracing](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html)).
- **Bounces.** Raise `/rtx/pathtracing/maxSpecularAndTransmissionBounces` (default 6) and `maxBounces` (default 4).
  A dot inside a water-filled tube takes 2 transmissions to reach. The background seen through the whole tube
  takes 4, more with internal reflections.
- **Camera.** A pinhole: `fStop = 0` (no depth of field), no motion blur and no lens distortion until the
  distortion step of the ladder.

## Cameras: Isaac Sim to `refractive_camera`

`refractive_camera.isaac` converts without importing Isaac Sim:
- `camera_from_usd(focalLength, horizontalAperture, verticalAperture, width, height, world_matrix, offsets)` builds
  a camera from a USD camera prim and its world transform (row-vector convention);
- `camera_from_pose(position, quaternion_wxyz, axes, K, size)` builds one from a pose in Isaac Sim's camera axes
  ("ros", "usd" or "world"; see
  [conventions](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/reference_material/reference_conventions.html));
- `usd_from_camera()` and `pose_from_camera()` go back the other way;
- `opencv_pinhole_properties()` gives the arguments of the camera's OpenCV lens model.

A sketch, inside Isaac Sim, with names from USD's Python bindings:

```python
import numpy as np
from pxr import Usd, UsdGeom
import refractive_camera as rc
from refractive_camera import isaac

prim = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
camera = isaac.camera_from_usd(
    prim.GetFocalLengthAttr().Get(), prim.GetHorizontalApertureAttr().Get(), prim.GetVerticalApertureAttr().Get(),
    width, height, np.array(prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
    prim.GetHorizontalApertureOffsetAttr().Get(), prim.GetVerticalApertureOffsetAttr().Get())
tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33),
                            axis_point=(0.0, 0.0, 0.0), axis_direction=(0.0, 0.0, 1.0))
expected = rc.project(dot_centres, camera, tube)        # every image of every dot, or none
```

## The scene ladder

Each step adds one difficulty, and each must pass before the next.

1. **Air only.** Dots in air, with no window, compared with `Camera.project_points()`. This checks the camera
   conversion: the half-pixel convention, the principal point, fx and fy, and then the OpenCV distortion.
2. **A flat slab.** An acrylic plate between the camera and dots in air, compared with
   `FlatWindow(normal, (d, d + thickness), (1.0, 1.49, 1.0))`. This checks refraction at planar surfaces and the
   bounce settings.
3. **A solid acrylic cylinder** (n = 1.49) with small opaque dots embedded in it, compared with
   `CylindricalWindow((R,), (1.0, 1.49))`. This is one curved interface. It checks the tessellation of the cylinder,
   and whether USD's analytic `Cylinder` is rendered as a mesh.
4. **The water-filled tube** (outer radius 100 mm, wall 6 mm, acrylic 1.49, water 1.33), compared with
   `CylindricalWindow((0.100, 0.094), (1.0, 1.49, 1.33))`. Represent the acrylic–water contact in both ways and
   compare each with the library:
   - **(a) touching volumes:** an acrylic shell (IOR 1.49) and a water cylinder (IOR 1.33) whose surfaces coincide.
     **Unverified:** NVIDIA documents no handling of nested or coincident dielectrics. A renderer that does not
     track nested media treats the contact as acrylic → air → water. Where the ray is inclined enough this reflects
     totally, though acrylic → water would transmit it.
   - **(b) one interface with the relative index:** the outer surface carries IOR 1.49 (air to acrylic), and the
     inner surface alone carries 1.33/1.49 = 0.893 (acrylic to water), with no separate water body. This is the
     usual way to render a liquid in a glass with a path tracer that does not track nested media.

## Measurement

- **Targets.** Small spheres (radius 0.5–1 mm) at known positions, bright on black: emissive spheres, so that the
  measurement does not depend on how the tube lights them. Spread them over the tube, and include positions where
  `project()` predicts two images or none (the double-image band and the shadow next to the wall).
- **Centroids.** Threshold each rendered frame and take connected components. The intensity-weighted centroid of
  each blob is the measurement.
- **Matching.** Match each blob to the nearest image `project()` predicts for the dot centre. Count missing images
  (predicted, not rendered) and extra blobs (rendered, not predicted).
- **Finite-size bias.** A finite dot's image centroid is not the image of its centre, where the window's
  magnification varies. Render the same dots with `refractive_camera.render_dots()`, which is exact geometry with
  the same radius. Then compare Isaac Sim's centroids with those as well as with `project()`. The difference between
  the two references is the finite-size bias, and the rest is the renderer's.

## Checks

- **Half-pixel.** A dot on the optical axis in air lands at ((W - 1)/2, (H - 1)/2) in OpenCV's pixel convention,
  which `refractive_camera.isaac` assumes, or at (W/2, H/2). Settle this first: every later number depends on it.
- **Intrinsics honoured.** Render with fx ≠ fy and with an off-centre principal point (aperture offsets, or the
  OpenCV lens properties). One forum report found the renderer using (fx + fy)/2 and ignoring cx, cy unless a
  distortion model was enabled
  ([forum](https://forums.developer.nvidia.com/t/new-camera-class-from-omni-isaac-sensor-api-intrinsic-paramaters-bug/274376)).
- **Convergence.** Double totalSpp, the transmission bounces and the cylinder's tessellation, for example 64 to 4096
  segments, until the centroids move by less than a tenth of the discrepancy being reported.
- **Images.** In the double-image band both blobs must be there. In the shadow, nothing may be. Near grazing,
  facets and the bounce limit are the likely culprits.
- **Annotators.** Segmentation rays pass through translucent geometry without registering a hit
  ([issue 700](https://github.com/isaac-sim/IsaacSim/issues/700)). A transparent mesh's Cast Shadows flag decides
  whether it appears in segmentation
  ([transparency](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotations_with_transparency.html)).
  So masks and boxes of objects behind the window are not refracted: measure from the RGB frames.

## Deliverables

- A version-pinned scene script that builds each step of the ladder from a JSON rig: radii, indices, camera poses
  and intrinsics, and dot positions.
- A capture script that writes, for every frame, the RGB image, the camera's USD attributes and world matrix, the
  renderer settings, and the Isaac Sim, driver and GPU versions.
- The dataset: frames plus the ground-truth JSON.
- A short report: the discrepancy in pixels (median, p95, max) by region (axis to wall; single image, double image,
  edge of the shadow); its convergence with samples, bounces and tessellation; the missing and extra images; and
  both representations of the acrylic–water contact.

## Acceptance

- The discrepancy is characterised and converged, and it is well below the detection noise of the task the frames
  will serve (for example under 0.1 px where detection is good to about 0.5 px).
- Every missing or extra image is explained.
- Until then, rendered frames are pictures, not measurements.

## Unverified

- How Isaac Sim treats coincident or nested dielectrics (step 4a).
- The half-pixel convention, and whether the renderer honours fx ≠ fy and cx, cy.
- Whether USD's analytic `Cylinder` and `Sphere` prims are rendered exactly or tessellated.
- The setting names quoted above in the installed release.
