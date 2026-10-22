"""Quick start, from inside: a camera in air behind a dome port, and a camera in a water-filled tube looking out.

    python examples/quickstart_inside.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # run from a checkout without installing
import refractive_camera as rc  # noqa: E402

# 1. A dome port under water: an acrylic sphere (outer radius 100 mm, wall 6 mm) with air inside.  A window lists its
#    media from the outermost inward, whichever side the camera is on: water, acrylic, air.  Lengths in metres.
dome = rc.SphericalWindow(radii=(0.100, 0.094), indices=(1.33, 1.49, 1.0))

# The camera sits in the air 20 mm behind the dome's centre (a slightly misaligned port), looking out along +z.
cam = rc.Camera.look_at(position=(0.0, 0.0, -0.020), target=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0),
                        f=600.0, size=(1280, 1024))
fish = np.array([[0.00, 0.00, 1.00],                      # straight ahead
                 [0.30, 0.10, 0.80],                      # off to the side
                 [-0.50, 0.30, 0.50]])                    # near the edge of the view
images = rc.project(fish, cam, dome)
print("Dome port, camera 20 mm behind the centre")
print(f"    images per point: {images.counts()}  (a camera in air sees every point in the water exactly once)")
pinhole, _ = cam.project_points(fish)                     # where a pinhole camera that ignores the dome puts them
for i in range(len(fish)):
    k = images.rows_of(i)[0]
    print(f"    point {np.round(fish[i], 2)} m: pixel ({images.uv[k, 0]:7.1f}, {images.uv[k, 1]:7.1f}), "
          f"{np.linalg.norm(images.uv[k] - pinhole[i]):5.1f} px from the pinhole's")

# At the dome's centre every ray meets the dome at right angles: nothing bends, and the pinhole is exact.
centred = rc.Camera.look_at(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0),
                            f=600.0, size=(1280, 1024))
gap = np.linalg.norm(rc.project(fish, centred, dome).uv - centred.project_points(fish)[0], axis=1).max()
print(f"    the same camera at the centre: {gap:.1e} px from the pinhole's")

# 2. A camera in a water-filled acrylic tube in air, 85 mm from the axis, looking out across the tube.  The same
#    tube seen from outside would be the same object: media from the outermost inward, air, acrylic, water.
tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))
diver = rc.Camera.look_at(position=(0.085, 0.0, 0.0), target=(-0.15, 0.25, 0.0), f=500.0, size=(1280, 1024))
angles = np.radians([140.0, 125.0, 96.0])                 # about the axis, from the camera's side
targets = 0.200 * np.c_[np.cos(angles), np.sin(angles), np.zeros(3)]
images = rc.project(targets, diver, tube)
print("\nCamera in a water-filled tube, 85 mm from the axis, looking out")
for i, a in enumerate(np.degrees(angles)):
    rows = images.rows_of(i)
    pixels = ", ".join(f"({images.uv[k, 0]:.1f}, {images.uv[k, 1]:.1f})" for k in rows)
    print(f"    a point 200 mm from the axis at {a:5.1f} deg: {len(rows)} image(s) {pixels}")
print("    From the water, rays that meet the wall too obliquely are reflected back, so the wall hides part of the")
print("    outside; next to that part, rays bent by different amounts reach the same point twice.")

# Which pixels see out at all?  Trace every 8th pixel's ray outward; total internal reflection stops the others.
u, v = np.meshgrid(np.arange(0, 1280, 8), np.arange(0, 1024, 8))
tr = rc.back_project(diver, tube, np.c_[u.ravel(), v.ravel()])
print(f"    {100 * tr.ok.mean():.0f} % of this camera's pixels see out of the tube; the wall reflects the others back")
