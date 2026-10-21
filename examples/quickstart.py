"""Quick start: where does a point inside a water-filled tube appear in a camera outside it?

    python examples/quickstart.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # run from a checkout without installing
import refractive_camera as rc  # noqa: E402

# An acrylic tube (outer radius 100 mm, wall 6 mm) filled with water, seen from air; lengths in metres.
tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))

# A camera 350 mm from the tube's axis, looking at it (OpenCV conventions: f and the image size in pixels).
cam = rc.Camera.look_at(position=(0.35, 0.0, 0.0), target=(0.0, 0.0, 0.0), f=2000.0, size=(1280, 1024))

points = np.array([
    [0.020, 0.030, 0.010],                                    # an ordinary point: one image
    [0.087 * np.cos(np.radians(135.5)), 0.087 * np.sin(np.radians(135.5)), 0.0],   # seen twice
    [0.087 * np.cos(np.radians(87.5)), 0.087 * np.sin(np.radians(87.5)), 0.0],     # hidden by the wall
])
images = rc.project(points, cam, tube)

for i, X in enumerate(points):
    rows = images.rows_of(i)
    print(f"point {np.round(X * 1000, 1)} mm: {len(rows)} image(s)")
    for k in rows:
        print(f"    pixel ({images.uv[k, 0]:8.2f}, {images.uv[k, 1]:8.2f})   {images.branch[k]} branch")

# Every image is a real ray: trace it back through the tube with Snell's law and it passes through its point.
tr = rc.back_project(cam, tube, images.uv)
print(f"largest miss of the back-traced rays: {tr.miss_distance(points[images.point]).max():.1e} m")
