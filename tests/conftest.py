import numpy as np
import pytest

import refractive_camera as rc

# The paper's reference port: an acrylic tube (outer radius 100 mm, wall 6 mm) filled with water, seen from air.
REF_RADII, REF_INDICES = (0.100, 0.094), (1.0, 1.49, 1.33)

BACKENDS = rc.available_backends()


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture(params=["cylinder", "sphere"])
def kind(request):
    return request.param


def make_window(kind, radii=REF_RADII, indices=REF_INDICES, **frame):
    if kind == "cylinder":
        return rc.CylindricalWindow(radii, indices, **frame)
    return rc.SphericalWindow(radii, indices, **frame)


def reference_camera(distance=0.35, size=(1280, 1024), f=2000.0):
    """The paper's pinhole, at 350 mm from the axis, looking at it."""
    return rc.Camera.look_at((distance, 0.0, 0.0), (0.0, 0.0, 0.0), f=f, size=size)


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def points_inside(window, m, rng, zspan=None):
    """Points uniform in the innermost medium (for a cylinder, within |z| < zspan of the origin's height)."""
    R = window.radii[-1]
    if window.kind == "cylinder":
        zspan = R if zspan is None else zspan
        r = R * np.sqrt(rng.uniform(0, 1, m)) * (1 - 1e-9)
        th = rng.uniform(-np.pi, np.pi, m)
        local = np.c_[r * np.cos(th), r * np.sin(th), rng.uniform(-zspan, zspan, m)]
        return local @ window._Q.T + window.axis_point
    v = rng.normal(size=(m, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return window.center + v * (R * rng.uniform(0, 1, m) ** (1 / 3) * (1 - 1e-9))[:, None]


def camera_outside(window, rng, size=(1280, 1024), f=1800.0):
    """A camera at 1.5-4 outer radii from the axis or centre, looking at a point near it, randomly rolled."""
    R0 = window.radii[0]
    if window.kind == "cylinder":
        th = rng.uniform(-np.pi, np.pi)
        local = np.array([np.cos(th), np.sin(th), 0.0]) * R0 * rng.uniform(1.5, 4.0)
        local[2] = rng.uniform(-R0, R0)
        pos = window._Q @ local + window.axis_point
        target = window.axis_point + rng.normal(scale=0.2 * R0, size=3)
    else:
        v = rng.normal(size=3)
        pos = window.center + v / np.linalg.norm(v) * R0 * rng.uniform(1.5, 4.0)
        target = window.center + rng.normal(scale=0.2 * R0, size=3)
    up = rng.normal(size=3)
    return rc.Camera.look_at(pos, target, up, f=f, size=size)
