"""Demo E's rigs: one camera in three underwater housings, and the truth by tracing alone.

The housing frame: x forward (out through the port), y to the left, z up.  The camera looks along +x with its image
upright.  Each housing is given as its drawing has it (the nominal camera) and as it is built (the true camera: the
lens's entrance pupil a few millimetres off, its axis a degree off, its focal length 1.5 % off the datasheet).
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))                     # run from a checkout without installing
import refractive_camera as rc  # noqa: E402

CACHE = Path(os.environ["REFCAM_CACHE"]) / "demo_e_housing" if "REFCAM_CACHE" in os.environ else HERE / "_cache"
W, H = 1280, 960                         # pixels
F_DATASHEET = 800.0                      # px: 77 degrees across in air
WATER, ACRYLIC, AIR = 1.333, 1.49, 1.0

# the camera's axes in the housing frame, as rows: image right = -y, image down = -z, forward = +x
R_UPRIGHT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def expm(w):
    """Rotation matrix of a rotation vector."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def camera(centre, R=R_UPRIGHT, f=F_DATASHEET, fy=None, principal_point=None):
    """A distortion-free camera with its centre and axes (rows of R) in the housing frame."""
    cx, cy = ((W - 1) / 2.0, (H - 1) / 2.0) if principal_point is None else principal_point
    K = [[f, 0.0, cx], [0.0, f if fy is None else fy, cy], [0.0, 0.0, 1.0]]
    centre = np.asarray(centre, float)
    return rc.Camera(K, R, -R @ centre, None, (W, H))


def housings():
    """name -> dict(window, nominal camera, true camera, a short description)."""
    tilt = expm(np.radians([0.4, 0.8, -0.5]))                  # the lens's axis about a degree off the drawing's
    R_true = R_UPRIGHT @ tilt.T
    built = dict(f=812.0, principal_point=(644.0, 476.5))        # the lens as built: f 1.5 % long, centre 5 px off
    return {
        "flat": dict(
            window=rc.FlatWindow((1.0, 0.0, 0.0), (0.020, 0.030), (AIR, ACRYLIC, WATER)),
            nominal=camera((0.0, 0.0, 0.0)),
            true=camera((-0.0015, 0.0004, -0.0003), R_true, **built),
            label="flat port",
            about="a 10 mm acrylic flat port, 20 mm in front of the lens (the pupil 1.5 mm further back)"),
        "dome": dict(
            window=rc.SphericalWindow((0.050, 0.045), (WATER, ACRYLIC, AIR)),
            nominal=camera((0.0, 0.0, 0.0)),
            true=camera((0.006, 0.001, -0.0005), R_true, **built),
            label="dome port",
            about="a 100 mm acrylic dome (5 mm wall), the lens's pupil 6 mm in front of its centre and 1 mm aside"),
        "tube": dict(
            window=rc.CylindricalWindow((0.050, 0.045), (WATER, ACRYLIC, AIR), axis_direction=(0.0, 0.0, 1.0)),
            nominal=camera((0.015, 0.0, 0.0)),
            true=camera((0.017, 0.001, 0.002), R_true, **built),
            label="tube housing",
            about="a 100 mm acrylic tube (5 mm wall), vertical, the lens 15 mm in front of its axis (17 mm built)"),
    }


def board(nx, ny, spacing):
    """Dot centres of a planar target in its own frame (z = 0), row by row, centred on its middle."""
    g = np.stack(np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2), -1).reshape(-1, 2)
    return np.c_[g * spacing, np.zeros(len(g))]


def shoot(cam, window, X, iters=40, tol=1e-11):
    """The pixels of points X (M, 3), found by tracing alone: Gauss-Newton on each ray's normalised image
    coordinates until its traced path passes within `tol` (m) of the point.

    A camera in air behind any of these housings sees each point in the water once, along one ray, so the solve is
    local; no closed form of the camera model is used.  Returns (uv (M, 2), converged (M,))."""
    X = np.asarray(X, float).reshape(-1, 3)
    Xc = X @ cam.R.T + cam.t
    a = Xc[:, :2] / Xc[:, 2:3]                                     # start: the straight line's pixel
    every = np.arange(len(X))

    def miss(a, rows):
        """The traced rays' offsets from their points (rows of X), perpendicular to the rays; NaN if a ray fails."""
        d = np.c_[a, np.ones(len(a))] @ cam.R                        # world directions
        tr = rc.trace(window, np.broadcast_to(cam.center, d.shape), d)
        v = X[rows] - tr.entry
        s = np.einsum("ij,ij->i", v, tr.direction)
        return v - s[:, None] * tr.direction

    r = miss(a, every)
    h = 1e-7
    for _ in range(iters):
        J = np.stack([(miss(a + [h, 0.0], every) - r) / h, (miss(a + [0.0, h], every) - r) / h], -1)   # (M, 3, 2)
        with np.errstate(invalid="ignore"):
            step = -np.linalg.solve(np.einsum("kia,kib->kab", J, J) + 1e-30 * np.eye(2),
                                    np.einsum("kia,ki->ka", J, r)[..., None])[..., 0]
        cur = np.linalg.norm(r, axis=1)
        lam = np.ones(len(a))
        todo = np.all(np.isfinite(step), 1) & (cur >= tol)
        for _ in range(30):                                           # halve steps that do not reduce the miss
            if not todo.any():
                break
            idx = np.flatnonzero(todo)
            at = a[idx] + lam[idx, None] * step[idx]
            rt = miss(at, idx)
            better = np.linalg.norm(rt, axis=1) < cur[idx]               # False for NaN
            a[idx[better]], r[idx[better]] = at[better], rt[better]
            todo[idx[better]] = False
            lam[todo] *= 0.5
        if np.all(np.linalg.norm(r, axis=1) < tol):
            break
    uv, _ = cam.pixels_from_directions(np.c_[a, np.ones(len(a))])
    return uv, np.linalg.norm(r, axis=1) < tol
