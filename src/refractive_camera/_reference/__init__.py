"""The pure-Python backend: batch calls over the reference implementation.

Both functions take the window's own frame (the axis along z through the origin for cylinders, the centre at the
origin for spheres) and return one row per image, in the layout of the compiled backend:

    point   (K,)    index of the point, into `points`
    branch  (K,)    0 near, 1 far (for a ring, as the compiled backend reports it: see `ring`)
    lam     (K,)    Lambda, signed
    c_z     (K,)    C_z (0 for spheres)
    dir     (K, 3)  unit direction of the ray at the pinhole (NaN for a ring)
    ring    (K,)    a sphere's axial ring: a cone of rays, not one
    capped  (M,)    the far-branch search hit its box cap (never, here)

The caller checks that the model applies (pinhole outside the outermost interface, point inside the innermost).
"""
import numpy as np

from . import cylinder, sphere


def _rows(per_point, m):
    point, branch, lam, c_z, dirs, ring = [], [], [], [], [], []
    for i, rays in enumerate(per_point):
        for b, L, C, d in rays:
            point.append(i)
            branch.append(1 if b == "far" else 0)
            lam.append(L)
            c_z.append(C)
            dirs.append(np.full(3, np.nan) if d is None else np.asarray(d, float))
            ring.append(b == "ring")
    return {
        "point": np.asarray(point, np.int64),
        "branch": np.asarray(branch, np.int8),
        "lam": np.asarray(lam, float),
        "c_z": np.asarray(c_z, float),
        "dir": np.asarray(dirs, float).reshape(-1, 3),
        "ring": np.asarray(ring, bool),
        "capped": np.zeros(m, bool),
    }


def project_cylinder(n, R, pinhole, points, tol=1e-9, threads=1):
    """Every image of each point through coaxial cylinders, by cylinder.project().  `threads` is ignored."""
    n, R = [float(v) for v in n], [float(v) for v in R]
    P0 = np.asarray(pinhole, float)
    pts = np.asarray(points, float).reshape(-1, 3)
    return _rows(([(b, L, C, d) for b, L, C, d in cylinder.project(n, R, P0, X, tol)] for X in pts), len(pts))


def project_cylinder_pairs(n, R, pinholes, points, tol=1e-9, threads=1):
    """project_cylinder for pairs (pinhole i, point i).  `threads` is ignored."""
    n, R = [float(v) for v in n], [float(v) for v in R]
    C, pts = np.asarray(pinholes, float).reshape(-1, 3), np.asarray(points, float).reshape(-1, 3)
    return _rows((list(cylinder.project(n, R, c, X, tol)) for c, X in zip(C, pts)), len(pts))


def project_sphere_pairs(n, R, pinholes, points, tol=1e-9, threads=1):
    """project_sphere for pairs (pinhole i, point i).  `threads` is ignored."""
    n, R = [float(v) for v in n], [float(v) for v in R]
    C, pts = np.asarray(pinholes, float).reshape(-1, 3), np.asarray(points, float).reshape(-1, 3)
    O = np.zeros(3)
    return _rows(([(b, L, 0.0, d) for b, L, d in sphere.project(n, R, O, c, X, tol)] for c, X in zip(C, pts)),
                 len(pts))


def project_sphere(n, R, pinhole, points, tol=1e-9, threads=1):
    """Every image of each point through concentric spheres centred at the origin, by sphere.project().  `threads`
    is ignored."""
    n, R = [float(v) for v in n], [float(v) for v in R]
    P0 = np.asarray(pinhole, float)
    pts = np.asarray(points, float).reshape(-1, 3)
    O = np.zeros(3)
    return _rows(([(b, L, 0.0, d) for b, L, d in sphere.project(n, R, O, P0, X, tol)] for X in pts), len(pts))
