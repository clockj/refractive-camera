"""A published iterative projection method, for comparison with `project()`.

Alternating forward ray tracing (AFRT), as modified by Belden for a gas-solid-liquid wall: L. Belden, "Calibration
of multi-camera systems with refractive interfaces", Exp. Fluids 54, 1463 (2013), Fig. 5.  It iterates on the point
where the ray enters the window and returns at most one ray, from one start; next to a shadow or where a point has
two images it can return nothing, or a ray that does not reach the point, which is why its answer must be checked by
tracing (`trace`).
"""
import math

import numpy as np

TOL = 1e-9          # length units: both pairs of crossings must agree to this
KMAX = 2000         # iterations (Belden states no value)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(a):
    s = math.sqrt(_dot(a, a))
    return (a[0] / s, a[1] / s, a[2] / s)


def _roots(p, d, R, cyl):
    """Both parameters t, in increasing order, where p + t d meets the cylinder r = R (axis z) or the sphere |x| = R;
    None if it misses."""
    if cyl:
        a, b, c = d[0] * d[0] + d[1] * d[1], p[0] * d[0] + p[1] * d[1], p[0] * p[0] + p[1] * p[1] - R * R
    else:
        a, b, c = _dot(d, d), _dot(p, d), _dot(p, p) - R * R
    disc = b * b - a * c
    if a <= 0.0 or disc < 0.0:
        return None
    q = -(b + math.copysign(math.sqrt(disc), b))
    if q == 0.0:
        return (0.0, 0.0)
    t1, t2 = q / a, c / q
    return (t1, t2) if t1 <= t2 else (t2, t1)


def _normal(h, cyl):
    if cyl:
        s = math.hypot(h[0], h[1])
        return (h[0] / s, h[1] / s, 0.0)
    return _unit(h)


def _snell(d, nrm, eta):
    c = -_dot(nrm, d)
    if c < 0.0:
        nrm, c = (-nrm[0], -nrm[1], -nrm[2]), -c
    k = 1.0 - eta * eta * (1.0 - c * c)
    if k < 0.0:
        return None
    f = eta * c - math.sqrt(k)
    return (eta * d[0] + f * nrm[0], eta * d[1] + f * nrm[1], eta * d[2] + f * nrm[2])


def _afrt_local(n, R, cyl, XC, X, tol, kmax):
    """Belden's AFRT in the window frame.  Returns (direction or None, status, iterations)."""
    (R1, R2) = R
    n12, n23, n32 = n[0] / n[1], n[1] / n[2], n[2] / n[1]
    r1 = _unit((X[0] - XC[0], X[1] - XC[1], X[2] - XC[2]))          # k = 1: the straight line to the point
    t = _roots(XC, r1, R1, cyl)
    if t is None or t[0] <= 0.0:
        return None, "breakdown", 0
    Xi1 = (XC[0] + t[0] * r1[0], XC[1] + t[0] * r1[1], XC[2] + t[0] * r1[2])
    Xi1p = None
    for k in range(1, kmax + 1):
        if k > 1:
            # the midpoint of the forward and backward hits on S1, projected radially onto S1
            M = (0.5 * (Xi1[0] + Xi1p[0]), 0.5 * (Xi1[1] + Xi1p[1]), 0.5 * (Xi1[2] + Xi1p[2]))
            if cyl:
                s = R1 / math.hypot(M[0], M[1])
                Xi1 = (M[0] * s, M[1] * s, M[2])
            else:
                s = R1 / math.sqrt(_dot(M, M))
                Xi1 = (M[0] * s, M[1] * s, M[2] * s)
            r1 = _unit((Xi1[0] - XC[0], Xi1[1] - XC[1], Xi1[2] - XC[2]))
        r2 = _snell(r1, _normal(Xi1, cyl), n12)                       # forward: into the wall
        if r2 is None:
            return None, "breakdown", k
        ts = _roots(Xi1, r2, R2, cyl)
        if ts is None or ts[0] <= 0.0:
            return None, "breakdown", k                                # the ray passes through the wall only
        Xi2 = (Xi1[0] + ts[0] * r2[0], Xi1[1] + ts[0] * r2[1], Xi1[2] + ts[0] * r2[2])
        r3 = _snell(r2, _normal(Xi2, cyl), n23)                       # into the liquid
        if r3 is None:
            return None, "breakdown", k
        m3 = (-r3[0], -r3[1], -r3[2])                                  # backward: from the point along -r3
        ts = _roots(X, m3, R2, cyl)
        if ts is None:
            return None, "breakdown", k
        Xi2p = (X[0] + ts[1] * m3[0], X[1] + ts[1] * m3[1], X[2] + ts[1] * m3[2])
        r4 = _snell(m3, _normal(Xi2p, cyl), n32)
        if r4 is None:
            return None, "breakdown", k
        ts = _roots(Xi2p, r4, R1, cyl)
        if ts is None:
            return None, "breakdown", k
        Xi1p = (Xi2p[0] + ts[1] * r4[0], Xi2p[1] + ts[1] * r4[1], Xi2p[2] + ts[1] * r4[2])
        if math.dist(Xi1, Xi1p) <= tol and math.dist(Xi2, Xi2p) <= tol:
            return r1, "converged", k
    return r1, "cap", kmax


def afrt(window, origin, point, tol=TOL, kmax=KMAX):
    """AFRT from `origin` (the camera centre) to `point`, through a window of two interfaces (three media).

    Returns
    -------
    direction : (3,) ndarray or None
        The ray's unit direction at the origin, world frame; None when the method breaks down (a ray misses an
        interface or is totally reflected).  A returned ray is not guaranteed to reach the point: check it with
        `trace`.
    status : str
        "converged", "cap" (kmax iterations without convergence) or "breakdown".
    iterations : int
    """
    if len(window.radii) != 2:
        raise ValueError("AFRT is defined for a window of two interfaces (a wall between two media)")
    cyl = window.kind == "cylinder"
    XC = tuple(float(v) for v in window.to_local(origin))
    X = tuple(float(v) for v in window.to_local(point))
    d, status, it = _afrt_local(window.indices, window.radii, cyl, XC, X, tol, kmax)
    return (None if d is None else window.vectors_to_world(np.array(d))), status, it
