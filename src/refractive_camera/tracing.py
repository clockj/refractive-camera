"""Ray tracing through the window by vector Snell's law.

Nothing here uses the paper's closed forms: rays are intersected with each interface and refracted there, so the
tracer is an independent check of `project()` (the tests trace every ray it returns), and the way back from pixels
to the scene (`back_project()`, triangulation, rendering).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trace:
    """Rays traced from outside the window into its innermost medium, where they run straight.

    Attributes
    ----------
    ok : (L,) bool
        The ray reached the innermost medium.
    entry : (L, 3) float
        Where it entered the innermost medium (world frame); NaN unless ok.
    direction : (L, 3) float
        Its unit direction there (world frame); NaN unless ok.
    length : (L,) float
        The length of its straight path through the innermost medium, from `entry` to where it reaches the innermost
        interface again (infinite for a flat window).  Every point the ray reaches lies on ``entry + s * direction`` with 0 <= s <= length.
    hits : (L, N - 1, 3) float
        Its crossing of each interface, outermost first (world frame); NaN from the first failure on.
    segments : (L, N, 3) float
        Its unit direction in each medium, from the camera's inward (world frame); NaN from the first failure on.
    reason : (L,) str
        "" when ok; "miss" when it passes outside an interface (or starts inside one); "tir" when an interface
        reflects it totally.
    """

    ok: np.ndarray
    entry: np.ndarray
    direction: np.ndarray
    length: np.ndarray
    hits: np.ndarray
    segments: np.ndarray
    reason: np.ndarray

    def miss_distance(self, points):
        """Distance from each point (L, 3) to its ray's straight path in the innermost medium; inf where the ray did
        not get there or the point lies outside the path's extent."""
        v = np.asarray(points, float) - self.entry
        s = np.einsum("ij,ij->i", v, self.direction)
        dist = np.linalg.norm(v - s[:, None] * self.direction, axis=1)
        span = 1e-12 * (1.0 + np.nan_to_num(self.length, posinf=0.0))
        good = self.ok & (s >= -span) & (s <= self.length + span)
        return np.where(good, dist, np.inf)


def _refract(d, nrm, eta):
    """Vector Snell's law for unit directions d (L, 3) at unit normals nrm (L, 3), eta = n_from / n_to; the
    refracted directions and whether each is transmitted (False: total internal reflection)."""
    c = -np.einsum("ij,ij->i", nrm, d)
    nrm = np.where((c < 0)[:, None], -nrm, nrm)
    c = np.abs(c)
    k = 1.0 - eta * eta * (1.0 - c * c)
    out = eta * d + (eta * c - np.sqrt(np.maximum(k, 0.0)))[:, None] * nrm
    return out / np.linalg.norm(out, axis=1, keepdims=True), k >= 0


def _first_hit(p, d, R, cylinder):
    """The nearest positive parameter t with |p + t d| = R (transverse part only for a cylinder), from outside;
    NaN if the line misses or the circle lies behind."""
    if cylinder:
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1]
        c = p[:, 0] ** 2 + p[:, 1] ** 2 - R * R
    else:
        a = np.einsum("ij,ij->i", d, d)
        b = np.einsum("ij,ij->i", p, d)
        c = np.einsum("ij,ij->i", p, p) - R * R
    disc = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -(b + np.copysign(np.sqrt(np.maximum(disc, 0.0)), b))   # the roots q/a and c/q, without cancellation
        t1, t2 = q / a, c / q
    lo = np.minimum(t1, t2)
    good = (a > 0) & (disc >= 0) & (c > 0) & (lo > 0)
    return np.where(good, lo, np.nan)


def trace(window, origins, directions):
    """Trace rays from points outside the window through every interface into its innermost medium.

    Parameters
    ----------
    window : CylindricalWindow, SphericalWindow or FlatWindow
    origins, directions : (L, 3) or (3,) array_like
        World points outside the window (beyond its outermost interface, or on the camera's side of a flat one), and
        the rays' directions (any length).

    Returns
    -------
    Trace
    """
    if window.kind == "flat":                          # planes: the last medium is unbounded
        from .flat import trace_flat
        p, d = np.broadcast_arrays(np.atleast_2d(np.asarray(origins, float)),
                                   np.atleast_2d(np.asarray(directions, float)))
        ok, reason, hits, segs, entry, d = trace_flat(window, p, d / np.linalg.norm(d, axis=1, keepdims=True))
        return Trace(ok=ok, entry=entry, direction=d, length=np.where(ok, np.inf, np.nan), hits=hits,
                     segments=segs, reason=reason)
    p = window.to_local(np.atleast_2d(np.asarray(origins, float)))
    d = window.vectors_to_local(np.atleast_2d(np.asarray(directions, float)))
    p, d = np.broadcast_arrays(p, d)
    p, d = p.copy(), d / np.linalg.norm(d, axis=1, keepdims=True)
    L, N = len(p), len(window.indices)
    cyl = window.kind == "cylinder"
    ok = np.ones(L, bool)
    reason = np.full(L, "", "<U4")
    hits = np.full((L, N - 1, 3), np.nan)
    segs = np.full((L, N, 3), np.nan)
    segs[:, 0] = d
    for k, Rk in enumerate(window.radii):
        t = _first_hit(p, d, Rk, cyl)
        miss = ok & np.isnan(t)
        reason[miss], ok = "miss", ok & ~miss
        p = np.where(ok[:, None], p + np.nan_to_num(t)[:, None] * d, np.nan)
        nrm = p * np.array([1.0, 1.0, 0.0]) if cyl else p.copy()
        with np.errstate(invalid="ignore"):
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        d_new, through = _refract(d, np.nan_to_num(nrm), window.indices[k] / window.indices[k + 1])
        tir = ok & ~through
        reason[tir], ok = "tir", ok & ~tir
        d = np.where(ok[:, None], d_new, np.nan)
        hits[:, k] = p
        segs[:, k + 1] = d
    # Inside the innermost interface the ray runs straight until it reaches that interface again: the other root.
    if cyl:
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1]
    else:
        a = np.einsum("ij,ij->i", d, d)
        b = np.einsum("ij,ij->i", p, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        length = np.where(ok, np.maximum(-2.0 * b / a, 0.0), np.nan)
    return Trace(ok=ok, entry=_to_world(window, p), direction=window.vectors_to_world(d), length=length,
                 hits=_to_world(window, hits), segments=window.vectors_to_world(segs), reason=reason)


def _to_world(window, x):
    """Window-frame points (..., 3) in the world frame."""
    if window.kind == "cylinder":
        return window.vectors_to_world(x) + window.axis_point
    return np.asarray(x, float) + window.center


def back_project(camera, window, uv):
    """The rays that pixels (L, 2) see: traced from the camera through the window into its innermost medium.

    Every point imaged at a pixel lies on that pixel's ray, ``entry + s * direction`` with 0 <= s <= length, of
    the returned Trace.
    """
    origins, directions = camera.rays(np.atleast_2d(np.asarray(uv, float)))
    return trace(window, origins, directions)


def search_images(window, origin, point, grid=(400, 400), tol=1e-12, iters=40):
    """Rays from `origin` that reach `point`, found by tracing alone: a brute-force reference for validation.

    Directions on a grid over the window's extent are traced; every grid cell whose ray passes near the point
    starts a Gauss-Newton refinement of the ray's two angles, and the rays that converge are kept once each.  No
    closed form of the paper is used, so this is an independent check of `project()`.  It is slow, and like any
    sampled search it can miss a ray where two images merge (next to a caustic) or graze an interface: use
    `project()` for real work.

    Parameters
    ----------
    window : CylindricalWindow or SphericalWindow
    origin, point : (3,) array_like
        The camera centre (outside the window) and a point inside its innermost interface, in world coordinates.
    grid : (int, int)
        Samples across and along the window's extent.
    tol : float
        Accepted distance between the refined ray and the point, relative to the origin's distance from the axis
        or centre.

    Returns
    -------
    (K, 3) ndarray of unit world directions at the origin.
    """
    origin, point = np.asarray(origin, float), np.asarray(point, float)
    o = window.to_local(origin)
    cyl = window.kind == "cylinder"
    # A frame at the origin: e1 from the axis (or centre) to the origin, e3 along the axis.
    e1 = o * np.array([1.0, 1.0, 0.0]) if cyl else o.copy()
    rP = np.linalg.norm(e1)
    e1 /= rP
    if cyl:
        e3 = np.array([0.0, 0.0, 1.0])
    else:
        h = np.eye(3)[int(np.argmin(np.abs(e1)))]
        e3 = h - (h @ e1) * e1
        e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    s_max = window.radii[0] / rP                      # the outermost interface subtends |sin(alpha)| < s_max

    def directions(a, b):
        """Unit local directions for sin(alpha) = a s_max (across) and sin(beta) = b (along)."""
        sa, sb = a * s_max, b
        ca, cb = np.sqrt(np.maximum(1 - sa * sa, 0)), np.sqrt(np.maximum(1 - sb * sb, 0))
        return -(ca * cb)[:, None] * e1 + (sa * cb)[:, None] * e2 + sb[:, None] * e3

    def residual(a, b):
        tr = trace(window, np.repeat(origin[None], len(a), 0), window.vectors_to_world(directions(a, b)))
        v = point - tr.entry
        s = np.einsum("ij,ij->i", v, tr.direction)
        r = v - s[:, None] * tr.direction
        bad = ~tr.ok | (s < 0) | (s > tr.length)
        r[bad] = np.nan
        return r

    ga, gb = grid
    a = np.linspace(-1, 1, ga + 2)[1:-1]
    b = np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, gb + 2)[1:-1])       # denser near the horizontal
    A, B = (m.ravel() for m in np.meshgrid(a, b, indexing="ij"))
    miss = np.linalg.norm(residual(A, B), axis=1).reshape(ga, gb)
    # Starts: the cells that are local minima of the miss over their 3 x 3 neighbourhood
    padded = np.pad(np.nan_to_num(miss, nan=np.inf), 1, constant_values=np.inf)
    neigh = np.min([padded[1 + i:1 + i + ga, 1 + j:1 + j + gb] for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i, j) != (0, 0)], 0)
    start = np.isfinite(miss) & (miss <= neigh)
    x = np.stack([A.reshape(ga, gb)[start], B.reshape(ga, gb)[start]], 1)
    lo, hi = np.full(2, -1 + 1e-15), np.full(2, 1 - 1e-15)
    r = residual(x[:, 0], x[:, 1])
    for _ in range(iters):
        # One-sided differences, taken away from the nearer end of the range so that they stay inside it
        h = np.where(x > 0, -1e-7, 1e-7)
        J = np.stack([(residual(x[:, 0] + h[:, 0], x[:, 1]) - r) / h[:, :1],
                      (residual(x[:, 0], x[:, 1] + h[:, 1]) - r) / h[:, 1:]], -1)       # (K, 3, 2)
        JtJ = np.einsum("kia,kib->kab", J, J)
        Jtr = np.einsum("kia,ki->ka", J, r)
        with np.errstate(invalid="ignore"):
            step = -np.linalg.solve(JtJ + 1e-30 * np.eye(2), Jtr[..., None])[..., 0]
        # Backtracking: halve a step that leaves the rays reaching the point, or does not reduce the miss
        cur = np.linalg.norm(r, axis=1)
        lam = np.ones(len(x))
        todo = np.all(np.isfinite(step), 1) & np.isfinite(cur)
        for _ in range(40):
            if not todo.any():
                break
            xt = np.clip(x[todo] + lam[todo, None] * step[todo], lo, hi)
            rt = residual(xt[:, 0], xt[:, 1])
            better = np.linalg.norm(rt, axis=1) < cur[todo]          # False for NaN
            idx = np.flatnonzero(todo)
            x[idx[better]], r[idx[better]] = xt[better], rt[better]
            todo[idx[better]] = False
            lam[todo] *= 0.5
        if np.all(np.linalg.norm(r, axis=1)[np.isfinite(cur)] <= tol * rP):
            break
    r = np.linalg.norm(r, axis=1)
    keep = x[np.isfinite(r) & (r <= tol * rP)]
    found = []
    for d in directions(keep[:, 0], keep[:, 1]):
        if not any(np.linalg.norm(d - f) < 1e-9 for f in found):
            found.append(d)
    return window.vectors_to_world(np.array(found).reshape(-1, 3))
