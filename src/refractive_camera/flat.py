"""Flat windows: parallel planar interfaces, such as a tank wall or a flat camera port.

A ray through parallel planes keeps Lambda = n sin(theta), the component of n d along the planes, so its lateral
travel in medium i is L_i = H_i Lambda / sqrt(n_i^2 - Lambda^2), with H_i the medium's thickness along the normal.
Forward projection solves

    f(Lambda) = sum_i H_i Lambda / sqrt(n_i^2 - Lambda^2) = L_tot,

with L_tot the lateral offset from the camera to the point (Gu et al., Opt. Express 31, 31383 (2023), restated in this
invariant in the paper's Section "Flat windows in invariant form").  f(0) = 0, f increases strictly and tends to
infinity at the smallest index, so every point beyond the window has exactly one image.  f is also convex, so a
Newton step from Lambda = 0 overshoots, possibly past the domain's edge.  Newton is therefore kept inside a bracket
that shrinks at every step, and falls back to bisection; and the unknown is the gap n_min - Lambda to that edge,
which keeps oblique rays accurate.
"""
import numpy as np


class FlatWindow:
    """Parallel planar interfaces: a tank wall, an aquarium, a flat port.

    Parameters
    ----------
    normal : array_like
        The interfaces' normal, pointing from the camera's side to the points' side (any length).
    offsets : sequence of float
        The N - 1 interfaces' signed distances along `normal` from `origin`, strictly increasing: the camera
        meets the first one first.
    indices : sequence of float
        The N refractive indices, from the camera's medium onward.
    origin : array_like
        The point the offsets are measured from.

    A camera in air behind a 10 mm acrylic wall of a water tank, the wall's inner face 50 mm in front of the
    origin along +z, is ``FlatWindow((0, 0, 1), (0.040, 0.050), (1.0, 1.49, 1.33))``.
    """

    kind = "flat"

    def __init__(self, normal, offsets, indices, origin=(0.0, 0.0, 0.0)):
        nrm = np.asarray(normal, float)
        if nrm.shape != (3,) or not np.linalg.norm(nrm) > 0:
            raise ValueError("normal must be a non-zero 3-vector")
        offsets = np.atleast_1d(np.asarray(offsets, float))
        indices = np.atleast_1d(np.asarray(indices, float))
        if offsets.ndim != 1 or offsets.size < 1:
            raise ValueError("a window needs at least one interface")
        if indices.size != offsets.size + 1:
            raise ValueError("one index per medium: len(indices) must be len(offsets) + 1")
        if not (np.all(np.isfinite(offsets)) and np.all(np.isfinite(indices))) or np.any(indices <= 0):
            raise ValueError("offsets must be finite and indices finite and positive")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("offsets must strictly increase along the normal")
        self.normal = nrm / np.linalg.norm(nrm)
        self.offsets = tuple(float(v) for v in offsets)
        self.indices = tuple(float(v) for v in indices)
        self.origin = np.array(origin, float).reshape(3)

    def __repr__(self):
        return (f"FlatWindow(normal={self.normal.tolist()}, offsets={self.offsets}, indices={self.indices}, "
                f"origin={self.origin.tolist()})")

    def height(self, points):
        """Signed distance of points (..., 3) from `origin` along the normal."""
        return (np.asarray(points, float) - self.origin) @ self.normal

    def outside(self, points):
        """Whether points lie strictly on the camera's side of the first interface, where the camera must be."""
        return self.height(points) < self.offsets[0]

    def inside(self, points):
        """Whether points lie strictly beyond the last interface, where the model projects them."""
        return self.height(points) > self.offsets[-1]


def solve_gap(H, n, L_tot, iters=200):
    """The root of f(Lambda) = sum_i H_i Lambda / sqrt(n_i^2 - Lambda^2) = L_tot for each row, as the gap
    g = n_min - Lambda to the domain's edge, with n_min the smallest index among the media crossed (H_i > 0).

    Solving for g rather than Lambda keeps oblique rays accurate: next to grazing, Lambda is within a few ulp of
    n_min, but g and every n_i^2 - Lambda^2 = (n_i - n_min + g)(n_i + n_min - g) keep their full relative precision.
    f - L_tot decreases in g, from +infinity as g -> 0 to -L_tot at g = n_min (Lambda = 0).  Newton, kept inside a
    bracket that shrinks at every step, with bisection when a step would leave it.

    H : (M, N) thicknesses (>= 0); n : (N,) indices; L_tot : (M,) lateral offsets (>= 0).  Returns (g, n_min).
    """
    H, n, L_tot = np.asarray(H, float), np.asarray(n, float), np.asarray(L_tot, float)
    active = H > 0
    n_min = np.min(np.where(active, n[None, :], np.inf), axis=1)
    dn = np.where(active, n[None, :] - n_min[:, None], 1.0)          # >= 0 on the media crossed

    def F(g):
        lam = n_min - g
        root = np.sqrt((dn + g[:, None]) * (n[None, :] + lam[:, None]))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.sum(np.where(active, H * lam[:, None] / root, 0.0), axis=1) - L_tot
            fp = np.sum(np.where(active, H * n[None, :] ** 2 / root ** 3, 0.0), axis=1)     # df/dLambda
        return f, fp

    g = n_min.copy()                                   # Lambda = 0, where f = -L_tot
    lo, hi = np.zeros_like(g), n_min.copy()            # the root lies in (lo, hi]: f - L_tot > 0 at lo+, <= 0 at hi
    for _ in range(iters):
        f, fp = F(g)
        lo = np.where(f > 0, g, lo)
        hi = np.where(f < 0, g, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = g + f / fp                          # Newton in g: dg = -dLambda
        inside = np.isfinite(step) & (step > lo) & (step < hi)
        new = np.where(inside, step, 0.5 * (lo + hi))
        done = (f == 0) | (np.abs(new - g) <= 4 * np.finfo(float).eps * np.abs(new))
        g = np.where(f == 0, g, new)
        if np.all(done):
            break
    return g, n_min


def project_flat(window, centre, points):
    """Every image of each point (one per point) through a flat window, in the row layout of the backends.

    `centre` (the camera centre) must lie on the camera's side and the points beyond the last interface; the caller
    checks both.  Directions are in the world frame.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    m = len(pts)
    nrm, n = window.normal, np.array(window.indices)
    hc, hx = window.height(centre), window.height(pts)
    off = np.array(window.offsets)
    H = np.empty((m, len(n)))
    H[:, 0] = off[0] - hc
    H[:, 1:-1] = np.diff(off)[None, :]
    H[:, -1] = hx - off[-1]
    w = pts - centre
    w = w - np.outer(w @ nrm, nrm)                         # the lateral offset, along the planes
    L_tot = np.linalg.norm(w, axis=1)
    with np.errstate(invalid="ignore"):
        e = np.where(L_tot[:, None] > 0, w / L_tot[:, None], 0.0)
    g, n_min = solve_gap(H, n, L_tot)
    lam = n_min - g
    normal_part = np.sqrt((n[0] - n_min + g) * (n[0] + lam))       # n_1 cos(theta_1), from the gap
    d = (normal_part[:, None] * nrm + lam[:, None] * e) / n[0]
    return {
        "point": np.arange(m, dtype=np.int64),
        "branch": np.zeros(m, np.int8),
        "lam": lam,
        "c_z": np.zeros(m),
        "dir": d,
        "ring": np.zeros(m, bool),
        "capped": np.zeros(m, bool),
    }


def trace_flat(window, p, d):
    """Vector-Snell tracing through the planes, for `tracing.trace`: (ok, reason, hits, segments, entry, direction)
    with world-frame arrays, p and d of shape (L, 3), d unit."""
    from .tracing import _refract
    L, N = len(p), len(window.indices)
    nrm = window.normal
    ok = np.ones(L, bool)
    reason = np.full(L, "", "<U4")
    hits = np.full((L, N - 1, 3), np.nan)
    segs = np.full((L, N, 3), np.nan)
    segs[:, 0] = d
    p = p.copy()
    for k, off in enumerate(window.offsets):
        dn = d @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (off - (p - window.origin) @ nrm) / dn
        miss = ok & ~((dn > 0) & (t > 0))
        reason[miss], ok = "miss", ok & ~miss
        p = np.where(ok[:, None], p + np.nan_to_num(t)[:, None] * d, np.nan)
        d_new, through = _refract(d, np.broadcast_to(nrm, d.shape).copy(), window.indices[k] / window.indices[k + 1])
        tir = ok & ~through
        reason[tir], ok = "tir", ok & ~tir
        d = np.where(ok[:, None], d_new, np.nan)
        hits[:, k] = p
        segs[:, k + 1] = d
    return ok, reason, hits, segs, p, d
