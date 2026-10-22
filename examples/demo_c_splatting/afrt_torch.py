"""AFRT (Belden 2013) for many points at once, in PyTorch: the baseline of Demo C's derivatives.

The same iteration as `refractive_camera.baselines.afrt`, one start per point, vectorised; a point stops moving once
converged, as the scalar version returns then.  Being written in PyTorch it can also be differentiated through its
iterations by autograd, which is one of the two ways to give AFRT a Jacobian (the other being finite differences).
Cylindrical windows of two interfaces only.
"""
import torch

DT = torch.float64


def _roots(p, d, R):
    """Both parameters (t1 <= t2) where the lines p + t d meet the cylinder r = R; NaN where they miss."""
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = p[..., 0] * d[:, 0] + p[..., 1] * d[:, 1]
    c = p[..., 0] ** 2 + p[..., 1] ** 2 - R * R
    disc = b * b - a * c
    bad = (a <= 0) | (disc < 0)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    q = -(b + torch.where(b >= 0, sq, -sq))
    q = torch.where(q == 0, torch.full_like(q, 1e-300), q)
    t1, t2 = q / a, c / q
    lo, hi = torch.minimum(t1, t2), torch.maximum(t1, t2)
    nan = torch.full_like(lo, float("nan"))
    return torch.where(bad, nan, lo), torch.where(bad, nan, hi)


def _normal(h):
    s = torch.sqrt(h[:, 0] ** 2 + h[:, 1] ** 2)
    return torch.stack([h[:, 0] / s, h[:, 1] / s, torch.zeros_like(s)], -1)


def _snell(d, nrm, eta):
    c = -(nrm * d).sum(1)
    flip = c < 0
    nrm = torch.where(flip[:, None], -nrm, nrm)
    c = torch.where(flip, -c, c)
    k = 1.0 - eta * eta * (1.0 - c * c)
    f = eta * c - torch.sqrt(torch.clamp(k, min=0.0))
    out = eta * d + f[:, None] * nrm
    return torch.where((k < 0)[:, None], torch.full_like(out, float("nan")), out)


def _unit(v):
    return v / torch.sqrt((v * v).sum(-1, keepdim=True))


def afrt_local(n, R, XC, X, tol=1e-9, kmax=2000):
    """Directions r1 (M, 3) at the camera, window frame, of AFRT from XC (3,) to points X (M, 3); NaN where it breaks
    down.  Also returns the converged mask and the number of iterations of the slowest point."""
    R1, R2 = R
    n12, n23, n32 = n[0] / n[1], n[1] / n[2], n[2] / n[1]
    r1 = _unit(X - XC)
    t, _ = _roots(XC.expand_as(X), r1, R1)
    Xi1 = XC + t[:, None] * r1
    done = torch.zeros(len(X), dtype=torch.bool, device=X.device)
    Xi1p = Xi1
    k = 0
    for k in range(1, kmax + 1):
        if k > 1:
            M = 0.5 * (Xi1 + Xi1p)
            s = R1 / torch.sqrt(M[:, 0] ** 2 + M[:, 1] ** 2)
            new = torch.stack([M[:, 0] * s, M[:, 1] * s, M[:, 2]], -1)
            Xi1 = torch.where(done[:, None], Xi1, new)
            r1 = torch.where(done[:, None], r1, _unit(Xi1 - XC))
        r2 = _snell(r1, _normal(Xi1), n12)
        ts, _ = _roots(Xi1, r2, R2)
        Xi2 = Xi1 + ts[:, None] * r2
        r3 = _snell(r2, _normal(Xi2), n23)
        m3 = -r3
        _, ts = _roots(X, m3, R2)
        Xi2p = X + ts[:, None] * m3
        r4 = _snell(m3, _normal(Xi2p), n32)
        _, ts = _roots(Xi2p, r4, R1)
        Xi1p_new = Xi2p + ts[:, None] * r4
        Xi1p = torch.where(done[:, None], Xi1p, Xi1p_new)
        conv = (torch.linalg.norm(Xi1 - Xi1p_new, dim=1) <= tol) & (torch.linalg.norm(Xi2 - Xi2p, dim=1) <= tol)
        done = done | conv
        if bool(done.all()) or not bool((~done & torch.isfinite(Xi1p_new).all(1)).any()):
            break
    r1 = torch.where(torch.isfinite(Xi1p).all(1)[:, None], r1, torch.full_like(r1, float("nan")))
    return r1, done, k


class AFRTCamera:
    """Pixels of points by one AFRT start each, through a cylindrical window, in PyTorch (float64)."""

    def __init__(self, cam, window, device):
        if window.kind != "cylinder" or len(window.radii) != 2:
            raise NotImplementedError("a cylinder of two interfaces")
        self.n, self.R = [float(v) for v in window.indices], [float(v) for v in window.radii]
        Q = window.vectors_to_world(__import__("numpy").eye(3)).T                 # window -> world (columns)
        self.Q = torch.tensor(Q, dtype=DT, device=device)
        self.a = torch.tensor(window.axis_point, dtype=DT, device=device)
        self.XC = (torch.tensor(cam.center, dtype=DT, device=device) - self.a) @ self.Q
        self.Rc = torch.tensor(cam.R, dtype=DT, device=device)
        self.K = torch.tensor(cam.K, dtype=DT, device=device)
        self.dist = [float(v) for v in cam.dist]
        self.iterations = 0

    def directions(self, X, tol=1e-9, kmax=2000):
        """World directions at the camera (M, 3) and the converged mask."""
        Xl = (X.to(DT) - self.a) @ self.Q
        r1, done, k = afrt_local(self.n, self.R, self.XC, Xl, tol, kmax)
        self.iterations = k
        return r1 @ self.Q.T, done

    def pixels(self, X, tol=1e-9, kmax=2000):
        """Pixels (M, 2) of the points; NaN where AFRT does not converge."""
        from splat import _pixels
        d, done = self.directions(X, tol, kmax)
        uv, _ = _pixels(self.K, self.dist, d @ self.Rc.T)
        return torch.where(done[:, None], uv, torch.full_like(uv, float("nan"))), done
