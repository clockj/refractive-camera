"""Structure from motion through a fixed window: matching, tracks, triangulation and bundle adjustment with
interchangeable camera models.

The models of how a camera images a point:
* ``RefractiveModel``: `project()` finds every image of the point; the observation is paired with the nearest image,
  and the Jacobians are the analytic ones (`image_jacobians`).
* ``AFRTModel``: the same window, but each point is projected by one AFRT start (Belden 2013), which returns at most
  one ray.  Its Jacobian is the same analytic one, at the ray it found, so the two models differ only in which images
  they find.
* ``PinholeModel``: a pinhole camera with OpenCV distortion that ignores the window, as a user unaware of refraction
  would model it.  Its intrinsics (shared by all views: one physical camera) are refined with the poses.

Bundle adjustment is Levenberg-Marquardt on the Schur complement, with a Huber loss on pixel residuals and priors on
the poses (the robot's kinematics).  Every camera is parameterised by a rotation increment omega (R -> expm([omega]x) R)
and its centre c.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sps

import refractive_camera as rc
from refractive_camera.baselines import afrt
from refractive_camera.jacobian import image_jacobians, pixel_jacobian


# ---- small geometry helpers ---------------------------------------------------------------------------------------

def expm_so3(w):
    """Rotation matrices exp([w]x) of rotation vectors (..., 3)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(th > 1e-12, np.sin(th) / th, 1.0 - th ** 2 / 6)
        b = np.where(th > 1e-12, (1 - np.cos(th)) / th ** 2, 0.5 - th ** 2 / 24)
    return np.eye(3) + a * K + b * K @ K


def log_so3(R):
    """Rotation vectors of rotation matrices (..., 3, 3); exact away from half-turns (all our rotations are small)."""
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.where(th > 1e-12, th / (2 * np.sin(np.maximum(th, 1e-300))), 0.5)
    return v * s[..., None]


def camera(K, dist, R, c, size):
    return rc.Camera(K, R, -R @ c, dist, size)


def huber(s, delta):
    """Huber cost and IRLS weight of residual norms s."""
    cost = np.where(s <= delta, 0.5 * s * s, delta * (s - 0.5 * delta))
    w = np.where(s <= delta, 1.0, delta / np.maximum(s, 1e-300))
    return cost, w


# ---- camera models ------------------------------------------------------------------------------------------------

class RefractiveModel:
    """`project()` through the window: every image; each observation takes its nearest one."""
    name = "refractive"
    n_intr = 0

    def __init__(self, window, K, dist, size, threads=8):
        self.window, self.K, self.dist, self.size, self.threads = window, K, dist, size, threads

    def predict(self, Rs, cs, intr, X, obs_view, obs_point, obs_uv, jac=True):
        n = len(obs_view)
        uv = np.full((n, 2), np.nan)
        Jp = np.zeros((n, 2, 3))
        Jc = np.zeros((n, 2, 6))
        for i in np.unique(obs_view):
            o = np.flatnonzero(obs_view == i)
            cam = camera(self.K, self.dist, Rs[i], cs[i], self.size)
            pts, inv = np.unique(obs_point[o], return_inverse=True)
            im = rc.project(X[pts], cam, self.window, threads=self.threads)
            ok = np.isfinite(im.uv).all(1)
            # nearest image of each observation's point
            best = np.full(len(o), -1)
            bestd = np.full(len(o), np.inf)
            rows_by_pt = [[] for _ in range(len(pts))]
            for r in np.flatnonzero(ok):
                rows_by_pt[im.point[r]].append(r)
            for a, (p, u) in enumerate(zip(inv, obs_uv[o])):
                for r in rows_by_pt[p]:
                    d = (im.uv[r, 0] - u[0]) ** 2 + (im.uv[r, 1] - u[1]) ** 2
                    if d < bestd[a]:
                        best[a], bestd[a] = r, d
            got = best >= 0
            uv[o[got]] = im.uv[best[got]]
            if jac and got.any():
                rows = best[got]
                sub = _rows(im, rows)
                dP, dS = image_jacobians(X[pts], cam, self.window, sub)
                Jp[o[got]], Jc[o[got]] = dP, dS
        return uv, Jp, Jc, np.zeros((n, 2, 0))


class _Rows:
    pass


def _rows(im, rows):
    s = _Rows()
    for f in ("point", "uv", "branch", "lam", "c_z", "direction"):
        setattr(s, f, getattr(im, f)[rows])
    return s


_WINDOW = None


def _afrt_init(window):
    global _WINDOW
    _WINDOW = window


def _afrt_one(args):
    centre, x = args
    d, status, _ = afrt(_WINDOW, centre, x)
    if d is None:
        return None
    tr = rc.trace(_WINDOW, centre, d)                   # accept only a ray that reaches the point (as the benchmark)
    if not tr.ok[0] or tr.miss_distance(x[None])[0] > 1e-9 * max(1.0, np.linalg.norm(x)):
        return None
    return d


class AFRTModel(RefractiveModel):
    """One AFRT start per point; the ray it returns (if it reaches the point) gives the pixel."""
    name = "afrt"

    def __init__(self, window, K, dist, size, workers=8):
        super().__init__(window, K, dist, size)
        self.workers = workers
        self._pool = None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def predict(self, Rs, cs, intr, X, obs_view, obs_point, obs_uv, jac=True):
        n = len(obs_view)
        uv = np.full((n, 2), np.nan)
        Jp = np.zeros((n, 2, 3))
        Jc = np.zeros((n, 2, 6))
        if self._pool is None:
            self._pool = ProcessPoolExecutor(self.workers, initializer=_afrt_init, initargs=(self.window,))
        dirs = list(self._pool.map(_afrt_one, [(cs[i], X[p]) for i, p in zip(obs_view, obs_point)], chunksize=256))
        got = np.array([d is not None for d in dirs])
        if not got.any():
            return uv, Jp, Jc, np.zeros((n, 2, 0))
        D = np.array([d for d in dirs if d is not None])
        g = np.flatnonzero(got)
        for i in np.unique(obs_view[g]):
            o = g[obs_view[g] == i]
            cam = camera(self.K, self.dist, Rs[i], cs[i], self.size)
            Do = D[np.searchsorted(g, o)]
            u, front = cam.pixels_from_directions(Do @ cam.R.T)
            uv[o] = np.where(front[:, None], u, np.nan)
            if jac:
                sub = _from_directions(self.window, cam, X[obs_point[o]], Do)
                dP, dS = image_jacobians(X[obs_point[o]], cam, self.window, sub)
                Jp[o], Jc[o] = dP, dS
        return uv, Jp, Jc, np.zeros((n, 2, 0))


def _from_directions(window, cam, X, D):
    """Rows (point, branch, Lambda, C_z, direction) of rays D from the camera that reach the points X."""
    P = window.to_local(cam.center)
    d = window.vectors_to_local(D)
    n1 = window.indices[0]
    s = _Rows()
    s.point = np.arange(len(X))
    s.direction = D
    s.lam = n1 * (P[0] * d[:, 1] - P[1] * d[:, 0])
    s.c_z = n1 * d[:, 2]
    tr = rc.trace(window, np.broadcast_to(cam.center, D.shape), D)
    e, dd = window.to_local(tr.entry), window.vectors_to_local(tr.direction)
    s_star = -(e[:, 0] * dd[:, 0] + e[:, 1] * dd[:, 1]) / (dd[:, 0] ** 2 + dd[:, 1] ** 2)
    s_x = np.einsum("ij,ij->i", window.to_local(X) - e, dd)
    s.branch = np.where(s_x > s_star, "far", "near")
    return s


class PinholeModel:
    """A pinhole camera with OpenCV distortion (k1, k2, p1, p2, k3), the window ignored; intrinsics shared and
    refined: (fx, fy, cx, cy, k1, k2, p1, p2, k3)."""
    name = "pinhole"
    n_intr = 9

    def __init__(self, size):
        self.size = size

    def predict(self, Rs, cs, intr, X, obs_view, obs_point, obs_uv, jac=True):
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr
        R, c = Rs[obs_view], cs[obs_view]
        dc = np.einsum("nij,nj->ni", R, X[obs_point] - c)                 # camera-frame vector
        x, y = dc[:, 0] / dc[:, 2], dc[:, 1] / dc[:, 2]
        r2 = x * x + y * y
        rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        uv = np.c_[fx * xd + cx, fy * yd + cy]
        uv[dc[:, 2] <= 0] = np.nan
        n = len(obs_view)
        if not jac:
            return uv, None, None, None
        cam = rc.Camera([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dist=(k1, k2, p1, p2, k3))
        PJ = pixel_jacobian(cam, dc)                                     # (n, 2, 3)
        Jp = PJ @ R
        Jw = PJ @ -_skew(dc)
        Jc = np.concatenate([Jw, -Jp], -1)
        Ji = np.zeros((n, 2, 9))
        Ji[:, 0, 0], Ji[:, 1, 1], Ji[:, 0, 2], Ji[:, 1, 3] = xd, yd, 1.0, 1.0
        for col, (gx, gy) in enumerate([(x * r2, y * r2), (x * r2 ** 2, y * r2 ** 2), (2 * x * y, r2 + 2 * y * y),
                                        (r2 + 2 * x * x, 2 * x * y), (x * r2 ** 3, y * r2 ** 3)]):
            Ji[:, 0, 4 + col], Ji[:, 1, 4 + col] = fx * gx, fy * gy
        return uv, Jp, Jc, Ji


def _skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1), np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], -2)


# ---- bundle adjustment --------------------------------------------------------------------------------------------

def bundle_adjust(model, Rs, cs, intr, X, obs, *, prior_R, prior_c, sigma_rot, sigma_c, delta=2.0, miss_px=50.0,
                  iters=25, lam0=1e-3, fix_points=False, log=None):
    """Levenberg-Marquardt on poses (and the model's intrinsics) and points.

    obs : dict with arrays ``view``, ``point``, ``uv``.
    Observations without a prediction (no image, or a failed projection) cost a fixed Huber penalty at `miss_px`
    and carry no gradient.
    Returns the refined (Rs, cs, intr, X) and a dict of diagnostics.
    """
    V, P, m = len(Rs), len(X), model.n_intr
    nc = 6 * V + m
    ov, op, ouv = obs["view"], obs["point"], obs["uv"]
    Rs, cs, intr, X = Rs.copy(), cs.copy(), np.array(intr, float), X.copy()
    miss_cost = huber(np.array(miss_px), delta)[0]

    def prior_terms(Rs_, cs_):
        rr = log_so3(np.einsum("nij,nkj->nik", Rs_, prior_R)) / sigma_rot
        rc_ = (cs_ - prior_c) / sigma_c
        return rr, rc_

    def total_cost(Rs_, cs_, intr_, X_):
        uv, *_ = model.predict(Rs_, cs_, intr_, X_, ov, op, ouv, jac=False)
        s = np.linalg.norm(uv - ouv, axis=1)
        ok = np.isfinite(s)
        c = huber(s[ok], delta)[0].sum() + miss_cost * (~ok).sum()
        rr, rc_ = prior_terms(Rs_, cs_)
        return c + 0.5 * (rr ** 2).sum() + 0.5 * (rc_ ** 2).sum(), s

    cost, s = total_cost(Rs, cs, intr, X)
    lam = lam0
    hist = [cost]
    for it in range(iters):
        uv, Jp, Jc, Ji = model.predict(Rs, cs, intr, X, ov, op, ouv, jac=True)
        r = uv - ouv
        s = np.linalg.norm(r, axis=1)
        ok = np.isfinite(s) & np.isfinite(Jp).all((1, 2)) & np.isfinite(Jc).all((1, 2))
        w = np.zeros(len(s))
        w[ok] = huber(s[ok], delta)[1]
        r = np.where(ok[:, None], r, 0.0)
        # camera-side Jacobian per observation, with its column indices in the reduced system
        Jcam = np.concatenate([Jc, Ji], -1) if m else Jc
        idx = np.concatenate([6 * ov[:, None] + np.arange(6), 6 * V + np.tile(np.arange(m), (len(ov), 1))], 1) \
            if m else 6 * ov[:, None] + np.arange(6)
        Jcam = np.where(ok[:, None, None], Jcam, 0.0)
        Jp_ = np.where(ok[:, None, None], Jp, 0.0)
        wJc = w[:, None, None] * Jcam
        Hcc = np.zeros((nc, nc))
        np.add.at(Hcc, (idx[:, :, None], idx[:, None, :]), np.einsum("nki,nkj->nij", wJc, Jcam))
        gc = np.zeros(nc)
        np.add.at(gc, idx, np.einsum("nki,nk->ni", wJc, r))
        Hpp = np.zeros((P, 3, 3))
        np.add.at(Hpp, op, np.einsum("nki,nkj->nij", w[:, None, None] * Jp_, Jp_))
        gp = np.zeros((P, 3))
        np.add.at(gp, op, np.einsum("nki,nk->ni", w[:, None, None] * Jp_, r))
        Hcp = np.einsum("nki,nkj->nij", wJc, Jp_)                         # (n, mc, 3)
        frozen = np.bincount(op[ok], minlength=P) < 2                     # a point needs two views to move
        # priors on the poses
        rr, rc_ = prior_terms(Rs, cs)
        for i in range(V):
            Hcc[6 * i:6 * i + 3, 6 * i:6 * i + 3] += np.eye(3) / sigma_rot ** 2
            Hcc[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] += np.eye(3) / sigma_c ** 2
            gc[6 * i:6 * i + 3] += rr[i] / sigma_rot
            gc[6 * i + 3:6 * i + 6] += rc_[i] / sigma_c
        while True:
            Hc = Hcc + lam * np.diag(np.diag(Hcc))
            Hp = Hpp + lam * np.einsum("pii->pi", Hpp)[:, :, None] * np.eye(3) + 1e-12 * np.eye(3)
            Hp[frozen] = 1e12 * np.eye(3)
            if fix_points:
                dc = np.linalg.solve(Hc, -gc)
                dp = np.zeros((P, 3))
            else:
                Hp_inv = np.linalg.inv(Hp)
                # Schur complement: S = Hc - sum_j F_j Hp_j^-1 F_j^T with F_j = sum_{o in j} Hcp_o (scattered)
                E = np.einsum("nij,njk->nik", Hcp, Hp_inv[op])                # (n, mc, 3)
                mc = idx.shape[1]
                rows = np.repeat(idx, 3, axis=1).ravel()
                cols = (3 * op[:, None, None] + np.arange(3)[None, None, :]).repeat(mc, 1).ravel()
                F = sps.csr_matrix((Hcp.reshape(-1), (rows, cols)), shape=(nc, 3 * P))
                G = sps.csr_matrix((E.reshape(-1), (rows, cols)), shape=(nc, 3 * P))
                S = Hc - (G @ F.T).toarray()
                b = -gc + G @ (gp.reshape(-1))
                dc = np.linalg.solve(S, b)
                # back-substitution: dp_j = Hp_j^-1 (-gp_j - sum_o Hcp_o^T dc[idx_o])
                t = np.zeros((P, 3))
                np.add.at(t, op, np.einsum("nij,ni->nj", Hcp, dc[idx]))
                dp = np.einsum("pij,pj->pi", Hp_inv, -gp - t)
            dR = expm_so3(dc[:6 * V].reshape(V, 6)[:, :3])
            Rn = np.einsum("nij,njk->nik", dR, Rs)
            cn = cs + dc[:6 * V].reshape(V, 6)[:, 3:]
            In = intr + dc[6 * V:] if m else intr
            Xn = X + dp
            new, s_new = total_cost(Rn, cn, In, Xn)
            if new < cost:
                Rs, cs, intr, X, cost, s = Rn, cn, In, Xn, new, s_new
                lam = max(lam / 3, 1e-9)
                break
            lam *= 4
            if lam > 1e8:
                break
        hist.append(cost)
        if log:
            log(f"    it {it:2d}  cost {cost:.6e}  lambda {lam:.1e}  median residual {np.nanmedian(s):.3f} px  "
                f"unpredicted {np.isnan(s).sum()}")
        if lam > 1e8 or (len(hist) > 2 and hist[-2] - hist[-1] < 1e-9 * hist[-2]):
            break
    return Rs, cs, intr, X, {"cost": hist, "residual": s}


# ---- triangulation ----------------------------------------------------------------------------------------------

def triangulate_pinhole(K, dist, Rs, cs, size, track_obs):
    """Linear triangulation with straight rays (the window ignored)."""
    out = np.full((len(track_obs), 3), np.nan)
    for t, (views, uvs) in enumerate(track_obs):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i, u in zip(views, uvs):
            cam = camera(K, dist, Rs[i], cs[i], size)
            o, d = cam.rays(u[None])
            M = np.eye(3) - np.outer(d[0], d[0])
            A += M
            b += M @ o[0]
        out[t] = np.linalg.solve(A, b)
    return out
