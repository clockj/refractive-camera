"""Gaussian splatting through a refractive window: a small 3D Gaussian splatting in plain PyTorch, with the camera
model interchangeable.

A rasterizer needs, for every Gaussian and view, where its centre appears and how the projection stretches it there
(the Jacobian J, so that its 2-D footprint is J Sigma J^T).  Through a curved window a Gaussian may appear twice, or
not at all, so the projection returns a list of "splats", each tagged with its Gaussian:

* ``RefractiveSplatCamera``: the images are found by `project()` (complete), refreshed every few hundred steps; in
  between, each image's momenta (Lambda, C_z) follow the moving Gaussian by Newton's method from their last value.
  The gradient comes from one Newton step taken with autograd from the converged solution, which is the
  implicit-function derivative; J is the analytic Jacobian of `refractive_camera.jacobian`.
* ``PinholeSplatCamera``: straight rays, with OpenCV distortion if given.  The ordinary 3D Gaussian splatting camera.

Two options of the refractive cameras, off by default:
* ``mask`` (H, W): the pixels whose ray reaches the water (`water_mask`).  A Gaussian in the water can light no other
  pixel, so the rasterizer draws nothing there.  Next to the wall the linear footprint of an image at the water's edge
  would otherwise spill onto pixels that see only the background.
* ``repair`` (ours): an image that Newton's method loses at a step is not skipped until the view's next search; its
  Gaussian is searched again by `project()` at once.

The rasterizer is tile-based and differentiable through autograd; it needs no compiled CUDA.
"""
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import refractive_camera as rc
from refractive_camera.jacobian import cylinder_terms

DT = torch.float64


# ---- cameras --------------------------------------------------------------------------------------------------------

def _distort(x, y, d):
    k1, k2, p1, p2, k3 = d
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    D = k1 + r2 * (2 * k2 + 3 * k3 * r2)
    cross = 2 * x * y * D + 2 * p1 * x + 2 * p2 * y
    j = torch.stack([torch.stack([rad + 2 * x * x * D + 2 * p1 * y + 6 * p2 * x, cross], -1),
                     torch.stack([cross, rad + 2 * y * y * D + 6 * p1 * y + 2 * p2 * x], -1)], -2)
    return xd, yd, j


def _pixels(K, dist, dc):
    """Pixels of camera-frame vectors (M, 3) and d(uv)/d(dc) (M, 2, 3)."""
    z = dc[:, 2]
    x, y = dc[:, 0] / z, dc[:, 1] / z
    xd, yd, jd = _distort(x, y, dist)
    uv = torch.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)
    A = torch.stack([torch.stack([K[0, 0] * jd[:, 0, 0] + K[0, 1] * jd[:, 1, 0],
                                  K[0, 0] * jd[:, 0, 1] + K[0, 1] * jd[:, 1, 1]], -1),
                     torch.stack([K[1, 1] * jd[:, 1, 0], K[1, 1] * jd[:, 1, 1]], -1)], -2)
    zero = torch.zeros_like(z)
    P = torch.stack([torch.stack([1 / z, zero, -x / z], -1), torch.stack([zero, 1 / z, -y / z], -1)], -2)
    return uv, A @ P


def water_mask(cam, window):
    """(H, W) bool: the pixels whose ray, through the pixel's centre, reaches the innermost medium.  One forward trace
    per pixel (Snell's law at each surface), so no closed form of the window enters."""
    W, H = cam.size
    V, U = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    o, d = cam.rays(np.c_[U.ravel(), V.ravel()])
    return rc.trace(window, o, d).ok.reshape(H, W)


class PinholeSplatCamera:
    """uv = K distort(R X + t); the window, if any, ignored."""
    mask = None

    def __init__(self, cam, device):
        self.size = cam.size
        self.K = torch.tensor(cam.K, dtype=DT, device=device)
        self.R = torch.tensor(cam.R, dtype=DT, device=device)
        self.t = torch.tensor(cam.t, dtype=DT, device=device)
        self.dist = [float(v) for v in cam.dist]

    def refresh(self, means):
        pass

    def inherit(self, parents):
        pass

    def project(self, means):
        X = means.to(DT)
        dc = X @ self.R.T + self.t
        valid = dc[:, 2] > 1e-6
        uv, A = _pixels(self.K, self.dist, dc)
        J = A @ self.R
        idx = torch.arange(len(X), device=X.device)
        return idx, uv, J, dc.norm(dim=1), valid


class RefractiveSplatCamera:
    """Every image of every Gaussian through a cylindrical window."""
    mask = None
    repair = False

    def __init__(self, cam, window, device, threads=8):
        if window.kind != "cylinder":
            raise NotImplementedError("cylinders only")
        self.cam, self.window, self.threads, self.device = cam, window, threads, device
        self.size = cam.size
        self.n, self.radii = [float(v) for v in window.indices], [float(v) for v in window.radii]
        P = window.to_local(cam.center)
        self.rP, self.phP, self.zP = float(np.hypot(P[0], P[1])), float(np.arctan2(P[1], P[0])), float(P[2])
        Q = window.vectors_to_world(np.eye(3)).T                     # window -> world (columns)
        self.Q = torch.tensor(Q, dtype=DT, device=device)
        self.a = torch.tensor(window.axis_point, dtype=DT, device=device)
        self.Rc = torch.tensor(cam.R, dtype=DT, device=device)
        self.K = torch.tensor(cam.K, dtype=DT, device=device)
        self.dist = [float(v) for v in cam.dist]
        self.rh = torch.tensor([math.cos(self.phP), math.sin(self.phP), 0.0], dtype=DT, device=device)
        self.ph = torch.tensor([-math.sin(self.phP), math.cos(self.phP), 0.0], dtype=DT, device=device)
        self.zh = torch.tensor([0.0, 0.0, 1.0], dtype=DT, device=device)
        self.g = torch.zeros(0, dtype=torch.long, device=device)
        self.far = torch.zeros(0, dtype=torch.bool, device=device)
        self.lam = torch.zeros(0, dtype=DT, device=device)
        self.cz = torch.zeros(0, dtype=DT, device=device)
        self.failures = 0
        self.searched_again = 0

    def refresh(self, means):
        """All images of every Gaussian centre, by project()."""
        im = rc.project(means.detach().to(DT).cpu().numpy(), self.cam, self.window, threads=self.threads)
        W, H = self.size
        m = 64.0                                                        # keep images up to 64 px off the sensor
        keep = (np.isfinite(im.uv).all(1) & (im.uv[:, 0] > -m) & (im.uv[:, 0] < W + m) & (im.uv[:, 1] > -m)
                & (im.uv[:, 1] < H + m))
        dev = self.device
        self.g = torch.tensor(im.point[keep], dtype=torch.long, device=dev)
        self.far = torch.tensor(im.branch[keep] == "far", device=dev)
        self.lam = torch.tensor(im.lam[keep], dtype=DT, device=dev)
        self.cz = torch.tensor(im.c_z[keep], dtype=DT, device=dev)

    def inherit(self, parents, n_before):
        """New Gaussians n_before, n_before + 1, ... (children of `parents`, in order) start from their parents'
        images; Newton's method then moves each to the child's own position."""
        if not len(parents) or not len(self.g):
            return
        order = torch.argsort(self.g)
        gs = self.g[order]
        lo = torch.searchsorted(gs, parents)
        hi = torch.searchsorted(gs, parents, right=True)
        n = hi - lo
        if int(n.sum()) == 0:
            return
        kid = torch.repeat_interleave(torch.arange(len(parents), device=self.device), n)
        pos = torch.arange(int(n.sum()), device=self.device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        r = order[lo[kid] + pos]
        self.g = torch.cat([self.g, n_before + kid])
        self.far = torch.cat([self.far, self.far[r]])
        self.lam = torch.cat([self.lam, self.lam[r]])
        self.cz = torch.cat([self.cz, self.cz[r]])

    def remap(self, keep_mask):
        """After pruning: drop the rows of removed Gaussians and renumber the rest."""
        new_index = torch.cumsum(keep_mask.long(), 0) - 1
        k = keep_mask[self.g]
        self.g, self.far, self.lam, self.cz = new_index[self.g[k]], self.far[k], self.lam[k], self.cz[k]

    def _terms(self, lam, cz, rX, far):
        return cylinder_terms(lam, cz, self.n, self.radii, self.rP, rX, far, xp=torch)

    def _search_again(self, means, gs):
        """The images of Gaussians gs, found afresh by project(), in place of the ones tracked for them."""
        im = rc.project(means.detach().to(DT)[gs].cpu().numpy(), self.cam, self.window, threads=self.threads)
        W, H = self.size
        m = 64.0
        keep = (np.isfinite(im.uv).all(1) & (im.uv[:, 0] > -m) & (im.uv[:, 0] < W + m) & (im.uv[:, 1] > -m)
                & (im.uv[:, 1] < H + m))
        dev = self.device
        old = torch.isin(self.g, gs)
        self.g = torch.cat([self.g[~old], gs[torch.tensor(im.point[keep], dtype=torch.long, device=dev)]])
        self.far = torch.cat([self.far[~old], torch.tensor(im.branch[keep] == "far", device=dev)])
        self.lam = torch.cat([self.lam[~old], torch.tensor(im.lam[keep], dtype=DT, device=dev)])
        self.cz = torch.cat([self.cz[~old], torch.tensor(im.c_z[keep], dtype=DT, device=dev)])

    def project(self, means, newton_iters=2):
        """As `_project`; with `repair`, the Gaussians whose images Newton's method loses at this step are searched
        again first (an image skipped for good, such as one that grazes, is searched again at every step)."""
        if self.repair:
            failures = self.failures
            with torch.no_grad():
                self._project(means, newton_iters)
            self.failures = failures                    # counted by the call below: the images still lost
            lost = torch.ones(len(self.g), dtype=torch.bool, device=self.device)
            lost[self.last_keep] = False
            if bool(lost.any()):
                gs = torch.unique(self.g[lost])
                self.searched_again += len(gs)
                self._search_again(means, gs)
        return self._project(means, newton_iters)

    def _project(self, means, newton_iters=2):
        """Pixels, footprint Jacobians and depths of the images that Newton's method keeps up with.

        Only the images it keeps enter the autograd graph: an image dropped next to grazing would otherwise carry an
        infinite derivative, and infinity times a zero gradient is NaN."""
        with torch.no_grad():
            X = (means.detach().to(DT)[self.g] - self.a) @ self.Q             # local frame
            rXd = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
            dphi = torch.atan2(X[:, 1], X[:, 0]) - self.phP
            dzd = X[:, 2] - self.zP
            lam, cz = self.lam.clone(), self.cz.clone()
            t = self._terms(lam, cz, rXd, self.far)
            target = dphi + 2 * math.pi * torch.round((t["phi"] - dphi) / (2 * math.pi))
            for _ in range(newton_iters):
                t = self._terms(lam, cz, rXd, self.far)
                g1, g2 = t["phi"] - target, t["z"] - dzd
                det = t["phi_lam"] * t["z_c"] - t["phi_c"] * t["z_lam"]
                dl = -(t["z_c"] * g1 - t["phi_c"] * g2) / det
                dc = -(-t["z_lam"] * g1 + t["phi_lam"] * g2) / det
                lam, cz = lam + dl, cz + dc
                nu2 = torch.stack([nn * nn - cz * cz for nn in self.n], 0)
                rin = [*self.radii, None]
                lmax = torch.stack([torch.sqrt(torch.clamp(nu2[k], min=0)) * (rin[k] if rin[k] is not None else rXd)
                                    for k in range(len(self.n))], 0).min(0).values
                lam = torch.clamp(lam, -lmax * (1 - 1e-12), lmax * (1 - 1e-12))
            t = self._terms(lam, cz, rXd, self.far)
            res_phi, res_z = (t["phi"] - target).abs(), (t["z"] - dzd).abs()
            ok = torch.isfinite(res_phi) & (res_phi < 1e-7) & (res_z < 1e-7 * self.rP) & torch.isfinite(lam)
            for key in ("phi_lam", "phi_c", "z_c", "phi_rx", "z_rx", "phi_rp"):
                ok &= torch.isfinite(t[key])                                  # not on a grazing ray
            ok &= rXd > 1e-9
            self.failures += int((~ok).sum())
            self.lam, self.cz = torch.where(ok, lam, self.lam), torch.where(ok, cz, self.cz)
            keep = torch.nonzero(ok).flatten()
            self.last_keep = keep
            lam0, cz0, far = lam[keep], cz[keep], self.far[keep]
            tk = {k: v[keep] for k, v in t.items()}
            J = torch.stack([torch.stack([tk["phi_lam"], tk["phi_c"]], -1), torch.stack([tk["z_lam"], tk["z_c"]], -1)],
                            -2)
            Jinv = torch.linalg.inv(J)
            shift = (target - dphi)[keep]                                      # the multiple of 2 pi
        g = self.g[keep]
        # one Newton step with autograd: its derivative in X is the implicit-function derivative
        Xg = (means.to(DT)[g] - self.a) @ self.Q
        rX = torch.sqrt(Xg[:, 0] ** 2 + Xg[:, 1] ** 2)
        phX = torch.atan2(Xg[:, 1], Xg[:, 0])
        dz = Xg[:, 2] - self.zP
        tg = self._terms(lam0, cz0, rX, far)
        G = torch.stack([tg["phi"] - (phX - self.phP + shift), tg["z"] - dz], -1)
        step = (Jinv @ G[..., None])[..., 0]
        lam1, cz1 = lam0 - step[:, 0], cz0 - step[:, 1]
        # the ray's direction at the camera, then its pixel
        n1 = self.n[0]
        w1 = torch.sqrt(torch.clamp((n1 * n1 - cz1 * cz1) * self.rP ** 2 - lam1 * lam1, min=0))
        d_r, d_ph, d_z = -w1 / (n1 * self.rP), lam1 / (n1 * self.rP), cz1 / n1
        Dl = d_r[:, None] * self.rh + d_ph[:, None] * self.ph + d_z[:, None] * self.zh
        dc = (Dl @ self.Q.T) @ self.Rc.T
        uv, A = _pixels(self.K, self.dist, dc)
        # the Jacobian d(uv)/dX (EWA footprint), from the analytic chain; held constant in the backward pass
        with torch.no_grad():
            rXk = rX.detach()
            zero = torch.zeros_like(rXk)
            x, y = Xg[:, 0].detach(), Xg[:, 1].detach()
            drX = torch.stack([x / rXk, y / rXk, zero], -1)
            dphX = torch.stack([-y / rXk ** 2, x / rXk ** 2, zero], -1)
            dzX = torch.stack([zero, zero, torch.ones_like(zero)], -1)
            GX = torch.stack([tk["phi_rx"][:, None] * drX - dphX, tk["z_rx"][:, None] * drX - dzX], -2)
            q1 = 1.0 / w1.detach()
            dD_dlam = (lam0 * q1 / (n1 * self.rP))[:, None] * self.rh + (1.0 / (n1 * self.rP)) * self.ph
            dD_dc = (cz0 * self.rP * q1 / n1)[:, None] * self.rh + self.zh / n1
            dD = torch.stack([dD_dlam, dD_dc], -1)
            Jx = A.detach() @ self.Rc @ self.Q @ dD @ (-Jinv @ GX) @ self.Q.T
            # depth for ordering: the ray's geometric length from the camera to the point
            path = torch.zeros_like(rXk)
            radii = [self.rP, *self.radii, None]
            for k, nk in enumerate(self.n):
                nu2k = nk * nk - cz0 * cz0
                ro = radii[k]
                ri = radii[k + 1] if radii[k + 1] is not None else rXk
                wo = torch.sqrt(torch.clamp(nu2k * ro * ro - lam0 * lam0, min=0))
                wi = torch.sqrt(torch.clamp(nu2k * ri * ri - lam0 * lam0, min=0))
                last_far = far if k == len(self.n) - 1 else torch.zeros_like(far)
                path = path + nk * (wo + torch.where(last_far, wi, -wi)) / nu2k   # far: out to r_X and back
        valid = (dc[:, 2] > 0).detach() & torch.isfinite(uv).all(1).detach()
        return g, uv, Jx, path, valid


def sigma_camera(cam, rows, n_sigma):
    """A copy of a refractive splat camera whose images are the given rows, each repeated n_sigma times: Newton's
    method then follows each copy from its row's momenta to one sigma point of the Gaussian, on the same branch."""
    sub = object.__new__(type(cam))
    sub.__dict__.update(cam.__dict__)
    sub.g = torch.arange(len(rows) * n_sigma, device=cam.device)
    sub.far = cam.far[rows].repeat_interleave(n_sigma)
    sub.lam = cam.lam[rows].repeat_interleave(n_sigma)
    sub.cz = cam.cz[rows].repeat_interleave(n_sigma)
    sub.failures = 0
    sub.repair = False                                  # each sigma point stays on its image's branch
    return sub


def render_ut(gauss, camera, bg, max_radius=48.0, min_points=4, newton_iters=6, above=None):
    """Like `render`, but each image's footprint comes from the unscented transform instead of J Sigma J^T: the six
    sigma points mean +- sqrt(3) (columns of R S) are projected on the image's own branch, by Newton's method from the
    centre's momenta and one implicit-function step, and the 2-D mean and covariance are those of their pixels.  Next to
    a caustic the points beyond it have no image on that branch, so the footprint stays on the side where the Gaussian
    is seen, instead of the linear footprint's long symmetric streak.  Refractive splat cameras only.

    With `above` (px), only images whose linear footprint is wider than that (1 sigma) take the unscented one; the
    others keep J Sigma J^T and their centre's pixel."""
    idx, uv_c, J, depth, valid = camera.project(gauss.means)
    keep = valid & torch.isfinite(J).all((1, 2))
    rows = camera.last_keep[keep]
    idx, depth, uv_c, J = idx[keep], depth[keep], uv_c[keep], J[keep]
    lin = J @ gauss.covariances()[idx].to(J.dtype) @ J.transpose(1, 2)
    if above is not None:
        wide = torch.linalg.eigvalsh(lin.detach()).max(1).values > above ** 2
        out_m, out_c = uv_c.clone(), lin.clone()
        sel = torch.nonzero(wide).flatten()
        ok_all = torch.ones(len(idx), dtype=torch.bool, device=uv_c.device)
        if len(sel):
            m_s, c_s, ok_s = _unscented(gauss, camera, rows[sel], idx[sel], min_points, newton_iters)
            out_m = out_m.index_copy(0, sel, m_s.to(out_m.dtype))
            out_c = out_c.index_copy(0, sel, c_s.to(out_c.dtype))
            ok_all = ok_all.index_copy(0, sel, ok_s)
    else:
        out_m, out_c, ok_all = _unscented(gauss, camera, rows, idx, min_points, newton_iters)
    big = torch.linalg.eigvalsh(out_c.detach()).max(1).values > max_radius ** 2
    ok = ok_all & ~big
    m32 = out_m.float()
    m32.retain_grad() if m32.requires_grad else None                  # screen-space gradients, for densification
    img = rasterize(m32[ok], out_c[ok].float(), torch.sigmoid(gauss.color_logits)[idx[ok]],
                    torch.sigmoid(gauss.opacity_logits)[idx[ok]], depth[ok], camera.size[0], camera.size[1], bg,
                    mask=camera.mask)
    return img, m32, idx, int((~ok).sum())


def _unscented(gauss, camera, rows, idx, min_points, newton_iters):
    """2-D mean (K, 2), covariance (K, 2, 2) and usable flag (K,) of the images `rows` (of Gaussians idx), from their six
    sigma points projected on each image's branch."""
    K = len(idx)
    R = quat_to_rot(gauss.quats)[idx].to(DT)
    S = torch.exp(gauss.log_scales)[idx].to(DT)
    cols = (R * S[:, None, :]).transpose(1, 2) * math.sqrt(3.0)          # (K, 3, 3): rows are the scaled axes
    pts = gauss.means.to(DT)[idx][:, None, :] + torch.cat([cols, -cols], 1)  # (K, 6, 3)
    sub = sigma_camera(camera, rows, 6)
    g2, uv2, _, _, valid2 = sub.project(pts.reshape(-1, 3), newton_iters=newton_iters)
    g2, uv2 = g2[valid2], uv2[valid2]
    mask = torch.zeros(K * 6, dtype=torch.bool, device=pts.device)
    mask[g2] = True
    U = torch.zeros(K * 6, 2, dtype=uv2.dtype, device=uv2.device).index_copy(0, g2, uv2).reshape(K, 6, 2)
    mask = mask.reshape(K, 6)
    cnt = mask.sum(1)
    w = mask.to(U.dtype) / cnt.clamp(min=1)[:, None].to(U.dtype)
    m = (w[..., None] * U).sum(1)
    D = (U - m[:, None, :]) * mask[..., None]
    cov2 = torch.einsum("kj,kja,kjb->kab", w, D, D)
    return m, cov2, cnt >= min_points


def expm_so3(w):
    """exp([w]x) of a rotation vector w (3,), differentiable (Rodrigues; series near zero)."""
    th2 = (w * w).sum()
    th = torch.sqrt(th2 + 1e-300)
    small = th2 < 1e-12
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / (th2 + 1e-300))
    z = torch.zeros((), dtype=w.dtype, device=w.device)
    K = torch.stack([torch.stack([z, -w[2], w[1]]), torch.stack([w[2], z, -w[0]]), torch.stack([-w[1], w[0], z])])
    return torch.eye(3, dtype=w.dtype, device=w.device) + a * K + b * (K @ K)


class PoseSplatCamera(RefractiveSplatCamera):
    """A refractive splat camera whose pose is learned with the Gaussians: R = expm([omega]x) R0 and centre c0 + dc,
    omega and dc leaf tensors.

    The pose's gradient comes from the same implicit-function step as the Gaussians': the camera's position enters the
    closed form through r_P, phi_P and z_P, and its rotation turns the ray's direction into a pixel.  AFRT, which has
    no derivative, would need twelve more runs per Gaussian, per view, per step for it (central differences)."""

    def __init__(self, cam, window, device, threads=8):
        super().__init__(cam, window, device, threads)
        self.R0 = torch.tensor(cam.R, dtype=DT, device=device)
        self.c0 = torch.tensor(cam.center, dtype=DT, device=device)
        self.omega = torch.zeros(3, dtype=DT, device=device, requires_grad=True)
        self.dc = torch.zeros(3, dtype=DT, device=device, requires_grad=True)

    def pose(self):
        """(R, centre) as tensors that carry the gradient."""
        return expm_so3(self.omega) @ self.R0, self.c0 + self.dc

    def camera(self):
        """The current pose as a refractive_camera.Camera."""
        with torch.no_grad():
            R, c = self.pose()
            R, c = R.cpu().numpy(), c.cpu().numpy()
        return rc.Camera(self.cam.K, R, -R @ c, self.cam.dist, self.cam.size)

    def refresh(self, means):
        self.cam = self.camera()
        super().refresh(means)

    def project(self, means, newton_iters=2):
        R, c = self.pose()
        P = (c - self.a) @ self.Q
        self.rP = torch.sqrt(P[0] ** 2 + P[1] ** 2)
        self.phP = torch.atan2(P[1], P[0])
        self.zP = P[2]
        z = torch.zeros((), dtype=DT, device=self.device)
        self.rh = torch.stack([torch.cos(self.phP), torch.sin(self.phP), z])
        self.ph = torch.stack([-torch.sin(self.phP), torch.cos(self.phP), z])
        self.Rc = R
        return super().project(means, newton_iters)


_AFRT_WINDOW = None


def _afrt_init(window):
    global _AFRT_WINDOW
    _AFRT_WINDOW = window


def _afrt_ray(args):
    """One AFRT start from the camera centre to a point; the ray, if it reaches the point (as the benchmark checks)."""
    from refractive_camera.baselines import afrt
    centre, x = args
    d, _, _ = afrt(_AFRT_WINDOW, centre, x)
    if d is None:
        return None
    tr = rc.trace(_AFRT_WINDOW, centre, d)
    if not tr.ok[0] or tr.miss_distance(x[None])[0] > 1e-9 * max(1.0, float(np.linalg.norm(x))):
        return None
    return d


class AFRTSplatCamera(RefractiveSplatCamera):
    """Pinhole + AFRT: the same splat camera, but each Gaussian's images come from one AFRT start (Belden 2013)
    instead of project(): at most one image, and none where the iteration fails.  Between refreshes the image is
    followed by the same Newton steps, and its footprint is the same analytic Jacobian, so the two cameras differ only
    in which images they know."""

    _pool = None

    def refresh(self, means):
        from concurrent.futures import ProcessPoolExecutor
        if AFRTSplatCamera._pool is None:
            AFRTSplatCamera._pool = ProcessPoolExecutor(self.threads, initializer=_afrt_init, initargs=(self.window,))
        X = means.detach().to(DT).cpu().numpy()
        centre = self.cam.center
        dirs = list(AFRTSplatCamera._pool.map(_afrt_ray, [(centre, x) for x in X], chunksize=512))
        got = np.array([d is not None for d in dirs])
        g = np.flatnonzero(got)
        dev = self.device
        if not len(g):
            self.g = torch.zeros(0, dtype=torch.long, device=dev)
            self.far = torch.zeros(0, dtype=torch.bool, device=dev)
            self.lam = torch.zeros(0, dtype=DT, device=dev)
            self.cz = torch.zeros(0, dtype=DT, device=dev)
            return
        D = np.array([dirs[i] for i in g])
        w = self.window
        P = w.to_local(centre)
        d = w.vectors_to_local(D)
        n1 = w.indices[0]
        lam = n1 * (P[0] * d[:, 1] - P[1] * d[:, 0])
        cz = n1 * d[:, 2]
        tr = rc.trace(w, np.broadcast_to(centre, D.shape), D)
        e, dd = w.to_local(tr.entry), w.vectors_to_local(tr.direction)
        s_star = -(e[:, 0] * dd[:, 0] + e[:, 1] * dd[:, 1]) / (dd[:, 0] ** 2 + dd[:, 1] ** 2)
        s_x = np.einsum("ij,ij->i", w.to_local(X[g]) - e, dd)
        uv, front = self.cam.pixels_from_directions(D @ self.cam.R.T)
        W, H = self.size
        m = 64.0
        keep = front & np.isfinite(uv).all(1) & (uv[:, 0] > -m) & (uv[:, 0] < W + m) & (uv[:, 1] > -m) & (uv[:, 1] < H + m)
        self.g = torch.tensor(g[keep], dtype=torch.long, device=dev)
        self.far = torch.tensor((s_x > s_star)[keep], device=dev)
        self.lam = torch.tensor(lam[keep], dtype=DT, device=dev)
        self.cz = torch.tensor(cz[keep], dtype=DT, device=dev)


class AFRTPureSplatCamera:
    """Pinhole + AFRT on its own, with no closed form of the window: at every step each Gaussian's image comes from one
    AFRT start (batched, `afrt_torch`), its footprint Jacobian from central differences of AFRT (step h), and its depth
    is the straight distance from the camera.  The pixel enters autograd as uv0 + J (X - X0), so its gradient in the
    Gaussian's mean is that same Jacobian.  At most one image per Gaussian, none where AFRT does not converge."""
    mask = None

    def __init__(self, cam, window, device, threads=8, h=1e-5, kmax=300):
        from afrt_torch import AFRTCamera
        self.cam, self.window, self.device, self.size = cam, window, device, cam.size
        self.ac = AFRTCamera(cam, window, device)
        self.h, self.kmax = h, kmax
        self.Rc = torch.tensor(cam.R, dtype=DT, device=device)
        self.K = torch.tensor(cam.K, dtype=DT, device=device)
        self.dist = [float(v) for v in cam.dist]
        self.centre = torch.tensor(cam.center, dtype=DT, device=device)
        self.failures = 0
        self.last_keep = torch.zeros(0, dtype=torch.long, device=device)

    def refresh(self, means):
        pass                                                   # nothing is tracked between steps

    def inherit(self, parents, n_before):
        pass

    def remap(self, keep_mask):
        pass

    def project(self, means):
        with torch.no_grad():
            X0 = means.detach().to(DT)
            M = len(X0)
            E = torch.eye(3, dtype=DT, device=X0.device) * self.h
            batch = torch.cat([X0] + [X0 + e for e in E] + [X0 - e for e in E])     # one AFRT run for all seven
            d, done = self.ac.directions(batch, kmax=self.kmax)
            dc = d @ self.Rc.T
            uv_all, _ = _pixels(self.K, self.dist, dc)
            ok_all = (done & (dc[:, 2] > 0) & torch.isfinite(uv_all).all(1)).reshape(7, M)
            uv_all = uv_all.reshape(7, M, 2)
            uv0 = uv_all[0]
            J = torch.stack([(uv_all[1 + i] - uv_all[4 + i]) / (2 * self.h) for i in range(3)], -1)    # (M, 2, 3)
            W, H = self.size
            m = 64.0
            ok = ok_all.all(0) & (uv0[:, 0] > -m) & (uv0[:, 0] < W + m) & (uv0[:, 1] > -m) & (uv0[:, 1] < H + m)
            self.failures += int((~ok_all[0]).sum())
            keep = torch.nonzero(ok).flatten()
            self.last_keep = keep
        Xg = means.to(DT)[keep]
        uv = uv0[keep] + (J[keep] @ (Xg - X0[keep])[..., None])[..., 0]
        depth = torch.linalg.norm(X0[keep] - self.centre, dim=1)
        return keep, uv, J[keep], depth, torch.ones(len(keep), dtype=torch.bool, device=X0.device)


# ---- the Gaussians ------------------------------------------------------------------------------------------------

def quat_to_rot(q):
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


class Gaussians(torch.nn.Module):
    def __init__(self, means, scale, color, opacity=0.1):
        super().__init__()
        M = len(means)
        dev = means.device
        self.means = torch.nn.Parameter(means.float())
        self.log_scales = torch.nn.Parameter(torch.full((M, 3), math.log(scale), device=dev))
        q = torch.zeros(M, 4, device=dev)
        q[:, 0] = 1
        self.quats = torch.nn.Parameter(q)
        self.opacity_logits = torch.nn.Parameter(torch.full((M,), math.log(opacity / (1 - opacity)), device=dev))
        self.color_logits = torch.nn.Parameter(torch.logit(color.clamp(0.02, 0.98)).float())

    def covariances(self):
        R = quat_to_rot(self.quats)
        S = torch.exp(self.log_scales)
        M = R * S[:, None, :]
        return M @ M.transpose(1, 2)


# ---- rasterizer ---------------------------------------------------------------------------------------------------

def _chunk(px, py, uv, con, op, col, valid, Tin, pm=None):
    """One chunk of depth-sorted splats over a set of tiles: the colour it adds and the light it lets through.
    pm: the tiles' pixels that may be lit (1) or not (0), if a mask is given."""
    dx = px[:, :, None] - uv[:, None, :, 0]
    dy = py[:, :, None] - uv[:, None, :, 1]
    power = -0.5 * (con[:, None, :, 0] * dx * dx + con[:, None, :, 2] * dy * dy) - con[:, None, :, 1] * dx * dy
    alpha = torch.clamp(op[:, None, :] * torch.exp(torch.clamp(power, max=0.0)), max=0.99)
    alpha = alpha * (valid[:, None, :] & (power <= 0)).to(alpha.dtype)
    if pm is not None:
        alpha = alpha * pm[:, :, None]
    alpha = torch.where(alpha < 1.0 / 255, torch.zeros_like(alpha), alpha)
    Tc = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :, :1]), 1 - alpha[:, :, :-1]], 2), 2)
    w = alpha * Tc * Tin[:, :, None]
    return torch.einsum("tpk,tkc->tpc", w, col), torch.prod(1 - alpha, 2)


def rasterize(uv, cov2d, color, opacity, depth, W, H, bg, tile=16, chunk=64, mask=None):
    """Alpha-composite 2-D Gaussians front to back.  uv (S, 2), cov2d (S, 2, 2), color (S, 3), opacity (S,),
    depth (S,); mask (H, W), if given: the pixels the splats may light (the others show the background).  Returns the
    image (H, W, 3)."""
    dev = uv.device
    S = len(uv)
    a, b, c = cov2d[:, 0, 0] + 0.3, cov2d[:, 0, 1], cov2d[:, 1, 1] + 0.3        # 0.3 px^2: 3DGS's low-pass filter
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mid = 0.5 * (a + c)
    lam1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3 * torch.sqrt(lam1)).detach()
    tx, ty = (W + tile - 1) // tile, (H + tile - 1) // tile
    u, v = uv[:, 0].detach(), uv[:, 1].detach()
    x0 = torch.clamp(((u - radius) / tile).floor(), 0, tx - 1).long()
    x1 = torch.clamp(((u + radius) / tile).floor(), 0, tx - 1).long()
    y0 = torch.clamp(((v - radius) / tile).floor(), 0, ty - 1).long()
    y1 = torch.clamp(((v + radius) / tile).floor(), 0, ty - 1).long()
    on = (u + radius >= 0) & (u - radius < W) & (v + radius >= 0) & (v - radius < H) & (det > 0)
    nx, ny = (x1 - x0 + 1) * on, (y1 - y0 + 1) * on
    cnt = nx * ny
    total = int(cnt.sum())
    if S == 0 or total == 0:
        return bg.to(uv.dtype).expand(H, W, 3).clone()
    sid = torch.repeat_interleave(torch.arange(S, device=dev), cnt)
    off = torch.arange(total, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    tile_id = (y0[sid] + off // nx[sid]) * tx + (x0[sid] + off % nx[sid])
    rank = torch.empty(S, dtype=torch.long, device=dev)
    rank[torch.argsort(depth.detach())] = torch.arange(S, device=dev)
    key = tile_id * S + rank[sid]
    key, order = torch.sort(key)
    sid, tile_id = sid[order], tile_id[order]
    n_t = tx * ty
    counts = torch.bincount(tile_id, minlength=n_t)
    starts = torch.cumsum(counts, 0) - counts
    kmax = int(counts.max())
    # pixel centres of every tile
    py, px = torch.meshgrid(torch.arange(tile, device=dev), torch.arange(tile, device=dev), indexing="ij")
    tyy, txx = torch.div(torch.arange(n_t, device=dev), tx, rounding_mode="floor"), torch.arange(n_t, device=dev) % tx
    dt = uv.dtype
    pix_x = (txx[:, None] * tile + px.reshape(1, -1)).to(dt)
    pix_y = (tyy[:, None] * tile + py.reshape(1, -1)).to(dt)
    out = torch.zeros(n_t, tile * tile, 3, device=dev, dtype=dt)
    T = torch.ones(n_t, tile * tile, device=dev, dtype=dt)
    uvf, conf, colf, opf = uv, conic.to(dt), color.to(dt), opacity.to(dt)
    pm = None
    if mask is not None:                                     # the mask, laid out like the tiles' pixels
        mt = torch.zeros(ty * tile, tx * tile, device=dev, dtype=dt)
        mt[:H, :W] = torch.as_tensor(mask, device=dev).to(dt)
        pm = mt.reshape(ty, tile, tx, tile).permute(0, 2, 1, 3).reshape(n_t, tile * tile)
    for c0 in range(0, kmax, chunk):
        # tiles that still have splats in this chunk and light left to block
        live = torch.nonzero((counts > c0) & (T.detach().max(1).values > 1e-4)).flatten()
        if not len(live):
            break
        j = torch.arange(c0, min(c0 + chunk, kmax), device=dev)
        valid = j[None, :] < counts[live, None]                                 # (L, k)
        gidx = torch.where(valid, starts[live, None] + j[None, :], torch.zeros_like(j[None, :]))
        s = sid[gidx]                                                           # (L, k)
        o_add, t_mul = checkpoint(_chunk, pix_x[live], pix_y[live], uvf[s], conf[s], opf[s], colf[s], valid,
                                  T[live], None if pm is None else pm[live], use_reentrant=False)
        out = out.index_add(0, live, o_add)
        T = T.index_copy(0, live, T[live] * t_mul)
    out = out + T[:, :, None] * bg.to(dt)
    img = out.reshape(ty, tx, tile, tile, 3).permute(0, 2, 1, 3, 4).reshape(ty * tile, tx * tile, 3)
    return img[:H, :W]


def render(gauss, camera, bg, max_radius=48.0):
    """Render the Gaussians through `camera`; returns the image and the 2-D means (for densification)."""
    idx, uv, J, depth, valid = camera.project(gauss.means)
    keep = valid & torch.isfinite(J).all((1, 2))
    idx, uv, J, depth = idx[keep], uv[keep], J[keep], depth[keep]
    cov3 = gauss.covariances()[idx].to(J.dtype)
    cov2 = J @ cov3 @ J.transpose(1, 2)
    big = torch.linalg.eigvalsh(cov2.detach()).max(1).values > max_radius ** 2   # next to a caustic: J blows up
    small = ~big
    uv32 = uv.float()
    uv32.retain_grad() if uv32.requires_grad else None
    img = rasterize(uv32[small], cov2[small].float(), torch.sigmoid(gauss.color_logits)[idx[small]],
                    torch.sigmoid(gauss.opacity_logits)[idx[small]], depth[small], camera.size[0], camera.size[1], bg,
                    mask=camera.mask)
    return img, uv32, idx, int(big.sum())


def ssim(a, b):
    """SSIM of images (H, W, 3) with an 11-pixel Gaussian window."""
    import torch.nn.functional as F
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()
