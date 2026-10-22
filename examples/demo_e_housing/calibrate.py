"""Calibrating the camera in its housing from views of a dot board in the water, two ways.

* **Pinhole + distortion, calibrated in water** (OpenCV's `calibrateCamera`, five distortion coefficients): the
  usual practice, which lets the lens model absorb what it can of the housing's refraction.
* **The refractive model:** the housing's geometry and indices from its drawing, and as unknowns the lens's
  intrinsics (fx, fy, cx, cy) and the camera's pose in the housing.  The boards' poses are unknown too.  A
  Levenberg-Marquardt fit on the reprojection error, with the analytic derivatives of `image_jacobians`.

Both then estimate poses of known targets: `solvePnP` for the pinhole, the same Levenberg-Marquardt fit with the
camera fixed for the refractive model.
"""
import cv2
import numpy as np

import housing as hz
import refractive_camera as rc
from refractive_camera.jacobian import image_jacobians


def skew(v):
    """Cross-product matrices (..., 3, 3) of vectors (..., 3)."""
    v = np.asarray(v, float)
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def mean_rigid(pairs):
    """The rigid motion (R, t) closest to a list of (R_k, t_k): the projected mean rotation and the mean shift."""
    U, _, Vt = np.linalg.svd(sum(R for R, _ in pairs))
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return R, np.mean([t for _, t in pairs], 0)


def compose(a, b):
    """(R, t) of a after b."""
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def inverse(a):
    return a[0].T, -a[0].T @ a[1]


# ---- pinhole + distortion -------------------------------------------------------------------------------------------
def calibrate_pinhole(views):
    """OpenCV's calibration on views [dict(dots (n, 3) board frame, uv (n, 2))].  Returns dict(K, dist, rms, boards),
    with each board's pose (R, t) in the pinhole's camera frame."""
    obj = [v["dots"].astype(np.float32) for v in views]
    img = [v["uv"].astype(np.float32) for v in views]
    rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, (hz.W, hz.H), None, None)
    boards = [(cv2.Rodrigues(r)[0], t.ravel()) for r, t in zip(rv, tv)]
    return dict(K=K, dist=dist.ravel(), rms=float(rms), boards=boards)


def project_pinhole(model, X):
    """Pixels of points X (M, 3) in the pinhole's camera frame."""
    uv, _ = cv2.projectPoints(np.asarray(X, float).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), model["K"],
                              model["dist"])
    return uv[:, 0]


def pnp_pinhole(model, dots, uv):
    """A target's pose (R, t) in the pinhole's frame from its dots' pixels."""
    ok, r, t = cv2.solvePnP(dots.astype(np.float64), uv.astype(np.float64), model["K"], model["dist"],
                            flags=cv2.SOLVEPNP_ITERATIVE)
    return cv2.Rodrigues(r)[0], t.ravel()


# ---- the refractive model -------------------------------------------------------------------------------------------
def _camera(K, R, c):
    return rc.Camera(K, R, -R @ c, None, (hz.W, hz.H))


def _residuals(cam, window, X, uv_obs, jac=True):
    """Reprojection residuals (M, 2) of points X (M, 3) (housing frame), and their derivatives: d/dX (M, 2, 3), d/d
    (omega, c) of the camera (M, 2, 6) and d/d(fx, fy, cx, cy) (M, 2, 4).  A pixel off the sensor still counts; a
    point not imaged once, in front of the camera, gets NaN rows."""
    im = rc.project(X, cam, window)
    one = im.counts(on_sensor=False) == 1
    uv = np.full((len(X), 2), np.nan)
    rows = np.flatnonzero(one[im.point] & im.in_front)
    uv[im.point[rows]] = im.uv[rows]
    r = uv - uv_obs
    if not jac:
        return r
    dX, dS = image_jacobians(X, cam, window, im)
    dXp, dSp = np.full((len(X), 2, 3), np.nan), np.full((len(X), 2, 6), np.nan)
    dXp[im.point[rows]], dSp[im.point[rows]] = dX[rows], dS[rows]
    d_cam = np.full((len(X), 3), np.nan)
    d_cam[im.point[rows]] = im.direction[rows] @ cam.R.T
    xn, yn = d_cam[:, 0] / d_cam[:, 2], d_cam[:, 1] / d_cam[:, 2]
    zero, one_ = np.zeros(len(X)), np.ones(len(X))
    dK = np.stack([np.stack([xn, zero, one_, zero], -1), np.stack([zero, yn, zero, one_], -1)], -2)
    return r, dXp, dSp, dK


PRIOR_ROT, PRIOR_SHIFT = 3.0, 1.0       # rad, m: a weak prior that pins the housing's symmetries (its gauge)


def calibrate_refractive(views, window, start, iters=60, verbose=False):
    """Fit the intrinsics and the camera's pose in the housing, and the boards' poses, to views
    [dict(dots, uv)], starting from the camera `start` (the drawing's).

    A dome is symmetric about its centre, a tube about its axis and along it, a flat port along its plane: moving the
    camera and every board together by such a motion changes no pixel.  A weak prior that keeps the camera near the
    drawing's pose (3 rad and 1 m per pixel of residual) fixes those motions and barely biases the others.
    Returns dict(camera, rms, boards, iters), with each board's pose (R, t) in the housing frame."""
    K = start.K.copy()
    R0, c0 = start.R.copy(), start.center.copy()
    if window.kind == "sphere" and np.linalg.norm(window.to_local(c0)) < 1e-9:
        c0 = c0 + R0[2] * 1e-4          # 0.1 mm forward: at the very centre the centre's derivative is not computed
    R, c = R0.copy(), c0.copy()
    # Each board's start: a plain pinhole PnP with the drawing's camera, as if there were no housing
    boards = []
    for v in views:
        ok, rv, tv = cv2.solvePnP(v["dots"], v["uv"], K, None)
        Rb, tb = cv2.Rodrigues(rv)[0], tv.ravel()
        boards.append((R.T @ Rb, R.T @ (tb - start.t)))
    n_b = len(views)
    P = 10 + 6 * n_b
    uv_all = np.concatenate([v["uv"] for v in views])

    def prior(R, c):
        return np.concatenate([cv2.Rodrigues(R @ R0.T)[0].ravel() / PRIOR_ROT, (c - c0) / PRIOR_SHIFT])

    def evaluate(K, R, c, boards, jac=True):
        """The residuals (pixels, then the prior's 6) and, with jac, their Jacobian (rows, P)."""
        cam = _camera(K, R, c)
        X = np.concatenate([v["dots"] @ Rb.T + tb for v, (Rb, tb) in zip(views, boards)])
        if not jac:
            return np.concatenate([_residuals(cam, window, X, uv_all, jac=False).ravel(), prior(R, c)])
        r, dX, dS, dK = _residuals(cam, window, X, uv_all)
        J = np.zeros((len(X), 2, P))
        J[:, :, 0:4] = dK
        J[:, :, 4:10] = dS
        k = 0
        for j, (v, (Rb, tb)) in enumerate(zip(views, boards)):
            n = len(v["dots"])
            q = v["dots"] @ Rb.T                                       # the dots, turned, before the shift
            s = slice(10 + 6 * j, 16 + 6 * j)
            J[k:k + n, :, s] = np.concatenate([dX[k:k + n] @ -skew(q), dX[k:k + n]], -1)
            k += n
        Jp = np.zeros((6, P))
        Jp[0:3, 4:7] = np.eye(3) / PRIOR_ROT                           # to first order near the drawing's pose
        Jp[3:6, 7:10] = np.eye(3) / PRIOR_SHIFT
        return np.concatenate([r.ravel(), prior(R, c)]), np.concatenate([J.reshape(-1, P), Jp])

    def cost(r):
        return float(np.sum(r ** 2)) if np.all(np.isfinite(r)) else np.inf

    r, J = evaluate(K, R, c, boards)
    mu = 1e-3
    it = 0
    for it in range(1, iters + 1):
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(12):
            delta = -np.linalg.solve(A + mu * np.diag(np.diag(A)) + 1e-12 * np.eye(P), g)
            K2 = K.copy()
            K2[0, 0] += delta[0]
            K2[1, 1] += delta[1]
            K2[0, 2] += delta[2]
            K2[1, 2] += delta[3]
            if not (np.all(np.isfinite(delta)) and min(K2[0, 0], K2[1, 1]) > 0):
                mu *= 4
                continue
            R2, c2 = hz.expm(delta[4:7]) @ R, c + delta[7:10]
            b2 = [(hz.expm(delta[10 + 6 * j:13 + 6 * j]) @ Rb, tb + delta[13 + 6 * j:16 + 6 * j])
                  for j, (Rb, tb) in enumerate(boards)]
            r2 = evaluate(K2, R2, c2, b2, jac=False)
            if cost(r2) < cost(r):
                K, R, c, boards = K2, R2, c2, b2
                mu = max(mu / 3, 1e-9)
                improved = True
                break
            mu *= 4
        if not improved:
            break
        old = cost(r)
        r, J = evaluate(K, R, c, boards)
        if verbose:
            print(f"      LM {it}: rms {np.sqrt(np.mean(r[:-6] ** 2)):.4f} px")
        if old - cost(r) < 1e-12 * old:
            break
    return dict(camera=_camera(K, R, c), rms=float(np.sqrt(np.mean(r[:-6] ** 2))), boards=boards, iters=it)


def pnp_refractive(cam, window, dots, uv, start, iters=30):
    """A target's pose (R, t) in the housing frame from its dots' pixels, with the camera fixed: Levenberg-Marquardt
    from `start` on the reprojection error, with the analytic derivatives of the images in their points."""
    Rd, td = start

    def evaluate(Rd, td, jac=True):
        X = dots @ Rd.T + td
        if not jac:
            return _residuals(cam, window, X, uv, jac=False)
        r, dX, _, _ = _residuals(cam, window, X, uv)
        return r, np.concatenate([dX @ -skew(dots @ Rd.T), dX], -1)

    def cost(r):
        return float(np.sum(r ** 2)) if np.all(np.isfinite(r)) else np.inf

    r, J = evaluate(Rd, td)
    mu = 1e-3
    for _ in range(iters):
        Jm, rm = J.reshape(-1, 6), r.reshape(-1)
        A, g = Jm.T @ Jm, Jm.T @ rm
        improved = False
        for _ in range(12):
            delta = -np.linalg.solve(A + mu * np.diag(np.diag(A)) + 1e-15 * np.eye(6), g)
            R2, t2 = hz.expm(delta[:3]) @ Rd, td + delta[3:]
            r2 = evaluate(R2, t2, jac=False)
            if cost(r2) < cost(r):
                Rd, td = R2, t2
                mu = max(mu / 3, 1e-9)
                improved = True
                break
            mu *= 4
        if not improved:
            break
        old = cost(r)
        r, J = evaluate(Rd, td)
        if old - cost(r) < 1e-12 * old:
            break
    return Rd, td


def frame_map(estimated, true):
    """The rigid motion M (R, t) from the housing frame to a model's frame that best maps the boards' true poses
    (board -> housing) onto the model's estimates (board -> its frame): the same alignment for both models, as a
    perfect hand-eye calibration would give."""
    return mean_rigid([compose(e, inverse(t)) for e, t in zip(estimated, true)])
