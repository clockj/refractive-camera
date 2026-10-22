"""The image of the whole ball, not of its centre: fitting the ball's position to its blobs.

Near the wall a camera sees the ball stretched, clipped at the shadow's edge, or twice, and two copies may merge into
one blob; a blob's centroid is then not the image of the ball's centre, and a triangulation that takes it for one is
off by millimetres.  The blobs are predicted from the model instead:
  * points spread evenly over the ball's surface are projected by `project()`, which gives every image of each;
  * an image counts if its ray meets the surface from outside (the ball hides its own back);
  * it is weighted by the area it covers in the image, |det(J T)| with J = d(uv)/dX from `image_jacobians` and T the
    surface's tangent plane, and by how the detector weighs its colour (the lit side counts more);
  * each image joins the observed blob nearest to it, and each blob's predicted centroid is the weighted mean.
The ball's centre is then fitted (Gauss-Newton) so that the predicted centroids match the measured ones; a centroid's
derivative in the centre is the weighted mean of its images' J.  It takes every image and the Jacobian, which one AFRT
start does not give.
"""
import numpy as np

import refractive_camera as rc
from refractive_camera.jacobian import image_jacobians
from refractive_camera.scene import DEFAULT_LIGHTS

import arm


def _fibonacci(n):
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    s = np.sqrt(1 - z * z)
    return np.c_[s * np.cos(phi), s * np.sin(phi), z]


NORMALS = _fibonacci(400)


def _weights(normals, ball_rgb):
    """How much the detector (demo_d.orange) counts each surface point, from the renderer's shading."""
    lum = np.full(len(normals), 0.35)
    for direction, power in DEFAULT_LIGHTS:
        lum += power * np.abs(normals @ (direction / np.linalg.norm(direction)))
    rgb = np.clip(np.asarray(ball_rgb)[None] * lum[:, None], 0, 1)
    return np.clip((rgb[:, 0] - rgb[:, 2] - 0.35) / 0.3, 0, 1) * (rgb[:, 1] < 0.75 * rgb[:, 0])


class _Rows:
    pass


def surface_images(X, cam, window, ball_rgb):
    """Every visible image of the ball's surface points: pixels (K, 2), weights (K,) and d(uv)/dX (K, 2, 3)."""
    S = np.asarray(X, float) + arm.BALL_R * NORMALS
    im = rc.project(S, cam, window, threads=4)
    rows = np.flatnonzero(im.in_image)
    if not len(rows):
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2, 3))
    tr = rc.trace(window, np.broadcast_to(cam.center, (len(rows), 3)), im.direction[rows])
    pts = im.point[rows]
    front = np.einsum("ij,ij->i", tr.direction, NORMALS[pts]) < 0          # the ray meets the surface from outside
    rows, pts = rows[front], pts[front]
    sub = _Rows()
    for f in ("uv", "branch", "lam", "c_z", "direction"):
        setattr(sub, f, getattr(im, f)[rows])
    sub.point = np.arange(len(rows))
    J, _ = image_jacobians(S[pts], cam, window, sub)                        # (K, 2, 3)
    n = NORMALS[pts]
    t1 = np.cross(n, np.where(np.abs(n[:, [2]]) < 0.9, [0, 0, 1.0], [1.0, 0, 0]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    JT = np.einsum("kij,kjl->kil", J, np.stack([t1, t2], -1))
    w = np.abs(JT[:, 0, 0] * JT[:, 1, 1] - JT[:, 0, 1] * JT[:, 1, 0]) * _weights(n, ball_rgb)
    ok = np.isfinite(w) & np.isfinite(im.uv[rows]).all(1) & np.isfinite(J).all((1, 2))
    return im.uv[rows][ok], w[ok], J[ok]


def predicted_centroids(X, cam, window, ball_rgb, blobs, reach=60.0):
    """Predicted centroid of each observed blob (k, 2), the weighted mean of the surface images nearest to it (within
    `reach` px), and its derivative in the ball's centre (k, 2, 3); NaN for a blob that no image joins."""
    uv, w, J = surface_images(X, cam, window, ball_rgb)
    out, dout = np.full((len(blobs), 2), np.nan), np.full((len(blobs), 2, 3), np.nan)
    if not len(uv):
        return out, dout
    d = np.linalg.norm(uv[:, None, :] - np.asarray(blobs)[None], axis=2)
    which = np.argmin(d, axis=1)
    near = d[np.arange(len(uv)), which] < reach
    for k in range(len(blobs)):
        m = near & (which == k) & (w > 0)
        if m.any():
            out[k] = (w[m, None] * uv[m]).sum(0) / w[m].sum()
            dout[k] = (w[m, None, None] * J[m]).sum(0) / w[m].sum()       # the ball moves rigidly
    return out, dout


def fit(observations, X0, window, ball_rgb, iters=6):
    """The ball's centre whose predicted blob centroids best match the measured ones (Gauss-Newton with Levenberg's
    damping and the analytic derivative of the centroids).

    observations: list of (camera, blob pixels (k, 2)), one entry per camera with its blobs of the ball.
    Returns the centre and the rms of the centroid residuals (px)."""
    X = np.asarray(X0, float)

    def residuals(Y, jac=False):
        r, J = [], []
        for c, b in observations:
            p, dp = predicted_centroids(Y, c, window, ball_rgb, b)
            r.append(p - b)
            J.append(dp)
        r = np.concatenate(r).ravel()
        return (r, np.concatenate(J).reshape(-1, 3)) if jac else r

    r, J = residuals(X, True)
    good = np.isfinite(r)
    if good.sum() < 4:
        return X, np.nan
    cost, mu = np.sum(r[good] ** 2), 1e-3
    for it in range(iters):
        if it:
            r, J = residuals(X, True)
        use = good & np.isfinite(J).all(1)
        if use.sum() < 4:
            break
        Ju, ru = J[use], r[use]
        H, g = Ju.T @ Ju, Ju.T @ ru
        improved = False
        while mu < 1e6:
            dX = -np.linalg.solve(H + mu * np.diag(np.diag(H)), g)
            rn = residuals(X + dX)
            gn = np.isfinite(rn)
            cn = np.sum(rn[use & gn] ** 2) if (use & gn).sum() == use.sum() else np.inf
            if cn <= np.sum(ru ** 2):
                X, r, good, mu, improved = X + dX, rn, gn, max(mu / 10, 1e-9), True
                cost = np.sum(r[good] ** 2)
                break
            mu *= 10
        if not improved or np.linalg.norm(dX) < 1e-6:
            break
    return X, float(np.sqrt(cost / max(good.sum(), 1)))


def blobs_near(X, cam, window, blobs, gate=60.0):
    """The blobs (k, 2) within `gate` px of any image of X: the ball's blobs in that camera."""
    im = rc.project(np.asarray(X, float), cam, window)
    uv = im.uv[im.in_image]
    if not len(uv) or not len(blobs):
        return np.zeros((0, 2))
    d = np.linalg.norm(np.asarray(blobs)[:, None, :2] - uv[None], axis=2).min(1)
    return np.asarray(blobs)[d < gate, :2]
