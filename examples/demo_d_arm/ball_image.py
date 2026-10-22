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

Two options of `fit` (part 4 uses both):
  * the blobs' pixels: an image then joins the blob it lands on, and none if it lands on no blob, so a piece of the
    ball the detector missed (a thin sliver seen twice) biases no blob;
  * weigh: next to a caustic the area an image covers grows without bound, and a few surface points decide the blob's
    predicted centroid.  Each blob is then predicted from two sets of surface points (the second a rotation of the
    first); their mean is the prediction, half their difference its uncertainty, and the blob counts for less the
    less certain it is.
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


def _turned(normals, seed):
    """The same points turned by a fixed random rotation (from a random unit quaternion)."""
    q = np.random.default_rng(seed).normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return normals @ R.T


NORMALS = _fibonacci(400)
NORMALS_B = _turned(NORMALS, 1)              # the second set of `fit(weigh=True)`
SIGMA0 = 0.1                                 # px: a blob's uncertainty where the two sets agree
MARGIN = 3.0                                 # px: an image this close to a blob's pixels lands on it


def _weights(normals, ball_rgb):
    """How much the detector (demo_d.orange) counts each surface point, from the renderer's shading."""
    lum = np.full(len(normals), 0.35)
    for direction, power in DEFAULT_LIGHTS:
        lum += power * np.abs(normals @ (direction / np.linalg.norm(direction)))
    rgb = np.clip(np.asarray(ball_rgb)[None] * lum[:, None], 0, 1)
    return np.clip((rgb[:, 0] - rgb[:, 2] - 0.35) / 0.3, 0, 1) * (rgb[:, 1] < 0.75 * rgb[:, 0])


class _Rows:
    pass


def surface_images(X, cam, window, ball_rgb, normals=None):
    """Every visible image of the ball's surface points (`normals`, NORMALS by default): pixels (K, 2), weights (K,)
    and d(uv)/dX (K, 2, 3)."""
    normals = NORMALS if normals is None else normals
    S = np.asarray(X, float) + arm.BALL_R * normals
    im = rc.project(S, cam, window, threads=4)
    rows = np.flatnonzero(im.in_image)
    if not len(rows):
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2, 3))
    tr = rc.trace(window, np.broadcast_to(cam.center, (len(rows), 3)), im.direction[rows])
    pts = im.point[rows]
    front = np.einsum("ij,ij->i", tr.direction, normals[pts]) < 0          # the ray meets the surface from outside
    rows, pts = rows[front], pts[front]
    sub = _Rows()
    for f in ("uv", "branch", "lam", "c_z", "direction"):
        setattr(sub, f, getattr(im, f)[rows])
    sub.point = np.arange(len(rows))
    J, _ = image_jacobians(S[pts], cam, window, sub)                        # (K, 2, 3)
    n = normals[pts]
    t1 = np.cross(n, np.where(np.abs(n[:, [2]]) < 0.9, [0, 0, 1.0], [1.0, 0, 0]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    JT = np.einsum("kij,kjl->kil", J, np.stack([t1, t2], -1))
    w = np.abs(JT[:, 0, 0] * JT[:, 1, 1] - JT[:, 0, 1] * JT[:, 1, 0]) * _weights(n, ball_rgb)
    ok = np.isfinite(w) & np.isfinite(im.uv[rows]).all(1) & np.isfinite(J).all((1, 2))
    return im.uv[rows][ok], w[ok], J[ok]


def centroids(uv, w, J, blobs, reach=60.0, regions=None):
    """Each blob's predicted centroid (k, 2), the weighted mean of the surface images (uv, w, J) that join it, and its
    derivative in the ball's centre (k, 2, 3); NaN for a blob that no image joins.  An image joins the blob nearest to
    it (within `reach` px) or, given each blob's pixels (`regions`), the blob it lands on (within MARGIN px)."""
    out, dout = np.full((len(blobs), 2), np.nan), np.full((len(blobs), 2, 3), np.nan)
    if not len(uv):
        return out, dout
    if regions is None:
        d = np.linalg.norm(uv[:, None, :] - np.asarray(blobs)[None], axis=2)
        which = np.argmin(d, axis=1)
        near = d[np.arange(len(uv)), which] < reach
    else:
        from scipy.spatial import cKDTree
        owner = np.repeat(np.arange(len(regions)), [len(p) for p in regions])
        dist, i = cKDTree(np.concatenate(regions)).query(uv, distance_upper_bound=MARGIN)
        near = np.isfinite(dist)
        which = owner[np.minimum(i, len(owner) - 1)]
    for k in range(len(blobs)):
        m = near & (which == k) & (w > 0)
        if m.any():
            out[k] = (w[m, None] * uv[m]).sum(0) / w[m].sum()
            dout[k] = (w[m, None, None] * J[m]).sum(0) / w[m].sum()       # the ball moves rigidly
    return out, dout


def predicted_centroids(X, cam, window, ball_rgb, blobs, reach=60.0, regions=None, normals=None):
    """Predicted centroid of each observed blob (k, 2) and its derivative in the ball's centre (k, 2, 3), from the
    surface images of `normals` (`centroids`)."""
    uv, w, J = surface_images(X, cam, window, ball_rgb, normals)
    return centroids(uv, w, J, blobs, reach, regions)


def fit(observations, X0, window, ball_rgb, iters=6, weigh=False):
    """The ball's centre whose predicted blob centroids best match the measured ones (Gauss-Newton with Levenberg's
    damping and the analytic derivative of the centroids).

    observations: list of (camera, blob pixels (k, 2)) or (camera, blob pixels (k, 2), each blob's pixels), one entry
    per camera with its blobs of the ball; with the blobs' pixels an image joins the blob it lands on (`centroids`).
    weigh: predict each blob from NORMALS and NORMALS_B, take the mean, and weigh its residual by
    1 / sqrt(SIGMA0^2 + s^2), s half the two predictions' difference (held fixed during each step).
    Returns the centre and the rms of the centroid residuals (px)."""
    X = np.asarray(X0, float)
    sets = (NORMALS, NORMALS_B) if weigh else (NORMALS,)

    def residuals(Y):
        r, J, s = [], [], []
        for c, b, *reg in observations:
            pred = [predicted_centroids(Y, c, window, ball_rgb, b, regions=reg[0] if reg else None, normals=n)
                    for n in sets]
            r.append(np.mean([p for p, _ in pred], 0) - b)
            J.append(np.mean([dp for _, dp in pred], 0))
            s.append(np.repeat(0.5 * np.linalg.norm(pred[0][0] - pred[-1][0], axis=1), 2))
        return np.concatenate(r).ravel(), np.concatenate(J).reshape(-1, 3), np.concatenate(s)

    r, J, s = residuals(X)
    good = np.isfinite(r)
    if good.sum() < 4:
        return X, np.nan
    cost, mu = np.sum(r[good] ** 2), 1e-3
    for it in range(iters):
        if it:
            r, J, s = residuals(X)
        use = good & np.isfinite(J).all(1)
        if use.sum() < 4:
            break
        wt = 1 / np.sqrt(SIGMA0 ** 2 + s[use] ** 2) if weigh else np.ones(use.sum())
        Ju, ru = wt[:, None] * J[use], wt * r[use]
        H, g = Ju.T @ Ju, Ju.T @ ru
        improved = False
        while mu < 1e6:
            dX = -np.linalg.solve(H + mu * np.diag(np.diag(H)), g)
            rn = residuals(X + dX)[0]
            gn = np.isfinite(rn)
            cn = np.sum((wt * rn[use]) ** 2) if gn[use].all() else np.inf
            if cn <= np.sum(ru ** 2):
                X, r, good, mu, improved = X + dX, rn, gn, max(mu / 10, 1e-9), True
                cost = np.sum(r[good] ** 2)
                break
            mu *= 10
        if not improved or np.linalg.norm(dX) < 1e-6:
            break
    return X, float(np.sqrt(cost / max(good.sum(), 1)))


def blobs_near(X, cam, window, blobs, gate=60.0, regions=None):
    """The blobs (k, 2) within `gate` px of any image of X: the ball's blobs in that camera; with `regions` (each
    blob's pixels), also theirs."""
    im = rc.project(np.asarray(X, float), cam, window)
    uv = im.uv[im.in_image]
    if not len(uv) or not len(blobs):
        return np.zeros((0, 2)) if regions is None else (np.zeros((0, 2)), [])
    near = np.linalg.norm(np.asarray(blobs)[:, None, :2] - uv[None], axis=2).min(1) < gate
    if regions is None:
        return np.asarray(blobs)[near, :2]
    return np.asarray(blobs)[near, :2], [p for p, k in zip(regions, near) if k]
