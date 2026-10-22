"""The whole-ball fit of `ball_image`, done with AFRT alone: no closed form of the window.

`ball_image.fit` predicts each blob from every image of the ball's surface (`project()`) and weighs each image by
the area it covers, from the analytic Jacobian (`image_jacobians`).  Here the same fit takes:
  * each surface point's image from one AFRT start (Belden 2013), kept only if the traced ray reaches the point and
    meets the surface from outside;
  * the Jacobian d(uv)/dX of each image by central differences of AFRT (step H);
so it uses nothing but AFRT's iterations and vector Snell tracing.  AFRT gives at most one image per surface point:
where the wall shows part of the ball twice, the second copy is missing from the prediction, and the blob that
copy makes is left out of the fit.
"""
import numpy as np

import refractive_camera as rc
from refractive_camera.baselines import afrt

import ball_image

H = 1e-6                                     # m, the finite-difference step


def _afrt_pixel(cam, window, s):
    """The pixel of point s by one AFRT start, or None (no convergence, or a ray that does not reach the point)."""
    d, _, _ = afrt(window, cam.center, s)
    if d is None:
        return None, None
    tr = rc.trace(window, cam.center, d)
    if not tr.ok[0] or tr.miss_distance(np.asarray(s, float)[None])[0] > 1e-9 * np.linalg.norm(cam.center):
        return None, None
    uv, front = cam.pixels_from_directions((d @ cam.R.T)[None])
    if not front[0] or not np.all(np.isfinite(uv)):
        return None, None
    return uv[0], tr.direction[0]


def surface_images(X, cam, window, ball_rgb, normals=None):
    """The visible images of the ball's surface points (`normals`, ball_image.NORMALS by default) by AFRT: pixels
    (K, 2), weights (K,) and d(uv)/dX (K, 2, 3) by central differences of AFRT."""
    normals = ball_image.NORMALS if normals is None else normals
    S = np.asarray(X, float) + ball_image.arm.BALL_R * normals
    uv, w, J = [], [], []
    W, Hh = cam.size
    for k, s in enumerate(S):
        p, d_last = _afrt_pixel(cam, window, s)
        if p is None or not (-0.5 <= p[0] < W - 0.5 and -0.5 <= p[1] < Hh - 0.5):
            continue
        n = normals[k]
        if d_last @ n >= 0:
            continue                                        # the ray reaches the point from inside the ball
        cols = []
        for e in np.eye(3) * H:
            pp, _ = _afrt_pixel(cam, window, s + e)
            pm, _ = _afrt_pixel(cam, window, s - e)
            if pp is None or pm is None:
                break
            cols.append((pp - pm) / (2 * H))
        if len(cols) < 3:
            continue
        Jk = np.stack(cols, -1)
        t1 = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        JT = Jk @ np.stack([t1, t2], -1)
        wk = abs(JT[0, 0] * JT[1, 1] - JT[0, 1] * JT[1, 0]) * ball_image._weights(n[None], ball_rgb)[0]
        uv.append(p)
        w.append(wk)
        J.append(Jk)
    if not uv:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2, 3))
    return np.array(uv), np.array(w), np.array(J)


def predicted_centroids(X, cam, window, ball_rgb, blobs, reach=60.0, regions=None, normals=None):
    """As `ball_image.predicted_centroids`, from AFRT's images."""
    uv, w, J = surface_images(X, cam, window, ball_rgb, normals)
    return ball_image.centroids(uv, w, J, blobs, reach, regions)


def fit(observations, X0, window, ball_rgb, iters=6, weigh=False):
    """`ball_image.fit` with AFRT's predicted centroids: the ball's centre and the rms of the residuals (px)."""
    saved = ball_image.predicted_centroids
    ball_image.predicted_centroids = predicted_centroids
    try:
        return ball_image.fit(observations, X0, window, ball_rgb, iters, weigh)
    finally:
        ball_image.predicted_centroids = saved


def blobs_near(X, cam, window, blobs, gate=60.0, regions=None):
    """The blobs within `gate` px of AFRT's image of X (its only one): the ball's blobs as AFRT sees them; with
    `regions` (each blob's pixels), also theirs."""
    p, _ = _afrt_pixel(cam, window, np.asarray(X, float))
    if p is None or not len(blobs):
        return np.zeros((0, 2)) if regions is None else (np.zeros((0, 2)), [])
    near = np.linalg.norm(np.asarray(blobs)[:, :2] - p, axis=1) < gate
    if regions is None:
        return np.asarray(blobs)[near, :2]
    return np.asarray(blobs)[near, :2], [q for q, k in zip(regions, near) if k]
