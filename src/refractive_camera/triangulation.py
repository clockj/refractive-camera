"""Triangulation through the window: the point that two or more cameras see at given pixels."""
import numpy as np

from .projection import project
from .tracing import back_project


def _nearest(images, targets):
    """For each point of a projection, the pixel of its in-front image nearest to the matching target; NaN if none."""
    out = np.full((len(images.status), 2), np.nan)
    best = np.full(len(images.status), np.inf)
    for k in np.flatnonzero(images.in_front):
        i = images.point[k]
        e = np.hypot(*(images.uv[k] - targets[i]))
        if e < best[i]:
            best[i], out[i] = e, images.uv[k]
    return out


def triangulate(cameras, pixels, window, *, refine=True, backend=None, max_iter=20, step=None):
    """The point seen at ``pixels[i]`` by ``cameras[i]``, through `window`.

    One pixel per observation; a camera may appear more than once, as for both images of a double image.

    The pixels are back-projected through the window into straight lines in the innermost medium, and the point
    nearest to all of them in the least-squares sense is the first estimate; for exact pixels it is exact.  With
    `refine`, Gauss-Newton (with Levenberg's damping) then minimises the reprojection error, the maximum-likelihood
    estimate under equal, independent Gaussian pixel noise.  Each pixel is compared with the image of the estimate
    nearest to it, and derivatives are central differences of `project()`.

    Parameters
    ----------
    cameras : sequence of Camera
    pixels : (L, 2) array_like
    window : CylindricalWindow or SphericalWindow
    refine : bool
    backend : str, optional
        Passed to `project()`.
    max_iter : int
    step : float, optional
        Finite-difference step, in length units; 1e-6 of the innermost radius by default.

    Returns
    -------
    point : (3,) ndarray
    info : dict
        "rms_px" and "residuals_px" (L, 2) of the final estimate (NaN where a camera has no image near its pixel),
        "lines_ok" (L,) (pixels whose rays reach the innermost medium), "iterations".
    """
    uv = np.asarray(pixels, float).reshape(-1, 2)
    if len(cameras) != len(uv):
        raise ValueError("one camera per pixel")
    entry, direction, lines_ok = np.full((len(uv), 3), np.nan), np.full((len(uv), 3), np.nan), np.zeros(len(uv), bool)
    for cam in {id(c): c for c in cameras}.values():
        rows = [i for i, c in enumerate(cameras) if c is cam]
        tr = back_project(cam, window, uv[rows])
        entry[rows], direction[rows], lines_ok[rows] = tr.entry, tr.direction, tr.ok
    if lines_ok.sum() < 2:
        raise ValueError("fewer than two pixels see into the window")
    S, b = np.zeros((3, 3)), np.zeros(3)
    for p, d in zip(entry[lines_ok], direction[lines_ok]):
        A = np.eye(3) - np.outer(d, d)
        S += A
        b += A @ p
    X = np.linalg.solve(S, b)

    groups = {}
    for i, c in enumerate(cameras):
        groups.setdefault(id(c), (c, []))[1].append(i)
    h = 1e-6 * window.radii[-1] if step is None else step

    def residuals(Xs, targets):
        """Reprojection residuals (L, 2) of each candidate point in Xs (P, 3), against targets (P, L, 2)."""
        out = np.full((len(Xs), len(uv), 2), np.nan)
        for cam, rows in groups.values():
            imgs = project(Xs, cam, window, backend=backend)
            for i in rows:
                out[:, i] = _nearest(imgs, targets[:, i]) - uv[i]
        return out

    def cost(r):
        good = np.all(np.isfinite(r), 1)
        return np.sum(r[good] ** 2), good

    r = residuals(X[None], uv[None])[0]
    c0, good = cost(r)
    it, mu = 0, 1e-3
    if refine:
        for it in range(1, max_iter + 1):
            pred = r + uv                                         # current images; derivatives follow them
            probes = np.array([X + s * h * e for e in np.eye(3) for s in (1.0, -1.0)])
            rp = residuals(probes, np.repeat(pred[None], 6, 0))
            J = np.stack([(rp[2 * j] - rp[2 * j + 1]) / (2 * h) for j in range(3)], -1)      # (L, 2, 3)
            use = good & np.all(np.isfinite(J), (1, 2))
            if use.sum() < 2:
                break
            Jm, rm = J[use].reshape(-1, 3), r[use].reshape(-1)
            g, H = Jm.T @ rm, Jm.T @ Jm
            while True:
                dX = -np.linalg.solve(H + mu * np.diag(np.diag(H)), g)
                r_new = residuals((X + dX)[None], uv[None])[0]
                c_new, good_new = cost(r_new)
                if good_new.sum() >= good.sum() and c_new <= c0:
                    X, r, c0, good, mu = X + dX, r_new, c_new, good_new, max(mu / 10, 1e-12)
                    break
                mu *= 10
                if mu > 1e8:
                    break
            if mu > 1e8 or np.linalg.norm(dX) <= 1e-12 * window.radii[-1]:
                break
    return X, {"rms_px": float(np.sqrt(c0 / max(good.sum(), 1))), "residuals_px": r, "lines_ok": lines_ok,
               "iterations": it}
