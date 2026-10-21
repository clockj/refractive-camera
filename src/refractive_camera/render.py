"""A small ray-traced renderer: dots behind the window as the camera sees them, for synthetic data without a game
engine.  Geometry only: where the dots appear, not how bright they are."""
import numpy as np

from .tracing import trace


def render_dots(camera, window, centers, radius, *, supersample=3, region=None, background=0.85, dot=0.05,
                outside=0.3, chunk=1 << 18):
    """Render spheres of one radius inside the window's innermost medium, as the camera sees them.

    Every pixel is the mean of ``supersample**2`` rays traced from the camera through the window by vector Snell's
    law (`trace`).  A ray that reaches the innermost medium shows `dot` if its straight path there meets a sphere and
    `background` otherwise; a ray that never reaches it (it passes outside the window, or is reflected) shows
    `outside`.

    Parameters
    ----------
    camera : Camera
    window : CylindricalWindow or SphericalWindow
    centers : (D, 3) array_like
        Sphere centres, world frame.
    radius : float
    supersample : int
        Rays per pixel along each axis.
    region : (u0, v0, u1, v1), optional
        Pixel bounds to render, u0 <= u < u1 and v0 <= v < v1; the whole image (camera.size) by default.
    background, dot, outside : float
        Grey levels.
    chunk : int
        Rays traced at a time.

    Returns
    -------
    (v1 - v0, u1 - u0) float ndarray
    """
    if region is None:
        if camera.size is None:
            raise ValueError("give `region`, or a camera with an image size")
        region = (0, 0, camera.size[0], camera.size[1])
    u0, v0, u1, v1 = (int(v) for v in region)
    S = int(supersample)
    off = (np.arange(S) + 0.5) / S - 0.5
    uu = (np.arange(u0, u1)[:, None] + off[None, :]).ravel()
    vv = (np.arange(v0, v1)[:, None] + off[None, :]).ravel()
    V, U = np.meshgrid(vv, uu, indexing="ij")                 # rows v, columns u, each S x S per pixel
    uv = np.stack([U.ravel(), V.ravel()], 1)
    C = np.asarray(centers, float).reshape(-1, 3)
    out = np.empty(len(uv))
    for k in range(0, len(uv), chunk):
        origins, dirs = camera.rays(uv[k:k + chunk])
        tr = trace(window, origins, dirs)
        val = np.where(tr.ok, background, outside)
        ok = np.flatnonzero(tr.ok)
        if ok.size and C.size:
            p, d, L = tr.entry[ok], tr.direction[ok], tr.length[ok]
            w = p[:, None, :] - C[None, :, :]                                     # (rays, dots, 3)
            b = np.einsum("rj,rdj->rd", d, w)
            c = np.einsum("rdj,rdj->rd", w, w) - radius * radius
            disc = b * b - c
            sq = np.sqrt(np.maximum(disc, 0.0))
            hit = (disc >= 0) & (-b + sq >= 0) & (-b - sq <= L[:, None])          # the chord meets [0, L]
            val[ok[hit.any(1)]] = dot
        out[k:k + chunk] = val
    return out.reshape(v1 - v0, S, u1 - u0, S).mean(axis=(1, 3))
