"""The pinhole camera with OpenCV's lens distortion, in OpenCV's conventions."""
import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Camera:
    """A pinhole camera: intrinsics, lens distortion, pose and image size.

    Conventions (OpenCV's):

    * camera frame: x right, y down, z forward along the optical axis;
    * pose: ``X_cam = R @ X_world + t``, so the centre is ``-R.T @ t``;
    * pixels: (u, v) with u to the right and v down, the centre of the top-left pixel at (0, 0);
    * distortion: OpenCV's (k1, k2, p1, p2, k3), acting on the normalised coordinates (x/z, y/z).  It describes the
      lens only.  The window's refraction belongs to the window, never to a distortion term.

    Parameters
    ----------
    K : (3, 3) array_like
        Intrinsic matrix ``[[fx, s, cx], [0, fy, cy], [0, 0, 1]]``, in pixels.
    R, t : array_like, optional
        World-to-camera rotation (3, 3) and translation (3,); identity and zero by default.
    dist : array_like, optional
        Up to five distortion coefficients (k1, k2, p1, p2, k3); missing ones are zero.
    size : (int, int), optional
        Image (width, height) in pixels.  Without it every pixel counts as inside the image.
    """

    def __init__(self, K, R=None, t=None, dist=None, size=None):
        K = np.array(K, float)
        if K.shape != (3, 3) or K[1, 0] != 0 or K[2, 0] != 0 or K[2, 1] != 0 or K[2, 2] != 1:
            raise ValueError("K must have the form [[fx, s, cx], [0, fy, cy], [0, 0, 1]]")
        if not (K[0, 0] > 0 and K[1, 1] > 0):
            raise ValueError("fx and fy must be positive")
        R = np.eye(3) if R is None else np.array(R, float)
        if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-9) or np.linalg.det(R) <= 0:
            raise ValueError("R must be a rotation matrix")
        t = np.zeros(3) if t is None else np.array(t, float).reshape(3)
        d = np.zeros(5)
        if dist is not None:
            dist = np.ravel(np.asarray(dist, float))
            if dist.size > 5:
                raise ValueError("at most five distortion coefficients, (k1, k2, p1, p2, k3)")
            d[:dist.size] = dist
        self.K, self.R, self.t, self.dist = K, R, t, d
        self.size = None if size is None else (int(size[0]), int(size[1]))

    @classmethod
    def look_at(cls, position, target, up=(0.0, 0.0, 1.0), *, f, size, principal_point=None, fy=None, dist=None):
        """A camera at `position` whose optical axis passes through `target`, with `up` pointing up in the image.

        `f` (and `fy`, by default `f`) in pixels.  The principal point defaults to the image centre,
        ((W - 1)/2, (H - 1)/2).
        """
        position = np.asarray(position, float)
        z = _unit(np.asarray(target, float) - position)
        x = np.cross(z, np.asarray(up, float))
        if np.linalg.norm(x) < 1e-12 * np.linalg.norm(up):
            raise ValueError("`up` is parallel to the viewing direction")
        x = _unit(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        W, H = size
        cx, cy = ((W - 1) / 2.0, (H - 1) / 2.0) if principal_point is None else principal_point
        K = [[f, 0.0, cx], [0.0, f if fy is None else fy, cy], [0.0, 0.0, 1.0]]
        return cls(K, R, -R @ position, dist, size)

    def __repr__(self):
        k = self.K
        return (f"Camera(fx={k[0, 0]:g}, fy={k[1, 1]:g}, cx={k[0, 2]:g}, cy={k[1, 2]:g}, "
                f"centre={np.array2string(self.center, precision=4)}, size={self.size})")

    @property
    def center(self):
        """The camera centre (pinhole) in world coordinates."""
        return -self.R.T @ self.t

    # ------------------------------------------------------------------------------------------------------ the lens
    def distort(self, xy):
        """OpenCV's distortion of normalised coordinates (..., 2)."""
        k1, k2, p1, p2, k3 = self.dist
        xy = np.asarray(xy, float)
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([xd, yd], -1)

    def _distort_jacobian(self, xy):
        """d(distort)/d(x, y) as (j11, j12, j21, j22); radial'(r2) = k1 + 2 k2 r2 + 3 k3 r2^2."""
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        D = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)
        cross = 2.0 * x * y * D + 2.0 * p1 * x + 2.0 * p2 * y
        return (radial + 2.0 * x * x * D + 2.0 * p1 * y + 6.0 * p2 * x, cross,
                cross, radial + 2.0 * y * y * D + 6.0 * p1 * y + 2.0 * p2 * x)

    def _lens_monotone(self, xy):
        """Where the radial part still increases with the radius: beyond that the polynomial folds back, and a
        direction outside the lens's field could land inside the image."""
        k1, k2, _, _, k3 = self.dist
        r2 = xy[..., 0] ** 2 + xy[..., 1] ** 2
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        return (radial > 0) & (radial + 2.0 * r2 * (k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)) > 0)

    def undistort(self, xy_d, iters=30):
        """Invert distort() by Newton's method; NaN where it does not converge."""
        xy_d = np.asarray(xy_d, float)
        if not self.dist.any():
            return xy_d.copy()
        xy = xy_d.copy()
        for _ in range(iters):
            r = self.distort(xy) - xy_d
            if np.all(np.abs(r) <= 1e-15 * (1.0 + np.abs(xy_d))):
                break
            j11, j12, j21, j22 = self._distort_jacobian(xy)
            det = j11 * j22 - j12 * j21
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.stack([(j22 * r[..., 0] - j12 * r[..., 1]) / det,
                                 (j11 * r[..., 1] - j21 * r[..., 0]) / det], -1)
            xy = xy - step
        ok = np.all(np.abs(self.distort(xy) - xy_d) <= 1e-12 * (1.0 + np.abs(xy_d)), -1) & self._lens_monotone(xy)
        return np.where(ok[..., None], xy, np.nan)

    # -------------------------------------------------------------------------------------- directions and pixels
    def pixels_from_directions(self, d_cam):
        """Pixels of camera-frame directions (..., 3), and whether each points forwards (z > 0).

        A direction that points backwards, or beyond the lens model's monotone range, has no pixel (NaN).
        """
        d = np.asarray(d_cam, float)
        in_front = d[..., 2] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = d[..., :2] / d[..., 2:3]
        ok = in_front & self._lens_monotone(xy)
        xd = self.distort(np.where(ok[..., None], xy, np.nan))
        K = self.K
        u = K[0, 0] * xd[..., 0] + K[0, 1] * xd[..., 1] + K[0, 2]
        v = K[1, 1] * xd[..., 1] + K[1, 2]
        return np.stack([u, v], -1), in_front

    def directions_from_pixels(self, uv):
        """Unit camera-frame directions of pixels (..., 2)."""
        uv = np.asarray(uv, float)
        K = self.K
        y = (uv[..., 1] - K[1, 2]) / K[1, 1]
        x = (uv[..., 0] - K[0, 2] - K[0, 1] * y) / K[0, 0]
        xy = self.undistort(np.stack([x, y], -1))
        d = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def rays(self, uv):
        """World rays through pixels (..., 2): origins (the centre) and unit directions."""
        d = self.directions_from_pixels(uv) @ self.R
        return np.broadcast_to(self.center, d.shape).copy(), d

    def project_points(self, points):
        """Pinhole projection of world points (..., 3) with no window: pixels and in-front flags."""
        return self.pixels_from_directions(np.asarray(points, float) @ self.R.T + self.t)

    def in_image(self, uv):
        """Whether pixels (..., 2) fall on the sensor: -0.5 <= u < W - 0.5 and -0.5 <= v < H - 0.5."""
        uv = np.asarray(uv, float)
        u, v = uv[..., 0], uv[..., 1]
        if self.size is None:
            return np.isfinite(u) & np.isfinite(v)
        W, H = self.size
        return (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
