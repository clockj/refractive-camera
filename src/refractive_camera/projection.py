"""Forward projection: every image of every point, as the camera sees it through the window."""
from dataclasses import dataclass

import numpy as np

from . import _reference

try:
    from . import _core
except ImportError:                                   # built without a C++ compiler: pure Python only
    _core = None


def available_backends():
    """The backends this installation has: ("cpp", "python") with the compiled core, else ("python",)."""
    return (("cpp",) if _core is not None else ()) + ("python",)


def _backend(name):
    if name is None:
        name = "cpp" if _core is not None else "python"
    if name == "cpp":
        if _core is None:
            raise RuntimeError("the compiled backend is not built: use backend='python', or reinstall the package "
                               "with a C++17 compiler available")
        return _core
    if name == "python":
        return _reference
    raise ValueError(f"unknown backend {name!r}: use 'cpp' or 'python'")


@dataclass(frozen=True)
class Images:
    """Every image of every point: one row per image, grouped by point in input order, and one status per point.

    A point can have no image (the window hides it), one, or several: a near- and a far-branch ray, rays turning
    either way round the axis, rays that sweep past pi.

    Attributes
    ----------
    point : (K,) int
        The index of the point each row images.
    uv : (K, 2) float
        The pixel; NaN where the ray leaves the camera backwards (or past the lens model's range), and for a ring.
    branch : (K,) str
        "near" (the ray meets the point before its closest approach to the axis or centre), "far" (after it), or
        "ring" (a sphere's axial ring: a cone of rays imaging a circle, for points on the line through the centre
        and the camera).
    lam, c_z : (K,) float
        The ray's conserved momenta: Lambda = n r dphi/ds (signed; its sign is the sense of rotation) and
        C_z = n dz/ds about the cylinders' axis (0 for spheres).
    direction : (K, 3) float
        The ray's unit direction at the camera centre, in the world frame (NaN for a ring).
    in_front : (K,) bool
        The ray leaves the camera forwards and has a pixel.
    in_image : (K,) bool
        ... and that pixel lies on the sensor.
    status : (M,) str
        Per point: "ok" (every image was searched for), "outside_model" (the point is not strictly inside the
        innermost interface, so nothing was searched), or "capped" (the far-branch search hit its cap and is not
        certified complete).
    """

    point: np.ndarray
    uv: np.ndarray
    branch: np.ndarray
    lam: np.ndarray
    c_z: np.ndarray
    direction: np.ndarray
    in_front: np.ndarray
    in_image: np.ndarray
    status: np.ndarray

    def __len__(self):
        return len(self.point)

    def counts(self, on_sensor=True):
        """Images per point: those on the sensor (default), or every ray found."""
        rows = self.in_image if on_sensor else np.ones(len(self.point), bool)
        return np.bincount(self.point[rows], minlength=len(self.status))

    def rows_of(self, i):
        """Indices of the rows that image point i."""
        return np.flatnonzero(self.point == i)


def project(points, camera, window, *, backend=None, tol=1e-9, threads=1):
    """Every image of each point, as `camera` sees it through `window`.

    Parameters
    ----------
    points : (M, 3) or (3,) array_like
        World points; those not strictly inside the innermost interface get the status "outside_model".
    camera : Camera
        Its centre must lie outside the outermost interface.
    window : CylindricalWindow or SphericalWindow
    backend : {"cpp", "python"}, optional
        The compiled core (default when built) or the pure-Python reference; both run the same search.
    tol : float
        Accepted residual of a ray, on the dimensionless sweep (the search scales lengths by a power of two near the
        camera's distance from the axis or centre, so the result does not depend on the length unit).
    threads : int
        Threads for the compiled backend (0: all cores); the Python backend runs on one.

    Returns
    -------
    Images
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    centre = camera.center
    if not window.outside(centre):
        raise ValueError("the camera must be outside the window's outermost interface")
    ok = np.all(np.isfinite(pts), 1)
    ok[ok] = window.inside(pts[ok])
    status = np.where(ok, "ok", "outside_model").astype("<U13")
    idx = np.flatnonzero(ok)

    be = _backend(backend)
    n, R = list(window.indices), list(window.radii)
    P0 = window.to_local(centre)
    X = window.to_local(pts[idx])
    if window.kind == "cylinder":
        rows = be.project_cylinder(n, R, P0, X, tol, threads)
    elif window.kind == "sphere":
        rows = be.project_sphere(n, R, P0, X, tol, threads)
    else:
        raise TypeError(f"unsupported window {window!r}")

    ring = rows["ring"]
    direction = window.vectors_to_world(rows["dir"])
    direction[ring] = np.nan
    uv, in_front = camera.pixels_from_directions(direction @ camera.R.T)
    in_front &= np.all(np.isfinite(uv), 1)
    status[idx[rows["capped"]]] = "capped"
    return Images(
        point=idx[rows["point"]],
        uv=uv,
        branch=np.where(ring, "ring", np.where(rows["branch"] == 1, "far", "near")),
        lam=rows["lam"],
        c_z=rows["c_z"],
        direction=direction,
        in_front=in_front,
        in_image=in_front & camera.in_image(uv),
        status=status,
    )
