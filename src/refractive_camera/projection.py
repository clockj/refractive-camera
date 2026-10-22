"""Forward projection: every image of every point, as the camera sees it through the window."""
from dataclasses import dataclass

import numpy as np

from . import _reference
from .flat import project_flat

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
    """Every image of every point: one row per image, grouped by point in input order (in no particular order within
    a point), and one status per point.

    A point can have no image (the window hides it), one, or several: a near- and a far-branch ray, rays turning
    either way round the axis, rays that sweep past pi.  A camera inside a curved window in its least dense medium
    (an air housing, a dome port under water) has exactly one image of every point outside.

    Attributes
    ----------
    point : (K,) int
        The index of the point each row images.
    uv : (K, 2) float
        The pixel; NaN where the ray leaves the camera backwards (or past the lens model's range), and for a ring.
    branch : (K,) str
        "near" (the ray meets the point before its closest approach to the axis or centre), "far" (after it), or
        "ring" (a sphere's axial ring: a cone of rays imaging a circle, for points on the line through the centre
        and the camera).  For a camera inside, "near" rays leave it moving away from the axis or centre and "far"
        rays pass their closest approach first.  A flat window's single ray is labelled "near".
    lam, c_z : (K,) float
        The ray's conserved momenta as it leaves the camera: Lambda = n r dphi/ds (signed; its sign is the sense of
        rotation) and C_z = n dz/ds about the cylinders' axis (0 for spheres).  A ring's row carries the |Lambda|
        that every ray of its cone shares.
    direction : (K, 3) float
        The ray's unit direction at the camera centre, in the world frame (NaN for a ring).
    in_front : (K,) bool
        The ray leaves the camera forwards and has a pixel.
    in_image : (K,) bool
        ... and that pixel lies on the sensor.
    status : (M,) str
        Per point: "ok" (every image was searched for), "outside_model" (the point is not strictly across the window
        from the camera, so nothing was searched), or "capped" (the far-branch search hit its cap and is not
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


def _project_inside(window, be, centre, points, tol):
    """Every image of points outside a curved window, for a camera inside its innermost interface.

    Light paths are reversible: the rays joining two points do not depend on which of them is the camera.  So each
    point is the pinhole of the outside problem whose object point is the camera, solved by the complete search of
    either backend.  Each ray found is traced from the point to the camera, and its direction there, reversed, is
    the camera's line of sight; its momenta change sign and its branch is kept."""
    from .tracing import trace
    n, R = list(window.indices), list(window.radii)
    P = window.to_local(centre)
    X = window.to_local(points)
    solve = be.project_cylinder if window.kind == "cylinder" else be.project_sphere
    keys = ("branch", "dir", "ring", "capped", "lam")
    parts = {k: [] for k in keys}
    point = []
    for i, x in enumerate(X):
        r = solve(n, R, x, P[None], tol, 1)
        m = len(r["branch"])
        point.append(np.full(m, i, np.int64))
        for k in keys:
            parts[k].append(np.asarray(r[k]))
    rows = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
    rows["point"] = np.concatenate(point) if point else np.zeros(0, np.int64)
    rows["ring"] = rows["ring"].astype(bool)
    rows["capped"] = rows["capped"].astype(bool)
    k = len(rows["point"])
    d_at_camera = np.full((k, 3), np.nan)
    good = ~rows["ring"] & np.all(np.isfinite(rows["dir"].reshape(k, 3)), 1) if k else np.zeros(0, bool)
    if good.any():
        origin = np.asarray(points, float)[rows["point"][good]]
        tr = trace(window, origin, window.vectors_to_world(rows["dir"].reshape(k, 3)[good]))
        d_at_camera[good] = -window.vectors_to_local(tr.direction)       # the ray reversed: leaving the camera
    n_cam = n[-1]
    if window.kind == "cylinder":
        lam = n_cam * (P[0] * d_at_camera[:, 1] - P[1] * d_at_camera[:, 0])
        c_z = n_cam * d_at_camera[:, 2]
    else:
        lam = n_cam * np.linalg.norm(np.cross(P, d_at_camera), axis=1)
        c_z = np.zeros(k)
        lam[rows["ring"]] = np.abs(rows["lam"][rows["ring"]])          # a ring's cone: |Lambda| of all its rays
    return {"point": rows["point"], "branch": rows["branch"].astype(np.int8), "lam": lam, "c_z": c_z,
            "dir": d_at_camera, "ring": rows["ring"], "capped": rows["capped"]}


def camera_side(window, centre):
    """"outside" or "inside": where the camera is with respect to the window (inside the innermost interface of a
    curved window).  A ValueError if it is neither, or on an interface or in a wall."""
    if window.outside(centre):
        return "outside"
    if window.kind != "flat" and window.inside(centre):
        return "inside"
    raise ValueError("the camera must be outside the window (beyond its outermost interface, or on the camera's side "
                     "of a flat window) or inside a curved window's innermost interface")


def project(points, camera, window, *, backend=None, tol=1e-9, threads=1):
    """Every image of each point, as `camera` sees it through `window`.

    Parameters
    ----------
    points : (M, 3) or (3,) array_like
        World points.  Those across the window from the camera are projected; the others get the status
        "outside_model" (for a camera outside: not strictly inside the innermost interface, or not beyond a flat
        window's last one; for a camera inside: not strictly outside the outermost interface).
    camera : Camera
        Its centre lies outside the window (beyond its outermost interface, or on the camera's side of a flat
        window), or inside a curved window's innermost interface: a camera in a housing or behind a dome port.
    window : CylindricalWindow, SphericalWindow or FlatWindow
    backend : {"cpp", "python"}, optional
        The compiled core (default when built) or the pure-Python reference; both run the same search.  A flat
        window has one image per point, found in NumPy whatever the backend.
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
    side = camera_side(window, centre)
    ok = np.all(np.isfinite(pts), 1)
    ok[ok] = window.inside(pts[ok]) if side == "outside" else window.outside(pts[ok])
    status = np.where(ok, "ok", "outside_model").astype("<U13")
    idx = np.flatnonzero(ok)

    if window.kind == "flat":
        rows = project_flat(window, centre, pts[idx])
        direction = rows["dir"].copy()
    elif side == "inside":
        if window.kind not in ("cylinder", "sphere"):
            raise TypeError(f"unsupported window {window!r}")
        rows = _project_inside(window, _backend(backend), centre, pts[idx], tol)
        direction = window.vectors_to_world(rows["dir"])
    else:
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
        direction = window.vectors_to_world(rows["dir"])
    ring = rows["ring"]
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
