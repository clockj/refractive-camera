"""Isaac Sim and USD cameras as `Camera` objects, and back.

Pure math: nothing here imports Isaac Sim, Omniverse or USD, so it runs, and is tested, anywhere.  Inside Isaac Sim,
read the camera's pose or USD attributes there and pass the numbers in.

Conventions
-----------
`Camera` follows OpenCV: camera frame x right, y down, z forward; ``X_cam = R @ X_world + t``; pixel centres at
integer coordinates, the centre of the top-left pixel at (0, 0).

Isaac Sim gives a camera's orientation as a unit quaternion (w, x, y, z) in one of three camera-axis conventions
(https://docs.isaacsim.omniverse.nvidia.com/6.2.0/reference_material/reference_conventions.html):

    "ros"    +Z forward, -Y up (+Y down), +X right: OpenCV's optical frame
    "usd"    -Z forward, +Y up, +X right: the USD camera's own axes
    "world"  +X forward, +Z up, +Y left

USD stores a prim's world transform as a 4x4 matrix in the row-vector convention, ``p_world = p_local @ M``, with
the translation in the last row.  A camera's film back is its focalLength, horizontalAperture, verticalAperture and
the two aperture offsets, all in one unit (Isaac Sim uses tenths of a stage unit); only their ratios matter here.

What Phase 2 must still check against rendered frames
------------------------------------------------------
* The half-pixel convention: whether Isaac Sim's renderer puts the sensor centre at ((W - 1)/2, (H - 1)/2), as
  assumed here (OpenCV's pixel centres), or at (W/2, H/2).
* Whether the renderer honours fx != fy and a principal point off the sensor centre (one report on NVIDIA's forum
  found it using (fx + fy)/2 and ignoring cx, cy unless a lens-distortion model was enabled).
"""
import numpy as np

from .camera import Camera

# _AXES[name] maps a vector's OpenCV camera-frame coordinates to its coordinates in the named camera axes,
# v_axes = A @ v_cv, so that world-from-OpenCV = R(q) @ A for the quaternion q Isaac Sim reports in those axes.
_AXES = {
    "ros": np.eye(3),
    "usd": np.diag([1.0, -1.0, -1.0]),                   # right, up, backward
    "world": np.array([[0.0, 0.0, 1.0],                  # forward
                       [-1.0, 0.0, 0.0],                 # left
                       [0.0, -1.0, 0.0]]),               # up
}


def _axes(name):
    try:
        return _AXES[name]
    except KeyError:
        raise ValueError(f"unknown camera axes {name!r}: use 'ros', 'usd' or 'world'") from None


def _check_rotation(R, tol=1e-9):
    R = np.asarray(R, float)
    if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=tol) or np.linalg.det(R) <= 0:
        raise ValueError("not a rotation matrix")
    return R


# --------------------------------------------------------------------------------------------------- quaternions
def quaternion_to_matrix(q_wxyz):
    """The rotation matrix of a quaternion (w, x, y, z), scalar first, as Isaac Sim and USD write them.

    The quaternion is normalised first.  The matrix is the active rotation: it maps a vector's coordinates in the
    rotated frame to its coordinates in the reference frame (for a camera pose, camera axes to world).
    """
    q = np.asarray(q_wxyz, float)
    norm = np.linalg.norm(q) if q.shape == (4,) else 0.0
    if not norm > 0 or not np.isfinite(norm):
        raise ValueError("a quaternion is four finite numbers, not all zero")
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def matrix_to_quaternion(R):
    """The unit quaternion (w, x, y, z) of a rotation matrix, with w >= 0.

    Shepperd's method: the component of largest magnitude is taken from the diagonal and the others from the
    off-diagonal sums, so the result keeps full accuracy at every angle, half-turns included (where the rotation
    vector, or a method that divides by w, loses it).  q and -q are the same rotation; w >= 0 picks one, and at
    w = 0 exactly the first non-zero component is made positive.
    """
    m = _check_rotation(R)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr >= max(m[0, 0], m[1, 1], m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + tr)                                      # 4 w
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])            # 4 x
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] >= m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])            # 4 y
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])            # 4 z
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q) / np.linalg.norm(q)
    if q[0] < 0.0 or (q[0] == 0.0 and q[np.flatnonzero(q)[0]] < 0.0):
        q = -q
    return q


# ------------------------------------------------------------------------------------------ poses (quaternions)
def camera_from_pose(position, orientation_wxyz, axes, K, size, dist=None):
    """A `Camera` from an Isaac Sim pose: the camera's world position and orientation in the named camera axes.

    Parameters
    ----------
    position : (3,) array_like
        The camera centre, world frame.
    orientation_wxyz : (4,) array_like
        Its orientation (w, x, y, z), as Isaac Sim returns it with ``camera_axes=axes``.
    axes : {"ros", "usd", "world"}
    K : (3, 3) array_like
        Intrinsics in OpenCV's pixel convention.  Isaac Sim's own intrinsic matrix may put the principal point half a
        pixel away (see the module notes).
    size : (int, int)
        Image (width, height).
    dist : array_like, optional
        OpenCV distortion (k1, k2, p1, p2, k3).
    """
    R_world_from_cv = quaternion_to_matrix(orientation_wxyz) @ _axes(axes)
    R = R_world_from_cv.T
    return Camera(K, R, -R @ np.asarray(position, float).reshape(3), dist, size)


def pose_from_camera(camera, axes):
    """The Isaac Sim pose of a `Camera`: (position, quaternion (w, x, y, z)) in the named camera axes, ready for
    e.g. ``set_world_pose(position, orientation, camera_axes=axes)``."""
    R_world_from_axes = camera.R.T @ _axes(axes).T
    return camera.center, matrix_to_quaternion(R_world_from_axes)


# ------------------------------------------------------------------------------------------- USD camera prims
def _rigid_from_usd_matrix(world_matrix):
    """(world-from-local rotation, translation) of a USD row-vector world matrix.

    A uniform scale is removed (it does not change what the camera sees); shear, non-uniform scale or a reflection
    is refused.  The rotation is re-orthonormalised, so a matrix stored in single precision is accepted.
    """
    M = np.asarray(world_matrix, float)
    if M.shape != (4, 4) or not np.allclose(M[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("a USD world matrix is 4 x 4, row-vector convention, last column (0, 0, 0, 1)")
    L = M[:3, :3]                                     # rows: the local axes in world coordinates, scaled
    scale = np.linalg.norm(L, axis=1)
    if not (scale[0] > 0 and np.allclose(scale, scale[0], rtol=1e-6)):
        raise ValueError("the camera's world matrix has a non-uniform scale")
    Rt = L / scale[0]
    if not np.allclose(Rt @ Rt.T, np.eye(3), atol=1e-6) or np.linalg.det(Rt) <= 0:
        raise ValueError("the camera's world matrix has shear or a reflection")
    U, _, Vt = np.linalg.svd(Rt)
    return (U @ Vt).T, M[3, :3].copy()


def camera_from_usd(focal_length, horizontal_aperture, vertical_aperture, width, height, world_matrix,
                    horizontal_aperture_offset=0.0, vertical_aperture_offset=0.0, dist=None):
    """A `Camera` from a USD camera prim's attributes and its world transform.

    The USD camera looks down its local -Z with +Y up.  A point (X, Y, Z) in its local frame lands on the film at
    ``(f X / -Z, f Y / -Z)``, and the film window spans [-ha/2 + ho, ha/2 + ho] across (left to right) and
    [-va/2 + vo, va/2 + vo] up (bottom to top), with f the focal length, ha and va the apertures and ho and vo
    their offsets: USD's GfCamera adds the offsets to the window, so a positive offset moves the film back right
    or up of the optical axis.  In pixels, with OpenCV's pixel centres:

        fx = f W / ha,   cx = (W - 1)/2 - ho W / ha
        fy = f H / va,   cy = (H - 1)/2 + vo H / va

    so the principal point moves by -ho W/ha in u and +vo H/va in v.  This sign follows USD's definition of the
    window (GfCamera); that Isaac Sim's renderer applies the offsets the same way is for Phase 2 to confirm.

    **Half-pixel convention.** The sensor centre is placed at ((W - 1)/2, (H - 1)/2), the centre of the sensor in
    OpenCV's convention, where the top-left pixel's centre is (0, 0).  Whether Isaac Sim's renderer agrees, or
    puts it at (W/2, H/2), is a half-pixel question that Phase 2 must settle against rendered frames.

    Isaac Sim renders square pixels; a verticalAperture other than ha H / W may not be honoured as authored.

    Parameters
    ----------
    focal_length, horizontal_aperture, vertical_aperture : float
        In one unit (Isaac Sim: tenths of a stage unit).
    width, height : int
        Image size in pixels.
    world_matrix : (4, 4) array_like
        The camera prim's world transform, USD row-vector convention (Gf.Matrix4d, as from
        ``UsdGeom.Imageable.ComputeLocalToWorldTransform``).
    horizontal_aperture_offset, vertical_aperture_offset : float
        In the apertures' unit.
    dist : array_like, optional
        OpenCV distortion (k1, k2, p1, p2, k3), which a USD camera does not carry itself.
    """
    f, ha, va = float(focal_length), float(horizontal_aperture), float(vertical_aperture)
    W, H = int(width), int(height)
    if not (f > 0 and ha > 0 and va > 0 and W > 0 and H > 0):
        raise ValueError("focal length, apertures and image size must be positive")
    fx, fy = f * W / ha, f * H / va
    cx = (W - 1) / 2.0 - float(horizontal_aperture_offset) * W / ha
    cy = (H - 1) / 2.0 + float(vertical_aperture_offset) * H / va
    R_world_from_usd, position = _rigid_from_usd_matrix(world_matrix)
    R = (R_world_from_usd @ _AXES["usd"]).T
    return Camera([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], R, -R @ position, dist, (W, H))


def usd_from_camera(camera, focal_length):
    """USD camera attributes and world transform for a `Camera`, the inverse of `camera_from_usd()`.

    Returns a dict with focalLength, horizontalAperture, verticalAperture, horizontalApertureOffset,
    verticalApertureOffset (in the unit of `focal_length`) and world_matrix (4 x 4, USD row-vector convention).  The
    lens distortion is not part of a USD camera: set it with `opencv_pinhole_properties()`.
    """
    if camera.size is None:
        raise ValueError("the camera needs an image size")
    K = camera.K
    if K[0, 1] != 0:
        raise ValueError("a USD camera has no skew")
    W, H = camera.size
    f = float(focal_length)
    if not f > 0:
        raise ValueError("the focal length must be positive")
    ha, va = f * W / K[0, 0], f * H / K[1, 1]
    M = np.eye(4)
    M[:3, :3] = (camera.R.T @ _AXES["usd"].T).T      # rows: the USD camera's local axes in world coordinates
    M[3, :3] = camera.center
    return {
        "focalLength": f,
        "horizontalAperture": ha,
        "verticalAperture": va,
        "horizontalApertureOffset": ((W - 1) / 2.0 - K[0, 2]) * ha / W,
        "verticalApertureOffset": (K[1, 2] - (H - 1) / 2.0) * va / H,
        "world_matrix": M,
    }


def opencv_pinhole_properties(camera):
    """The arguments of Isaac Sim's ``Camera.set_opencv_pinhole_properties(cx, cy, fx, fy, pinhole)`` for a `Camera`.

    ``pinhole`` is [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]; the rational (k4-k6) and thin-prism (s1-s4)
    terms, which `Camera` does not model, are zero.  That Isaac Sim reads cx and cy in OpenCV's pixel convention is
    part of the half-pixel check of Phase 2 (see the module notes).
    """
    K = camera.K
    if K[0, 1] != 0:
        raise ValueError("Isaac Sim's OpenCV lens model has no skew")
    k1, k2, p1, p2, k3 = (float(v) for v in camera.dist)
    return {"cx": float(K[0, 2]), "cy": float(K[1, 2]), "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "pinhole": [k1, k2, p1, p2, k3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
