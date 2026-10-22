"""Derivatives of the images: how each image's pixel moves with the point and with the camera.

For a cylinder, the ray of an image on branch b has momenta (Lambda, C_z) that solve

    Phi_b(Lambda, C_z; r_P, r_X) = phi_X - phi_P  (mod 2 pi),        Z_b(Lambda, C_z; r_P, r_X) = z_X - z_P,

with Lambda signed (the sense of rotation about the axis).  Differentiating both equations (the implicit-function
theorem) gives d(Lambda, C_z) = -J^-1 dG, where J = d(Phi, Z)/d(Lambda, C_z) is the paper's Jacobian and dG collects
the motion of the point and of the camera.  The pixel then follows from the ray's direction at the camera, which is a
closed form of (Lambda, C_z) and the camera's position.  Every partial derivative used here is checked with SymPy.

`cylinder_terms` evaluates the closed forms on arrays of any module with NumPy's names (NumPy itself, or PyTorch), so
the same formulas serve bundle adjustment and differentiable rendering.
"""
import numpy as np


def cylinder_terms(lam, cz, n, R, rP, rX, far, xp=np):
    """The sweep Phi, the axial offset Z and their partial derivatives, on one branch, for signed Lambda.

    Parameters
    ----------
    lam, cz : arrays of one shape
        The momenta Lambda (signed) and C_z.
    n : sequence of N floats
        Refractive indices from the camera inward.
    R : sequence of N - 1 floats
        Interface radii, outermost first.
    rP, rX : arrays broadcastable to ``lam``
        Distances of the camera and of the point from the axis.
    far : bool array broadcastable to ``lam``
        The far branch (the ray meets the point after its closest approach to the axis).
    xp : module
        ``numpy`` or ``torch``.

    Returns
    -------
    dict of arrays
        ``phi``, ``z`` and the partials ``phi_lam``, ``phi_c``, ``z_lam``, ``z_c``, ``phi_rp``, ``z_rp``,
        ``phi_rx``, ``z_rx``.  On the far branch ``phi`` is the reference's sweep (pi - a1 - a2 in the last medium),
        equal to the odd extension in Lambda up to a multiple of 2 pi.
    """
    N = len(n)
    radii = [rP] + list(R) + [rX]
    out = dict.fromkeys(("phi", "z", "phi_lam", "phi_c", "z_c"), 0.0)

    def parts(k, r):
        nu2 = n[k] ** 2 - cz * cz
        w = xp.sqrt(xp.clip(nu2 * r * r - lam * lam, 0.0, None))
        q = 1.0 / w
        a = xp.arcsin(xp.clip(lam / (xp.sqrt(nu2) * r), -1.0, 1.0))
        t = cz / nu2 * w
        dt_c = w / nu2 + 2.0 * cz * cz * w / (nu2 * nu2) - cz * cz * r * r * q / nu2
        return nu2, a, q, t, dt_c

    for k in range(N):
        nu2, a_i, q_i, t_i, dtc_i = parts(k, radii[k + 1])
        _, a_o, q_o, t_o, dtc_o = parts(k, radii[k])
        out["phi"] = out["phi"] + a_i - a_o
        out["z"] = out["z"] + t_o - t_i
        out["phi_lam"] = out["phi_lam"] + q_i - q_o
        out["phi_c"] = out["phi_c"] + lam * cz / nu2 * (q_i - q_o)
        out["z_c"] = out["z_c"] + dtc_o - dtc_i
    nu2_1, _, q_p, _, _ = parts(0, rP)
    nu2_N, a_x, q_x, t_x, dtc_x = parts(N - 1, rX)
    out["phi_rp"] = lam / rP * q_p
    out["z_rp"] = cz * rP * q_p
    out["phi_rx"] = -lam / rX * q_x
    out["z_rx"] = -cz * rX * q_x
    f = float(far) if isinstance(far, (bool, np.bool_)) else xp.where(far, 1.0, 0.0)
    out["phi"] = out["phi"] + f * (np.pi - 2.0 * a_x)
    out["z"] = out["z"] + f * 2.0 * t_x
    out["phi_lam"] = out["phi_lam"] - f * 2.0 * q_x
    out["phi_c"] = out["phi_c"] - f * 2.0 * lam * cz / nu2_N * q_x
    out["z_c"] = out["z_c"] + f * 2.0 * dtc_x
    out["phi_rx"] = out["phi_rx"] + f * 2.0 * lam / rX * q_x
    out["z_rx"] = out["z_rx"] + f * 2.0 * cz * rX * q_x
    out["z_lam"] = out["phi_c"]                                  # J is symmetric (the paper's eq. symmetry)
    return out


def _skew(v):
    """Cross-product matrices [v]x of vectors (..., 3)."""
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                     np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def pixel_jacobian(camera, d_cam):
    """d(uv)/d(d_cam) (..., 2, 3) of `camera.pixels_from_directions` at camera-frame directions (..., 3)."""
    d = np.asarray(d_cam, float)
    zc = d[..., 2]
    xy = d[..., :2] / zc[..., None]
    j11, j12, j21, j22 = camera._distort_jacobian(xy)
    K = camera.K
    a = np.stack([np.stack([K[0, 0] * j11 + K[0, 1] * j21, K[0, 0] * j12 + K[0, 1] * j22], -1),
                  np.stack([K[1, 1] * j21, K[1, 1] * j22], -1)], -2)             # d(uv)/d(normalised, distorted)
    inv = 1.0 / zc
    zero = np.zeros_like(zc)
    p = np.stack([np.stack([inv, zero, -xy[..., 0] * inv], -1),
                  np.stack([zero, inv, -xy[..., 1] * inv], -1)], -2)           # d(x/z, y/z)/d(d_cam)
    return a @ p


def image_jacobians(points, camera, window, images):
    """How each image's pixel moves with its point and with the camera.

    Parameters
    ----------
    points : (M, 3) array_like
        The world points given to `project`.
    camera : Camera
    window : CylindricalWindow
        Cylinders only, for now.
    images : Images
        ``project(points, camera, window)``.

    Returns
    -------
    d_point : (K, 2, 3) ndarray
        d(u, v)/dX for every row of `images`, world frame.
    d_pose : (K, 2, 6) ndarray
        d(u, v)/d(omega, c): omega rotates the camera, ``R -> expm([omega]x) R`` (camera frame), and c is its centre
        in the world frame.

    Rows on a caustic (det J = 0) are infinite there, and NaN for rays without a pixel.
    """
    if window.kind != "cylinder":
        raise NotImplementedError("image_jacobians supports cylindrical windows for now")
    pts = np.asarray(points, float).reshape(-1, 3)
    W = window.vectors_to_world(np.eye(3)).T                    # columns: the window's axes in the world frame
    P = window.to_local(camera.center)
    X = window.to_local(pts[images.point])
    lam, cz = images.lam, images.c_z
    rP, phP = np.hypot(P[0], P[1]), np.arctan2(P[1], P[0])
    rX = np.hypot(X[:, 0], X[:, 1])
    n = list(window.indices)
    t = cylinder_terms(lam, cz, n, list(window.radii), rP, rX, images.branch == "far")
    J = np.stack([np.stack([t["phi_lam"], t["phi_c"]], -1), np.stack([t["z_lam"], t["z_c"]], -1)], -2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Jinv = np.linalg.inv(J)

    # dG/dX_local for G = (Phi - phi_X + phi_P, Z - z_X + z_P)
    zero = np.zeros_like(rX)
    drX = np.stack([X[:, 0] / rX, X[:, 1] / rX, zero], -1)
    dphX = np.stack([-X[:, 1] / rX ** 2, X[:, 0] / rX ** 2, zero], -1)
    dzX = np.broadcast_to([0.0, 0.0, 1.0], X.shape)
    GX = np.stack([t["phi_rx"][:, None] * drX - dphX, t["z_rx"][:, None] * drX - dzX], -2)
    drP = np.array([P[0] / rP, P[1] / rP, 0.0])
    dphP = np.array([-P[1] / rP ** 2, P[0] / rP ** 2, 0.0])
    dzP = np.array([0.0, 0.0, 1.0])
    GP = np.stack([t["phi_rp"][:, None] * drP + dphP, t["z_rp"][:, None] * drP + dzP], -2)

    # The direction at the camera, D = d_r rhat + d_phi phat + d_z zhat (local frame), and its partials
    n1 = n[0]
    w1 = np.sqrt(np.maximum((n1 * n1 - cz * cz) * rP * rP - lam * lam, 0.0))
    q1 = 1.0 / w1
    d_r, d_ph = -w1 / (n1 * rP), lam / (n1 * rP)
    rh = np.array([np.cos(phP), np.sin(phP), 0.0])
    ph = np.array([-np.sin(phP), np.cos(phP), 0.0])
    zh = np.array([0.0, 0.0, 1.0])
    dD_dlam = (lam * q1 / (n1 * rP))[:, None] * rh + (1.0 / (n1 * rP)) * ph
    dD_dc = (cz * rP * q1 / n1)[:, None] * rh + zh / n1
    dD_drp = (-lam * lam * q1 / (n1 * rP ** 2))[:, None] * rh + (-lam / (n1 * rP ** 2))[:, None] * ph
    dD_dphp = d_r[:, None] * ph - d_ph[:, None] * rh
    dD_dLC = np.stack([dD_dlam, dD_dc], -1)                      # (K, 3, 2)

    dDl_dXl = dD_dLC @ (-Jinv @ GX)                              # (K, 3, 3)
    dDl_dPl = dD_dLC @ (-Jinv @ GP) + dD_drp[:, :, None] * drP + dD_dphp[:, :, None] * dphP

    d_cam = images.direction @ camera.R.T
    A = pixel_jacobian(camera, d_cam) @ camera.R @ W             # d(uv)/d(D local)
    d_point = A @ dDl_dXl @ W.T
    d_c = A @ dDl_dPl @ W.T
    d_omega = pixel_jacobian(camera, d_cam) @ -_skew(d_cam)
    return d_point, np.concatenate([d_omega, d_c], -1)
