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

    Every case is the implicit-function derivative of the projection's own equation, with closed-form partial
    derivatives; the direction at the camera then turns into a pixel.
    * cylinders, camera outside: (Phi, Z)(Lambda, C_z) = (azimuth, height) of the point, a 2 x 2 system;
    * cylinders, camera inside: the same system for the ray reversed, from the point to the camera (light paths
      are reversible);
    * spheres, either side: Phi(Lambda) = the point's angle in the ray's plane; a motion out of that plane turns the
      plane about the camera's or the point's radius;
    * flat windows: f(Lambda) = sum_i H_i Lambda / sqrt(n_i^2 - Lambda^2) = the lateral offset.

    Parameters
    ----------
    points : (M, 3) array_like
        The world points given to `project`.
    camera : Camera
    window : CylindricalWindow, SphericalWindow or FlatWindow
    images : Images
        ``project(points, camera, window)``.

    Returns
    -------
    d_point : (K, 2, 3) ndarray
        d(u, v)/dX for every row of `images`, world frame.
    d_pose : (K, 2, 6) ndarray
        d(u, v)/d(omega, c): omega rotates the camera, ``R -> expm([omega]x) R`` (camera frame), and c is its centre
        in the world frame.

    Rows on a caustic (det J = 0) are infinite there, and NaN for rays without a pixel and for a sphere's ring.  A
    camera on a cylinder's axis or at a sphere's centre (within 1e-6 of the innermost radius) takes the limit r_P -> 0
    of the inside formulas, its own motion off the axis or centre included.
    """
    from .projection import camera_side
    pts = np.asarray(points, float).reshape(-1, 3)
    if window.kind == "flat":
        dDX, dDP = _flat_direction_jacobians(window, camera.center, pts[images.point], images.lam)
    else:
        side = camera_side(window, camera.center)
        P = window.to_local(camera.center)
        X = window.to_local(pts[images.point])
        Dl = window.vectors_to_local(images.direction)
        if window.kind == "cylinder" and side == "outside":
            dDX, dDP = _cylinder_outside(window, P, X, images)
        elif window.kind == "cylinder":
            dDX, dDP = _cylinder_inside(window, P, X, images, Dl)
        elif window.kind == "sphere":
            dDX, dDP = _sphere(window, P, X, images, Dl, side)
        else:
            raise TypeError(f"unsupported window {window!r}")
        W = window.vectors_to_world(np.eye(3)).T                # columns: the window's axes in the world frame
        dDX, dDP = _frame_to_world(dDX, W), _frame_to_world(dDP, W)
    d_cam = images.direction @ camera.R.T
    A = pixel_jacobian(camera, d_cam) @ camera.R                # d(uv)/d(direction, world)
    d_omega = pixel_jacobian(camera, d_cam) @ -_skew(d_cam)
    return A @ dDX, np.concatenate([d_omega, A @ dDP], -1)


def _frame_to_world(dD, W):
    """d(direction)/d(position) (K, 3, 3) from the window frame to the world's, W the window's axes as columns.  A
    window-frame column that is unknown (NaN) makes unknown only the world columns it enters."""
    unknown = np.isnan(dD).any(axis=1)                              # (K, 3): per row, the local columns unknown
    world_unknown = (unknown.astype(float) @ (np.abs(W) > 1e-15).T.astype(float)) > 0
    out = W @ np.nan_to_num(dD, nan=0.0, posinf=np.inf, neginf=-np.inf) @ W.T
    out[np.broadcast_to(world_unknown[:, None, :], out.shape)] = np.nan
    return out


def _cylinder_outside(window, P, X, images):
    """d(direction)/dX and d(direction)/d(camera centre), window frame, for a camera outside coaxial cylinders."""
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
    return dDl_dXl, dDl_dPl


def _direction_at_inside_camera(lam, cz, n_c, rP, phP, s):
    """The direction at a camera inside coaxial cylinders, D = d_r rhat + d_phi phat + d_z zhat (window frame), from
    its momenta, s = +1 for a ray leaving outward (near branch), -1 inward (far), and its partial derivatives in
    Lambda, C_z, r_P and phi_P."""
    w = np.sqrt(np.maximum((n_c * n_c - cz * cz) * rP * rP - lam * lam, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 1.0 / w
    d_r, d_ph = s * w / (n_c * rP), lam / (n_c * rP)
    rh = np.array([np.cos(phP), np.sin(phP), 0.0])
    ph = np.array([-np.sin(phP), np.cos(phP), 0.0])
    zh = np.array([0.0, 0.0, 1.0])
    dD_dlam = (-s * lam * q / (n_c * rP))[:, None] * rh + (1.0 / (n_c * rP)) * ph
    dD_dc = (-s * cz * rP * q / n_c)[:, None] * rh + zh / n_c
    dD_drp = (s * lam * lam * q / (n_c * rP ** 2))[:, None] * rh + (-lam / (n_c * rP ** 2))[:, None] * ph
    dD_dphp = d_r[:, None] * ph - d_ph[:, None] * rh
    return dD_dlam, dD_dc, dD_drp, dD_dphp


def _cylinder_inside(window, P, X, images, Dl):
    """d(direction)/dX and d(direction)/d(camera centre), window frame, for a camera inside coaxial cylinders.

    The ray reversed runs from the point (the pinhole of the exchanged problem) to the camera, with momenta
    (-Lambda, -C_z) and the same branch, and solves G' = (Phi' - phi_P + phi_X, Z' - z_P + z_X) = 0 with the outside
    closed forms; the camera's momenta are minus the exchanged ones."""
    n, R = list(window.indices), list(window.radii)
    lam, cz = images.lam, images.c_z
    far = images.branch == "far"
    rP, phP = np.hypot(P[0], P[1]), np.arctan2(P[1], P[0])
    rX = np.hypot(X[:, 0], X[:, 1])
    if rP <= NEAR_CENTRE * R[-1]:
        return _cylinder_on_axis(window, P, X, images, Dl)
    t = cylinder_terms(-lam, -cz, n, R, rX, rP, far)               # exchanged: pinhole radius r_X, point radius r_P
    J = np.stack([np.stack([t["phi_lam"], t["phi_c"]], -1), np.stack([t["z_lam"], t["z_c"]], -1)], -2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Jinv = np.linalg.inv(J)
        zero = np.zeros_like(rX)
        drX = np.stack([X[:, 0] / rX, X[:, 1] / rX, zero], -1)
        dphX = np.stack([-X[:, 1] / rX ** 2, X[:, 0] / rX ** 2, zero], -1)
        drP = np.array([P[0] / rP, P[1] / rP, 0.0])
        dphP = np.array([-P[1] / rP ** 2, P[0] / rP ** 2, 0.0])
    dzX = np.broadcast_to([0.0, 0.0, 1.0], X.shape)
    dzP = np.array([0.0, 0.0, 1.0])
    GX = np.stack([t["phi_rp"][:, None] * drX + dphX, t["z_rp"][:, None] * drX + dzX], -2)   # X: exchanged pinhole
    GP = np.stack([t["phi_rx"][:, None] * drP - dphP, t["z_rx"][:, None] * drP - dzP], -2)   # P: exchanged point
    dLC_dX, dLC_dP = Jinv @ GX, Jinv @ GP                          # d(Lambda, C_z) = -d(exchanged) = +Jinv dG'
    rh = np.array([np.cos(phP), np.sin(phP), 0.0])
    s = np.sign(Dl @ rh)
    dD_dlam, dD_dc, dD_drp, dD_dphp = _direction_at_inside_camera(lam, cz, window.indices[-1], rP, phP, s)
    dD_dLC = np.stack([dD_dlam, dD_dc], -1)
    return dD_dLC @ dLC_dX, dD_dLC @ dLC_dP + dD_drp[:, :, None] * drP + dD_dphp[:, :, None] * dphP


# A camera this close to the axis or centre (relative to the innermost radius) takes the limit r_P -> 0: its error is
# O(r_P) there, below the rounding error of the general formulas, which divide by r_P (both about 1e-6 at the switch).
NEAR_CENTRE = 1e-6


def _sweep_coefficient(nu, R, rX):
    """T in the sweep of a ray from a camera next to the axis or centre, Phi = beta + Lambda T + O(Lambda^3): media
    from the camera outward (nu_1 the camera's), R innermost first, rX (K,) the point's distance."""
    T = sum((1.0 / R[j]) * (1.0 / nu[..., j + 1] - 1.0 / nu[..., j]) for j in range(len(R)))
    return T - 1.0 / (nu[..., -1] * rX)


def _cylinder_on_axis(window, P, X, images, Dl):
    """A camera on the cylinders' axis: Lambda = 0, and each ray stays in the plane through the axis and its point,
    where the interfaces are parallel lines: a flat stack.  In that plane f(C) = sum_k H_k C / sqrt(n_k^2 - C^2) =
    |z_X - z_P|, with H_k the media's radial thicknesses from the axis out; a motion of the point out of the plane
    turns the plane about the axis.

    The camera's own motion off the axis is the limit r_P -> 0 of the inside formulas.  At a camera a distance r_P
    from the axis, a ray leaving at the angle beta from the camera's radial direction has Lambda = nu_1 r_P sin(beta)
    and sweeps Phi = beta + Lambda T + O(r_P^3), and the transverse chord of the first medium shortens by
    r_P cos(beta).  So a motion of the camera across the point's plane (along phat) turns the ray by
    (nu_1^2 T / n_1) phat phat^T, and a motion within it (along rhat) changes C as the first medium's thickness does
    in the flat stack."""
    n = np.array(window.indices[::-1])                              # from the camera's medium outward
    R = np.array(window.radii[::-1])                                # innermost first
    rX = np.hypot(X[:, 0], X[:, 1])
    zero = np.zeros_like(rX)
    rh = np.stack([X[:, 0] / rX, X[:, 1] / rX, zero], -1)
    ph = np.stack([-X[:, 1] / rX, X[:, 0] / rX, zero], -1)
    zh = np.array([0.0, 0.0, 1.0])
    H = np.empty((len(X), len(n)))
    H[:, 0] = R[0]
    H[:, 1:-1] = np.diff(R)[None, :]
    H[:, -1] = rX - R[-1]
    dz = X[:, 2] - P[2]
    e = np.where(dz[:, None] >= 0, zh, -zh)
    C = np.abs(images.c_z)
    root = np.sqrt(n[None, :] ** 2 - C[:, None] ** 2)
    fp = np.sum(H * n[None, :] ** 2 / root ** 3, axis=1)
    dC_dX = (e - (C / root[:, -1])[:, None] * rh) / fp[:, None]
    g = (-(C / root[:, 0])[:, None] * rh + e) / n[0]                    # dD/dC in the plane
    dDX = (g[:, :, None] * dC_dX[:, None, :]
           + (np.einsum("ij,ij->i", Dl, rh) / rX)[:, None, None] * (ph[:, :, None] * ph[:, None, :]))
    T = _sweep_coefficient(root, R, rX)
    dDP = ((root[:, 0] ** 2 * T / n[0])[:, None, None] * (ph[:, :, None] * ph[:, None, :])     # across the plane
           + (C / (root[:, 0] * fp))[:, None, None] * (g[:, :, None] * rh[:, None, :])      # within it, radially
           + (-e[:, 2] / fp)[:, None, None] * (g[:, :, None] * zh[None, None, :]))           # along the axis
    return dDX, dDP


def _sphere(window, P, X, images, Dl, side):
    """d(direction)/dX and d(direction)/d(camera centre), window frame, for concentric spheres, either side.

    The ray stays in the plane of the centre, the camera and the point.  With e1 the camera's radial direction, e2
    the ray's tangential direction at the camera and m = e1 x e2 the plane's normal, the ray leaves the camera at
    D = d_r e1 + d_t e2, d_t = Lambda / (n_c r_P), and solves Phi(Lambda; r_P, r_X) = Phi_X, the point's angle from e1
    in the ray's sense.  A motion of the point out of the plane turns the plane about e1; a motion of the camera out
    of it turns the plane about the point's radius.  Phi is the cylinder's sweep at C_z = 0, for a camera inside on
    the ray reversed (light paths are reversible)."""
    n, R = list(window.indices), list(window.radii)
    far = images.branch == "far"
    rP = np.linalg.norm(P)
    rX = np.linalg.norm(X, axis=1)
    Xh = X / rX[:, None]
    if rP <= NEAR_CENTRE * R[-1]:                                     # a centred camera: every ray radial, undeviated
        # A camera moved off the centre by d sees X along a ray that leaves it at the angle alpha from d, with
        # Lambda = n_1 |d| sin(alpha) and alpha + Lambda T = the angle of X from d: the direction turns by
        # n_1 T (I - Xh Xh^T) d (T from the media, camera outward)
        proj = np.eye(3)[None] - Xh[:, :, None] * Xh[:, None, :]
        nu = np.broadcast_to(np.array(n[::-1], float), (len(rX), len(n)))
        T = _sweep_coefficient(nu, np.array(R[::-1], float), rX)
        return proj / rX[:, None, None], (n[-1] * T)[:, None, None] * proj
    e1 = P / rP
    d_r = Dl @ e1
    Dt = Dl - d_r[:, None] * e1
    d_t = np.linalg.norm(Dt, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e2 = Dt / d_t[:, None]
    m = np.cross(e1, e2)
    cF, sF = Xh @ e1, np.einsum("ij,ij->i", Xh, e2)
    n_c = n[0] if side == "outside" else n[-1]
    lam = n_c * rP * d_t
    zero = np.zeros_like(lam)
    if side == "outside":
        t = cylinder_terms(lam, zero, n, R, rP, rX, far)
        F_lam, F_rP, F_rX = t["phi_lam"], t["phi_rp"], t["phi_rx"]
    else:
        t = cylinder_terms(lam, zero, n, R, rX, rP, far)            # the ray reversed: pinhole r_X, point r_P
        F_lam, F_rP, F_rX = t["phi_lam"], t["phi_rx"], t["phi_rp"]
    s = np.sign(d_r)
    w = np.sqrt(np.maximum(n_c * n_c * rP * rP - lam * lam, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        dDr_dlam, dDt_dlam = -s * lam / (w * n_c * rP), np.full_like(lam, 1.0 / (n_c * rP))
        dDr_drp, dDt_drp = s * lam * lam / (w * n_c * rP ** 2), -lam / (n_c * rP ** 2)
        e_perpX = -sF[:, None] * e1 + cF[:, None] * e2                 # d(Xh)/d(Phi_X), in the plane
        dlam_dX = (e_perpX / rX[:, None] - F_rX[:, None] * Xh) / F_lam[:, None]
        dlam_dP = (-e2 / rP - F_rP[:, None] * e1) / F_lam[:, None]
        dD_dlam = dDr_dlam[:, None] * e1 + dDt_dlam[:, None] * e2
        dD_drp = dDr_drp[:, None] * e1 + dDt_drp[:, None] * e2
        mm = m[:, :, None] * m[:, None, :]
        dDX = dD_dlam[:, :, None] * dlam_dX[:, None, :] + (d_t / (rX * sF))[:, None, None] * mm
        dDP = (dD_dlam[:, :, None] * dlam_dP[:, None, :] + dD_drp[:, :, None] * e1[None, None, :]
               + (d_r[:, None] * e2 - d_t[:, None] * e1)[:, :, None] * (e2 / rP)[:, None, :]
               + ((d_r - d_t * cF / sF) / rP)[:, None, None] * mm)
    return dDX, dDP


def _flat_direction_jacobians(window, centre, X, lam):
    """d(direction)/dX and d(direction)/d(camera centre), world frame, for a flat window.

    f(Lambda) = sum_i H_i Lambda / sqrt(n_i^2 - Lambda^2) = L_tot, with H_1 the camera's distance to the first
    interface, H_N the point's beyond the last, and L_tot the lateral offset |w|, w the part of X - c along the
    planes; the direction is D = (sqrt(n_1^2 - Lambda^2) nhat + Lambda e) / n_1 with e = w / |w|."""
    nrm = window.normal
    n = np.array(window.indices)
    off = np.array(window.offsets)
    K = len(X)
    H = np.empty((K, len(n)))
    H[:, 0] = off[0] - window.height(centre)
    H[:, 1:-1] = np.diff(off)[None, :]
    H[:, -1] = window.height(X) - off[-1]
    w = X - centre
    w = w - np.outer(w @ nrm, nrm)
    L = np.linalg.norm(w, axis=1)
    # e along the lateral offset; for a point straight ahead any lateral direction gives the same derivative
    other = np.eye(3)[int(np.argmin(np.abs(nrm)))]
    e0 = other - (other @ nrm) * nrm
    e0 /= np.linalg.norm(e0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(L[:, None] > 0, w / L[:, None], e0)
    root = np.sqrt(n[None, :] ** 2 - lam[:, None] ** 2)
    fp = np.sum(H * n[None, :] ** 2 / root ** 3, axis=1)                  # df/dLambda
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(L > 0, lam / L, 1.0 / fp)                     # Lambda / L_tot, and its limit at L_tot = 0
    dlam_dX = (e - (lam / root[:, -1])[:, None] * nrm) / fp[:, None]
    dlam_dP = (-e + (lam / root[:, 0])[:, None] * nrm) / fp[:, None]
    n1, c = n[0], root[:, 0]
    g = (-(lam / c)[:, None] * nrm + e) / n1                           # dD/dLambda
    Pp = np.eye(3)[None] - e[:, :, None] * e[:, None, :] - np.outer(nrm, nrm)[None]
    dDX = g[:, :, None] * dlam_dX[:, None, :] + (ratio / n1)[:, None, None] * Pp
    dDP = g[:, :, None] * dlam_dP[:, None, :] - (ratio / n1)[:, None, None] * Pp
    return dDX, dDP
