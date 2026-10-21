"""refractive_camera.isaac: Isaac Sim and USD cameras.  Pure math, so it is tested without Isaac Sim; what only
rendered frames can settle (the half-pixel convention, whether the renderer honours cx, cy and fx != fy) is left to
the Phase 2 protocol (docs/isaac_validation.md)."""
import numpy as np
import pytest

import refractive_camera as rc
from refractive_camera import isaac


def unit_quaternion(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def is_rotation(R, tol=1e-14):
    return np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol


def random_camera(rng, size=(1280, 720)):
    return rc.Camera.look_at(rng.normal(size=3) * 2.0, rng.normal(size=3) * 0.1, up=rng.normal(size=3),
                             f=rng.uniform(800.0, 3000.0), size=size)


# --------------------------------------------------------------------------------------------------- quaternions
def test_quaternion_convention():
    """Scalar first, active rotation: 90 degrees about +z takes x to y; 180 degrees about +x flips y and z."""
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    np.testing.assert_allclose(isaac.quaternion_to_matrix(q), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_allclose(isaac.quaternion_to_matrix((0.0, 1.0, 0.0, 0.0)), np.diag([1.0, -1.0, -1.0]),
                               atol=1e-15)
    np.testing.assert_allclose(isaac.quaternion_to_matrix((2.0, 0.0, 0.0, 0.0)), np.eye(3), atol=1e-15)


def test_quaternion_round_trip_including_half_turns():
    """Random rotations, and rotations at and next to a half-turn (w ~ 0), where a rotation vector or a division by
    w loses accuracy: matrix -> quaternion -> matrix must stay exact."""
    rng = np.random.default_rng(41)
    qs = [unit_quaternion(rng) for _ in range(300)]
    for _ in range(60):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        for angle in (np.pi, np.pi - 1e-6, np.pi - 1e-9, np.pi - 1e-12, np.pi + 1e-9, 1e-12, 0.0):
            qs.append(np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis])
    qs += [np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0, 0.0])]
    for q in qs:
        R = isaac.quaternion_to_matrix(q)
        assert is_rotation(R)
        q2 = isaac.matrix_to_quaternion(R)
        assert q2[0] >= 0.0
        assert abs(abs(q2 @ (q / np.linalg.norm(q))) - 1.0) < 1e-14          # the same rotation: q2 = +-q
        np.testing.assert_allclose(isaac.quaternion_to_matrix(q2), R, atol=4e-16 * 8)


def test_matrix_to_quaternion_rejects_non_rotations():
    for bad in (np.diag([1.0, 1.0, -1.0]), 2.0 * np.eye(3), np.ones((3, 3))):
        with pytest.raises(ValueError):
            isaac.matrix_to_quaternion(bad)
    with pytest.raises(ValueError):
        isaac.quaternion_to_matrix((0.0, 0.0, 0.0, 0.0))


# --------------------------------------------------------------------------------------------------------- poses
@pytest.mark.parametrize("axes", ["ros", "usd", "world"])
def test_identity_orientation_in_each_convention(axes):
    """With the identity orientation each convention's camera looks and points up where Isaac Sim's documentation
    says: ros +Z forward and +Y down, usd -Z forward and +Y up, world +X forward, +Z up and +Y to the left."""
    forward, up, left = {"ros": ((0, 0, 1), (0, -1, 0), (-1, 0, 0)),
                         "usd": ((0, 0, -1), (0, 1, 0), (-1, 0, 0)),
                         "world": ((1, 0, 0), (0, 0, 1), (0, 1, 0))}[axes]
    K = [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
    cam = isaac.camera_from_pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), axes, K, (641, 481))
    f, u, l = (np.array(v, float) for v in (forward, up, left))
    (c, above, to_left), front = cam.project_points([f, f + 0.1 * u, f + 0.1 * l])
    assert front.all()
    np.testing.assert_allclose(c, [320.0, 240.0], atol=1e-12)
    assert above[1] < c[1] and abs(above[0] - c[0]) < 1e-9
    assert to_left[0] < c[0] and abs(to_left[1] - c[1]) < 1e-9


@pytest.mark.parametrize("axes", ["ros", "usd", "world"])
def test_pose_round_trip(axes):
    rng = np.random.default_rng(42)
    for _ in range(50):
        cam = random_camera(rng)
        position, q = isaac.pose_from_camera(cam, axes)
        back = isaac.camera_from_pose(position, q, axes, cam.K, cam.size)
        np.testing.assert_allclose(back.R, cam.R, atol=1e-14)
        np.testing.assert_allclose(back.center, cam.center, atol=1e-14)
        np.testing.assert_allclose(back.K, cam.K)


def test_the_three_conventions_name_the_same_camera():
    """Built from one optical frame, each convention's quaternion has the documented axes as its columns, and all
    three rebuild the same camera."""
    cam = rc.Camera.look_at((1.0, 2.0, 0.5), (0.0, 0.0, 0.3), up=(0.0, 0.0, 1.0), f=1000.0, size=(640, 480))
    right, down, forward = cam.R                          # rows of R: the OpenCV camera axes in world coordinates
    up, left = -down, -right
    frames = {"ros": np.c_[right, down, forward],         # +X right, +Y down, +Z forward
              "usd": np.c_[right, up, -forward],          # +X right, +Y up, -Z forward
              "world": np.c_[forward, left, up]}          # +X forward, +Y left, +Z up
    for axes, expected in frames.items():
        position, q = isaac.pose_from_camera(cam, axes)
        np.testing.assert_allclose(isaac.quaternion_to_matrix(q), expected, atol=1e-14)
        back = isaac.camera_from_pose(position, q, axes, cam.K, cam.size)
        np.testing.assert_allclose(back.R, cam.R, atol=1e-14)


def test_unknown_axes_are_refused():
    with pytest.raises(ValueError):
        isaac.camera_from_pose((0, 0, 0), (1, 0, 0, 0), "opengl", np.eye(3), (10, 10))


# ---------------------------------------------------------------------------------------------- USD camera prims
def usd_pixels(points, f, ha, va, ho, vo, W, H, M):
    """USD's film-back pinhole, written out on its own.  Local coordinates by the inverse of the row-vector world
    matrix; the film point (f X / -Z, f Y / -Z); the window [-ha/2 + ho, ha/2 + ho] x [-va/2 + vo, va/2 + vo], read
    left to right and top to bottom; OpenCV's pixel centres (the left edge of the image at u = -0.5)."""
    local = np.c_[points, np.ones(len(points))] @ np.linalg.inv(M)
    X, Y, Z = local[:, 0], local[:, 1], local[:, 2]
    x_film, y_film = f * X / -Z, f * Y / -Z
    s = (x_film - (ho - ha / 2)) / ha                       # 0 at the left edge of the window, 1 at the right
    t = ((vo + va / 2) - y_film) / va                       # 0 at the top edge, 1 at the bottom
    return np.c_[s * W - 0.5, t * H - 0.5]


def test_usd_projection_written_out_independently():
    rng = np.random.default_rng(43)
    for _ in range(25):
        W, H = (int(v) for v in rng.integers(320, 4096, 2))
        f, ha, va = rng.uniform(10.0, 60.0), rng.uniform(15.0, 40.0), rng.uniform(10.0, 30.0)
        ho, vo = rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0)
        M = np.eye(4)
        M[:3, :3] = rng.uniform(0.5, 2.0) * isaac.quaternion_to_matrix(unit_quaternion(rng)).T   # with a scale
        M[3, :3] = rng.normal(size=3) * 3.0
        # points in front of the camera and inside its window, made in its local frame
        Z = -rng.uniform(0.5, 10.0, 200)
        X = (rng.uniform(-0.5, 0.5, 200) * ha + ho) / f * -Z
        Y = (rng.uniform(-0.5, 0.5, 200) * va + vo) / f * -Z
        world = (np.c_[X, Y, Z, np.ones(200)] @ M)[:, :3]
        cam = isaac.camera_from_usd(f, ha, va, W, H, M, ho, vo)
        uv, front = cam.project_points(world)
        assert front.all()
        np.testing.assert_allclose(uv, usd_pixels(world, f, ha, va, ho, vo, W, H, M), atol=1e-9)
        assert cam.in_image(uv).all()


def test_usd_attributes_round_trip():
    rng = np.random.default_rng(44)
    for _ in range(30):
        W, H = (int(v) for v in rng.integers(320, 4096, 2))
        base = random_camera(rng)
        K = [[rng.uniform(500.0, 5000.0), 0.0, rng.uniform(0.3, 0.7) * W],
             [0.0, rng.uniform(500.0, 5000.0), rng.uniform(0.3, 0.7) * H], [0.0, 0.0, 1.0]]
        cam = rc.Camera(K, base.R, base.t, size=(W, H))
        usd = isaac.usd_from_camera(cam, focal_length=rng.uniform(10.0, 100.0))
        back = isaac.camera_from_usd(usd["focalLength"], usd["horizontalAperture"], usd["verticalAperture"], W, H,
                                     usd["world_matrix"], usd["horizontalApertureOffset"],
                                     usd["verticalApertureOffset"])
        np.testing.assert_allclose(back.K, cam.K, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(back.R, cam.R, atol=1e-13)
        np.testing.assert_allclose(back.center, cam.center, atol=1e-12)
        # the USD camera's -Z is the optical axis, its +Y is up in the image
        M = usd["world_matrix"]
        np.testing.assert_allclose(-M[2, :3], cam.R[2], atol=1e-14)
        np.testing.assert_allclose(M[1, :3], -cam.R[1], atol=1e-14)


def test_centred_film_back_puts_the_principal_point_at_the_sensor_centre():
    cam = isaac.camera_from_usd(24.0, 36.0, 24.0, 1920, 1280, np.eye(4))
    np.testing.assert_allclose(cam.K[:2, 2], [959.5, 639.5])
    np.testing.assert_allclose([cam.K[0, 0], cam.K[1, 1]], [1280.0, 1280.0])
    # an aperture offset moves the film back: +ho to the right of the axis, so the principal point moves left
    shifted = isaac.camera_from_usd(24.0, 36.0, 24.0, 1920, 1280, np.eye(4), 1.8, 1.2)
    np.testing.assert_allclose(shifted.K[:2, 2], [959.5 - 96.0, 639.5 + 64.0])


def test_world_matrices_with_shear_or_reflection_are_refused():
    shear = np.eye(4)
    shear[1, 0] = 0.3
    flip = np.diag([1.0, 1.0, -1.0, 1.0])
    stretch = np.diag([1.0, 2.0, 1.0, 1.0])
    for M in (shear, flip, stretch):
        with pytest.raises(ValueError):
            isaac.camera_from_usd(24.0, 36.0, 24.0, 640, 480, M)


def test_single_precision_world_matrices_are_accepted():
    rng = np.random.default_rng(45)
    M = np.eye(4)
    M[:3, :3] = isaac.quaternion_to_matrix(unit_quaternion(rng)).T
    M[3, :3] = (1.0, 2.0, 3.0)
    cam = isaac.camera_from_usd(24.0, 36.0, 24.0, 640, 480, M.astype(np.float32))
    np.testing.assert_allclose(cam.R, isaac.camera_from_usd(24.0, 36.0, 24.0, 640, 480, M).R, atol=1e-6)


def test_cameras_without_size_or_with_skew_are_refused():
    with pytest.raises(ValueError):
        isaac.usd_from_camera(rc.Camera(np.eye(3)), 24.0)
    skewed = rc.Camera([[1000.0, 0.5, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]], size=(640, 480))
    with pytest.raises(ValueError):
        isaac.usd_from_camera(skewed, 24.0)
    with pytest.raises(ValueError):
        isaac.opencv_pinhole_properties(skewed)


def test_opencv_pinhole_properties_layout():
    cam = rc.Camera([[1000.0, 0.0, 640.5], [0.0, 990.0, 360.25], [0.0, 0.0, 1.0]],
                    dist=(-0.1, 0.02, 1e-3, -2e-3, 5e-3), size=(1280, 720))
    p = isaac.opencv_pinhole_properties(cam)
    assert (p["cx"], p["cy"], p["fx"], p["fy"]) == (640.5, 360.25, 1000.0, 990.0)
    assert p["pinhole"] == [-0.1, 0.02, 1e-3, -2e-3, 5e-3] + [0.0] * 7
