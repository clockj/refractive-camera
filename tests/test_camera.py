import numpy as np
import pytest

import refractive_camera as rc

DIST = (-0.21, 0.07, 1.2e-3, -8e-4, -0.012)


def camera(dist=DIST):
    return rc.Camera([[1500.0, 0.3, 640.2], [0.0, 1490.0, 511.7], [0.0, 0.0, 1.0]], dist=dist, size=(1280, 1024))


def test_undistort_inverts_distort():
    cam = camera()
    x, y = np.meshgrid(np.linspace(-0.45, 0.45, 61), np.linspace(-0.35, 0.35, 41))
    xy = np.stack([x.ravel(), y.ravel()], 1)
    back = cam.undistort(cam.distort(xy))
    assert np.max(np.abs(back - xy)) < 1e-13


def test_distortion_jacobian_matches_finite_differences():
    cam = camera()
    xy = np.random.default_rng(1).uniform(-0.4, 0.4, (50, 2))
    j11, j12, j21, j22 = cam._distort_jacobian(xy)
    h = 1e-7
    dx = (cam.distort(xy + [h, 0]) - cam.distort(xy - [h, 0])) / (2 * h)
    dy = (cam.distort(xy + [0, h]) - cam.distort(xy - [0, h])) / (2 * h)
    np.testing.assert_allclose(np.c_[j11, j21], dx, atol=1e-7)
    np.testing.assert_allclose(np.c_[j12, j22], dy, atol=1e-7)


def test_pixels_and_directions_round_trip():
    cam = camera()
    u, v = np.meshgrid(np.linspace(0, 1279, 33), np.linspace(0, 1023, 27))
    uv = np.stack([u.ravel(), v.ravel()], 1)
    back, in_front = cam.pixels_from_directions(cam.directions_from_pixels(uv))
    assert in_front.all()
    assert np.max(np.abs(back - uv)) < 1e-9


def test_rays_pass_through_the_points_they_image():
    rng = np.random.default_rng(2)
    cam = rc.Camera.look_at((1.0, -2.0, 0.5), (0.1, 0.2, 0.0), up=(0, 0, 1), f=1200.0, size=(800, 600), dist=DIST)
    X = rng.normal(scale=0.3, size=(40, 3))
    uv, in_front = cam.project_points(X)
    o, d = cam.rays(uv[in_front])
    w = X[in_front] - o
    miss = np.linalg.norm(w - np.einsum("ij,ij->i", w, d)[:, None] * d, axis=1)
    assert np.max(miss) < 1e-12


def test_look_at_orientation():
    cam = rc.Camera.look_at((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), up=(0, 0, 1), f=1000.0, size=(641, 481))
    (uv0, uv_up, uv_y), _ = cam.project_points([[0, 0, 0], [0, 0, 0.1], [0, 0.1, 0]])
    np.testing.assert_allclose(uv0, [320.0, 240.0], atol=1e-12)        # the target is the principal point
    assert uv_up[1] < uv0[1] and abs(uv_up[0] - uv0[0]) < 1e-9          # up in the world is up in the image
    # Looking along -x with z up, +y lies to the camera's right
    assert uv_y[0] > uv0[0] and abs(uv_y[1] - uv0[1]) < 1e-9
    np.testing.assert_allclose(cam.center, [2.0, 0.0, 0.0], atol=1e-15)


def test_points_behind_have_no_pixel():
    cam = rc.Camera.look_at((0, 0, 0), (0, 0, 1), up=(0, 1, 0), f=500.0, size=(100, 100))
    uv, in_front = cam.project_points([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    assert list(in_front) == [False, True]
    assert np.isnan(uv[0]).all() and np.isfinite(uv[1]).all()


def test_in_image_bounds():
    cam = rc.Camera([[100.0, 0, 9.5], [0, 100.0, 4.5], [0, 0, 1]], size=(20, 10))
    uv = np.array([[-0.5, 0.0], [-0.5001, 0.0], [19.49, 9.49], [19.5, 0.0], [0.0, 9.5], [np.nan, 0.0]])
    assert list(cam.in_image(uv)) == [True, False, True, False, False, False]


def test_directions_beyond_the_lens_range_have_no_pixel():
    cam = camera(dist=(-0.5, 0.0, 0.0, 0.0, 0.0))     # r (1 - 0.5 r^2) turns back at r^2 = 2/3
    uv, in_front = cam.pixels_from_directions([[0.5, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert in_front.all()
    assert np.isfinite(uv[0]).all() and np.isnan(uv[1]).all()


@pytest.mark.parametrize("bad", [
    dict(K=[[0, 0, 1], [0, 1, 1], [0, 0, 1]]),
    dict(K=[[1, 0, 1], [0, 1, 1], [0, 1, 1]]),
    dict(K=np.eye(3), R=2 * np.eye(3)),
    dict(K=np.eye(3), R=np.diag([1.0, 1.0, -1.0])),
    dict(K=np.eye(3), dist=np.zeros(6)),
])
def test_invalid_cameras_are_rejected(bad):
    with pytest.raises(ValueError):
        rc.Camera(**bad)
