"""A camera inside a curved window: a housing, a dome port, a camera submerged in a tube."""
import numpy as np
import pytest

import refractive_camera as rc
from refractive_camera.tracing import search_images

AIR_INSIDE = (1.33, 1.49, 1.0)           # from the outermost medium inward: water, acrylic, air (a housing)
WATER_INSIDE = (1.0, 1.49, 1.33)         # air, acrylic, water: a camera submerged in a tube or a sphere


def _camera(c, look):
    return rc.Camera.look_at(c, look, f=600.0, size=(1280, 1024))


def _outside_points(rng, m, r_lo=0.11, r_hi=0.6):
    r = rng.uniform(r_lo, r_hi, m)
    a = rng.uniform(-np.pi, np.pi, m)
    return np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-0.3, 0.3, m)]


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_least_dense_camera_sees_every_point_once(kind):
    """A camera in the least dense medium (air in a housing or behind a dome) has exactly one image of every
    outside point; each ray, traced from the camera, reaches its point."""
    rng = np.random.default_rng(1)
    win = (rc.CylindricalWindow if kind == "cylinder" else rc.SphericalWindow)((0.100, 0.094), AIR_INSIDE)
    c = np.array([0.03, 0.01, 0.0])
    X = _outside_points(rng, 60)
    im = rc.project(X, _camera(c, (1.0, 0.3, 0.1)), win)
    assert np.all(im.counts(on_sensor=False) == 1)
    tr = rc.trace(win, np.broadcast_to(c, (len(im), 3)), im.direction)
    assert np.all(tr.ok)
    assert np.max(tr.miss_distance(X[im.point]) / np.linalg.norm(X[im.point], axis=1)) < 1e-10


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
@pytest.mark.parametrize("indices", [AIR_INSIDE, WATER_INSIDE], ids=["air inside", "water inside"])
def test_no_image_is_missed_from_inside(kind, indices):
    """Against a brute-force search by tracing alone from the camera over every direction, at random points and,
    for the submerged camera, at a point hidden by the wall and a point seen twice."""
    rng = np.random.default_rng(6)
    win = (rc.CylindricalWindow if kind == "cylinder" else rc.SphericalWindow)((0.100, 0.094), indices)
    c = np.array([0.085, 0.0, 0.01]) if indices == WATER_INSIDE else np.array([0.03, 0.01, 0.0])
    special = 0.2 * np.array([[np.cos(np.radians(96.0)), np.sin(np.radians(96.0)), 0.05],
                              [np.cos(np.radians(125.0)), np.sin(np.radians(125.0)), 0.05]])
    X = np.vstack([special, _outside_points(rng, 6, r_hi=0.4)])
    im = rc.project(X, _camera(c, (1.0, 0.3, 0.1)), win)
    if indices == WATER_INSIDE:
        assert list(im.counts(on_sensor=False)[:2]) == [0, 2]
    for i, x in enumerate(X):
        found = search_images(win, c, x, grid=(720, 360))
        mine = im.direction[im.rows_of(i)]
        assert len(found) == len(mine), f"point {i}: project {len(mine)}, brute force {len(found)}"
        for d in found:
            assert np.min(np.linalg.norm(mine - d, axis=1)) < 1e-7


def test_submerged_sphere_bands():
    """A camera in water 85 mm from the centre of an acrylic sphere in air: at r_X = 200 mm the shadow spans
    72.48-119.90 deg from the camera's radius and two images 119.90-128.09 deg (reference values from a dense
    sampling of the rays' sweep in their plane)."""
    win = rc.SphericalWindow((0.100, 0.094), WATER_INSIDE)
    c = np.array([0.085, 0.0, 0.0])
    cam = _camera(c, (0.0, 1.0, 0.0))
    want = {50.0: 1, 96.0: 0, 119.0: 0, 121.0: 2, 127.5: 2, 140.0: 1}
    ang = np.radians(list(want))
    X = 0.200 * np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    got = rc.project(X, cam, win).counts(on_sensor=False)
    assert dict(zip(want, got.tolist())) == want


def test_dome_bending():
    """An air camera 30 mm off the centre of an acrylic dome (radii 94, 100 mm) in water bends its view of distant
    points by up to 4.823 deg (reference value from a bisection on the rays' sweep in their plane)."""
    dome = rc.SphericalWindow((0.100, 0.094), AIR_INSIDE)
    c = np.array([0.030, 0.0, 0.0])
    dph = np.radians(np.linspace(0.05, 179.95, 1800))
    X = 1e3 * np.c_[np.cos(dph), np.sin(dph), np.zeros_like(dph)]
    im = rc.project(X, _camera(c, (1.0, 0.5, 0.0)), dome)
    theta = np.arccos(np.clip(im.direction[:, 0], -1, 1))
    assert abs(np.degrees(np.max(theta - dph[im.point])) - 4.823) < 0.002


def test_centred_dome_does_not_bend():
    dome = rc.SphericalWindow((0.100, 0.094), AIR_INSIDE)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3))
    X *= 0.5 / np.linalg.norm(X, axis=1, keepdims=True)
    im = rc.project(X, _camera(np.zeros(3), (1.0, 0.2, 0.0)), dome)
    assert np.max(np.linalg.norm(im.direction - X[im.point] / 0.5, axis=1)) < 1e-8


def test_escape_window():
    """From inside, a ray leaves the window if and only if |C_z| < min n_k and |Lambda| < min_{k>=2} nu_k R_{k-1}
    (media numbered from the camera outward, nu_k = sqrt(n_k^2 - C_z^2)); otherwise an interface reflects it
    back."""
    win = rc.CylindricalWindow((0.100, 0.094), WATER_INSIDE)
    n = np.array(WATER_INSIDE[::-1])                     # from the camera outward
    R = np.array([0.094, 0.100])                         # R_1 < R_2
    c = np.array([0.06, 0.0, 0.0])
    rng = np.random.default_rng(3)
    d = rng.normal(size=(4000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lam = n[0] * (c[0] * d[:, 1] - c[1] * d[:, 0])
    cz = n[0] * d[:, 2]
    nu = np.sqrt(np.maximum(n[None, :] ** 2 - cz[:, None] ** 2, 0.0))
    leaves = (np.abs(cz) < n.min()) & (np.abs(lam) < np.min(nu[:, 1:] * R[None, :], axis=1))
    tr = rc.trace(win, np.broadcast_to(c, d.shape), d)
    assert np.array_equal(tr.ok, leaves)
    assert set(tr.reason[~tr.ok]) <= {"tir"}


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_camera_on_the_axis(kind):
    rng = np.random.default_rng(4)
    win = (rc.CylindricalWindow if kind == "cylinder" else rc.SphericalWindow)((0.100, 0.094), WATER_INSIDE)
    X = _outside_points(rng, 40)
    im = rc.project(X, _camera(np.zeros(3), (1.0, 0.2, 0.1)), win)
    tr = rc.trace(win, np.zeros((len(im), 3)), im.direction)
    assert np.all(tr.ok)
    assert np.max(tr.miss_distance(X[im.point])) < 1e-12


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_threads_and_backends_agree_from_inside(kind):
    """The searches of a camera inside, one per point, give the same images on one thread or several, and the two
    backends the same image counts and rays."""
    rng = np.random.default_rng(8)
    win = (rc.CylindricalWindow if kind == "cylinder" else rc.SphericalWindow)((0.100, 0.094), WATER_INSIDE)
    cam = _camera(np.array([0.085, 0.0, 0.01]), (0.0, 1.0, 0.0))
    X = _outside_points(rng, 40, r_hi=0.4)
    one = rc.project(X, cam, win, threads=1)
    four = rc.project(X, cam, win, threads=4)
    assert np.array_equal(one.point, four.point) and np.array_equal(np.nan_to_num(one.uv), np.nan_to_num(four.uv))
    if "cpp" in rc.available_backends():
        py = rc.project(X, cam, win, backend="python")
        assert np.array_equal(one.counts(on_sensor=False), py.counts(on_sensor=False))
        for i in range(len(X)):
            for d in py.direction[py.rows_of(i)]:
                assert np.min(np.linalg.norm(one.direction[one.rows_of(i)] - d, axis=1)) < 1e-9


def test_points_on_the_camera_side_are_outside_the_model():
    win = rc.CylindricalWindow((0.100, 0.094), AIR_INSIDE)
    im = rc.project(np.array([[0.05, 0.0, 0.0], [0.3, 0.0, 0.0]]), _camera(np.array([0.02, 0, 0]), (1, 0, 0)), win)
    assert list(im.status) == ["outside_model", "ok"]


def test_camera_in_the_wall_is_refused():
    win = rc.CylindricalWindow((0.100, 0.094), AIR_INSIDE)
    with pytest.raises(ValueError):
        rc.project(np.array([[0.3, 0.0, 0.0]]), _camera(np.array([0.097, 0.0, 0.0]), (1, 0, 0)), win)
