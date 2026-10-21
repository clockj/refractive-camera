"""Flat windows: one image per point beyond the window, checked by tracing, and the solver's domain edge."""
import numpy as np
import pytest

import refractive_camera as rc
from refractive_camera.flat import solve_gap


def random_flat(rng):
    N = int(rng.integers(2, 5))
    normal = rng.normal(size=3)
    offsets = np.cumsum(rng.uniform(0.005, 0.05, N - 1))
    return rc.FlatWindow(normal, offsets, rng.uniform(1.0, 1.8, N), origin=rng.normal(size=3))


def camera_before(window, rng):
    """A camera 0.1-0.5 in front of the first interface, looking roughly across it."""
    n = window.normal
    base = window.origin + n * (window.offsets[0] - rng.uniform(0.1, 0.5))
    v = rng.normal(scale=0.1, size=3)
    pos = base + v - (v @ n) * n
    target = window.origin + n * (window.offsets[-1] + 0.2) + rng.normal(scale=0.05, size=3)
    return rc.Camera.look_at(pos, target, up=rng.normal(size=3), f=1500.0, size=(1280, 1024))


def points_beyond(window, m, rng, spread=0.3):
    n = window.normal
    lateral = rng.normal(scale=spread, size=(m, 3))
    lateral -= np.outer(lateral @ n, n)
    return window.origin + np.outer(window.offsets[-1] + rng.uniform(0.01, 0.6, m), n) + lateral


def test_every_point_has_one_ray_that_reaches_it():
    rng = np.random.default_rng(41)
    for _ in range(20):
        w = random_flat(rng)
        cam = camera_before(w, rng)
        X = points_beyond(w, 50, rng)
        im = rc.project(X, cam, w)
        assert (im.status == "ok").all() and (im.counts(on_sensor=False) == 1).all()
        tr = rc.trace(w, np.repeat(cam.center[None], len(im), 0), im.direction)
        scale = np.linalg.norm(X - cam.center, axis=1)
        assert np.max(tr.miss_distance(X[im.point]) / scale) < 1e-9
        k = im.in_front
        bp = rc.back_project(cam, w, im.uv[k])
        assert np.max(bp.miss_distance(X[im.point[k]]) / scale[k]) < 1e-9


def test_equal_indices_give_the_pinhole():
    rng = np.random.default_rng(42)
    w = rc.FlatWindow((0, 0, 1), (0.04, 0.05), (1.33, 1.33, 1.33))
    cam = rc.Camera.look_at((0.0, 0.0, -0.3), (0.0, 0.0, 1.0), up=(0, 1, 0), f=1500.0, size=(1280, 1024))
    X = points_beyond(w, 40, rng)
    im = rc.project(X, cam, w)
    np.testing.assert_allclose(im.uv, cam.project_points(X)[0], atol=1e-8)


def test_solver_stays_inside_the_domain():
    """From air into water the first Newton step from Lambda = 0 overshoots past the smallest index for oblique
    rays; the bracket must hold, and the root must satisfy the equation, even 30,000 depths to the side, where
    Lambda is within 1e-10 of its edge (the unknown is the gap to the edge, so f stays accurate there)."""
    n = np.array([1.0, 1.49, 1.33])
    H = np.tile([0.3, 0.01, 0.2], (6, 1))
    L_tot = np.array([0.0, 1e-6, 0.1, 1.0, 10.0, 1e4])
    g, n_min = solve_gap(H, n, L_tot)
    lam = n_min - g
    assert np.all((g > 0) & (g <= 1.0)) and np.all(n_min == 1.0)
    f = np.sum(H * lam[:, None] / np.sqrt((n - n_min[:, None] + g[:, None]) * (n + lam[:, None])), axis=1)
    np.testing.assert_allclose(f, L_tot, rtol=1e-12, atol=1e-15)


def test_outside_the_model_and_invalid_windows():
    w = rc.FlatWindow((0, 0, 1), (0.04, 0.05), (1.0, 1.49, 1.33))
    cam = rc.Camera.look_at((0.0, 0.0, -0.3), (0.0, 0.0, 1.0), up=(0, 1, 0), f=1500.0, size=(1280, 1024))
    im = rc.project([[0, 0, 0.2], [0, 0, 0.045], [0, 0, 0.01]], cam, w)
    assert list(im.status) == ["ok", "outside_model", "outside_model"]
    with pytest.raises(ValueError):
        rc.project([[0, 0, 0.2]], rc.Camera.look_at((0, 0, 0.1), (0, 0, 1), up=(0, 1, 0), f=1.0, size=(2, 2)), w)
    with pytest.raises(ValueError):
        rc.FlatWindow((0, 0, 1), (0.05, 0.04), (1.0, 1.49, 1.33))
    with pytest.raises(ValueError):
        rc.FlatWindow((0, 0, 1), (0.04,), (1.0, 1.49, 1.33))


def test_triangulation_through_a_flat_window():
    rng = np.random.default_rng(43)
    w = rc.FlatWindow((0, 0, 1), (0.04, 0.05), (1.0, 1.49, 1.33))
    cams = [rc.Camera.look_at((x, y, -0.3), (0.0, 0.0, 0.3), up=(0, 1, 0), f=1500.0, size=(1280, 1024))
            for x, y in ((-0.15, 0.0), (0.15, 0.05), (0.0, -0.15))]
    for X in points_beyond(w, 10, rng, spread=0.05):
        uv = np.array([rc.project(X, c, w).uv[0] for c in cams])
        P, _ = rc.triangulate(cams, uv, w, refine=False)
        assert np.linalg.norm(P - X) < 1e-9
