"""project(): every image it returns is a real ray (checked by the independent tracer), and none is missing (checked
by a brute-force search that uses tracing alone)."""
import numpy as np
import pytest
from conftest import (BACKENDS, REF_INDICES, REF_RADII, camera_outside, make_window, points_inside,
                      random_rotation, reference_camera)

import refractive_camera as rc
from refractive_camera.tracing import search_images

# Python is about 100 times slower than the compiled core: it gets fewer points.
POINTS = {"cpp": 300, "python": 25}


def traced_miss(window, camera, images, points):
    """Distance from each image's point to its ray, traced from the camera centre, relative to the camera's
    distance from the axis or centre."""
    rays = ~np.isnan(images.direction).any(1)
    tr = rc.trace(window, np.repeat(camera.center[None], rays.sum(), 0), images.direction[rays])
    scale = window.radial_distance(camera.center)
    return tr.miss_distance(points[images.point[rays]]) / scale


def test_every_ray_reaches_its_point(backend, kind):
    rng = np.random.default_rng(11)
    for trial in range(3):
        if trial == 0:
            window, cam = make_window(kind), reference_camera()
        else:                                          # random stacks of 1-3 interfaces, in a random frame
            N = int(rng.integers(2, 5))
            radii = np.sort(rng.uniform(0.02, 0.2, N - 1))[::-1]
            indices = rng.uniform(1.0, 1.8, N)
            frame = (dict(axis_point=rng.normal(size=3), axis_direction=rng.normal(size=3)) if kind == "cylinder"
                     else dict(center=rng.normal(size=3)))
            window = make_window(kind, radii, indices, **frame)
            cam = camera_outside(window, rng)
        X = points_inside(window, POINTS[backend] // 3, rng)
        im = rc.project(X, cam, window, backend=backend)
        assert (im.status == "ok").all()
        assert len(im) > 0
        assert np.max(traced_miss(window, cam, im, X)) < 1e-9
        # and the pixels lead back: the ray each in-front pixel sees passes through the point
        k = im.in_front
        tr = rc.back_project(cam, window, im.uv[k])
        assert np.max(tr.miss_distance(X[im.point[k]]) / window.radial_distance(cam.center)) < 1e-9


@pytest.mark.skipif("cpp" not in BACKENDS, reason="the compiled backend is not built")
def test_backends_agree(kind):
    rng = np.random.default_rng(12)
    window, cam = make_window(kind), reference_camera()
    X = points_inside(window, 60, rng)
    a = rc.project(X, cam, window, backend="cpp")
    b = rc.project(X, cam, window, backend="python")
    assert np.array_equal(a.counts(on_sensor=False), b.counts(on_sensor=False))
    # the order of a point's rays is not part of the result: compare them sorted
    oa, ob = (np.lexsort((im.lam, im.branch, im.point)) for im in (a, b))
    assert np.array_equal(a.point[oa], b.point[ob]) and np.array_equal(a.branch[oa], b.branch[ob])
    np.testing.assert_allclose(a.uv[oa], b.uv[ob], atol=1e-6)


def test_results_do_not_depend_on_unit_or_frame(kind):
    rng = np.random.default_rng(13)
    window, cam = make_window(kind), reference_camera()
    X = points_inside(window, 40, rng)
    base = rc.project(X, cam, window)
    # millimetres
    win_mm = make_window(kind, np.array(REF_RADII) * 1000, REF_INDICES)
    cam_mm = rc.Camera(cam.K, cam.R, cam.t * 1000, size=cam.size)
    mm = rc.project(X * 1000, cam_mm, win_mm)
    assert np.array_equal(base.counts(), mm.counts())
    np.testing.assert_allclose(mm.uv, base.uv, atol=1e-6)
    # a rigid motion of the whole scene
    Q, s = random_rotation(rng), rng.normal(size=3)
    frame = (dict(axis_point=s, axis_direction=Q[:, 2]) if kind == "cylinder" else dict(center=s))
    moved = rc.project(X @ Q.T + s, rc.Camera(cam.K, cam.R @ Q.T, cam.t - cam.R @ Q.T @ s, size=cam.size),
                       make_window(kind, **frame))
    assert np.array_equal(base.counts(), moved.counts())
    np.testing.assert_allclose(moved.uv, base.uv, atol=1e-6)


def test_equal_indices_give_the_pinhole(kind):
    rng = np.random.default_rng(14)
    window = make_window(kind, REF_RADII, (1.0, 1.0, 1.0))
    cam = reference_camera()
    X = points_inside(window, 50, rng)
    im = rc.project(X, cam, window)
    assert (im.counts(on_sensor=False) == 1).all()
    uv, _ = cam.project_points(X)
    np.testing.assert_allclose(im.uv, uv, atol=1e-7)


def test_no_image_is_missed(kind):
    """Against a brute-force search by tracing alone, at random points and at the reference port's shadow and
    two-image examples (z = 0, as the map of the paper's Fig. 14; each has the same count 0.5 mm and 0.5 degree
    around it)."""
    rng = np.random.default_rng(15)
    window, cam = make_window(kind), reference_camera()
    shadow = 0.087 * np.array([np.cos(np.radians(87.5)), np.sin(np.radians(87.5)), 0.0])
    double = 0.087 * np.array([np.cos(np.radians(135.5)), np.sin(np.radians(135.5)), 0.0])
    X = np.vstack([shadow, double, points_inside(window, 18, rng, zspan=0.06)])
    im = rc.project(X, cam, window)
    counts = im.counts(on_sensor=False)
    assert counts[0] == 0 and counts[1] == 2
    for i, x in enumerate(X):
        found = search_images(window, cam.center, x, grid=(600, 400))
        mine = im.direction[im.rows_of(i)]
        assert len(found) == len(mine), f"point {i}: project {len(mine)}, brute force {len(found)}"
        for d in found:
            assert np.min(np.linalg.norm(mine - d, axis=1)) < 1e-7


def test_points_on_the_axis_and_opposite_the_camera(kind):
    window, cam = make_window(kind), reference_camera()
    X = np.array([[0.0, 0.0, 0.0],               # the axis or centre
                  [0.0, 0.0, 0.03],              # on the axis, above (a cylinder only; inside the sphere too)
                  [-0.05, 0.0, 0.0],             # opposite the camera: Delta phi = pi
                  [-0.05, 0.0, 0.02]])
    im = rc.project(X, cam, window)
    assert (im.counts(on_sensor=False) >= 1).all()
    rays = im.branch != "ring"
    assert np.max(traced_miss(window, cam, im, X)) < 1e-9
    if kind == "sphere":
        # (-0.05, 0, 0) lies on the line through the centre and the camera: its rings are flagged, without pixels
        assert np.isnan(im.uv[~rays]).all() and not im.in_front[~rays].any()


def test_outside_the_model():
    window, cam = make_window("cylinder"), reference_camera()
    im = rc.project([[0.0, 0.0, 0.0], [0.097, 0.0, 0.0], [0.5, 0.0, 0.0], [np.nan, 0.0, 0.0]], cam, window)
    assert list(im.status) == ["ok", "outside_model", "outside_model", "outside_model"]
    assert set(im.point) == {0}
    with pytest.raises(ValueError):
        rc.project([[0.0, 0.0, 0.0]], reference_camera(distance=0.099), window)


@pytest.mark.parametrize("radii, indices", [((0.1,), (1.0,)), ((0.1, 0.12), (1.0, 1.5, 1.3)),
                                            ((0.1, 0.09), (1.0, -1.5, 1.3))])
def test_invalid_windows_are_rejected(radii, indices):
    with pytest.raises(ValueError):
        rc.CylindricalWindow(radii, indices)
