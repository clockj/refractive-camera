"""image_jacobians(): the analytic pixel derivatives against central differences of project()."""
import numpy as np
import pytest

import refractive_camera as rc
from refractive_camera.jacobian import cylinder_terms, image_jacobians


def _rot(w):
    th = np.linalg.norm(w)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _follow(uv, branch, images):
    rows = np.flatnonzero(images.branch == branch)
    return images.uv[rows[np.argmin(np.linalg.norm(images.uv[rows] - uv, axis=1))]]


@pytest.fixture(scope="module")
def setup():
    tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33), axis_point=(0.01, -0.02, 0.0),
                                axis_direction=(0.05, 0.02, 1.0))
    cam = rc.Camera.look_at((0.36, 0.06, 0.03), (0.0, 0.0, 0.0), f=1800.0, size=(1280, 1024), dist=(-0.08, 0.02))
    pts = np.array([[0.020, 0.030, 0.010], [-0.040, -0.020, 0.030], [0.060, 0.010, -0.020]])
    pts = tube.vectors_to_world(pts) + tube.axis_point
    ring = 0.087 * np.c_[np.cos(np.linspace(0, 2 * np.pi, 721)), np.sin(np.linspace(0, 2 * np.pi, 721)),
                         np.zeros(721)]                  # a point the camera sees twice, from a ring near the wall
    ring = tube.vectors_to_world(ring) + tube.axis_point
    twice = ring[np.flatnonzero(rc.project(ring, cam, tube).counts() == 2)]
    pts = np.vstack([pts, twice[len(twice) // 2]])
    return tube, cam, pts, rc.project(pts, cam, tube)


def test_two_images_present(setup):
    _, _, _, im = setup
    assert im.counts()[3] == 2


def test_point_derivative(setup):
    tube, cam, pts, im = setup
    dP, _ = image_jacobians(pts, cam, tube, im)
    h = 1e-7
    for r in range(len(im)):
        i = im.point[r]
        num = np.empty((2, 3))
        for a in range(3):
            e = np.zeros(3)
            e[a] = h
            up = _follow(im.uv[r], im.branch[r], rc.project(pts[i] + e, cam, tube))
            um = _follow(im.uv[r], im.branch[r], rc.project(pts[i] - e, cam, tube))
            num[:, a] = (up - um) / (2 * h)
        assert np.linalg.norm(dP[r] - num) <= 1e-6 * np.linalg.norm(dP[r])


def test_pose_derivative(setup):
    tube, cam, pts, im = setup
    _, dS = image_jacobians(pts, cam, tube, im)
    h = 1e-7
    for r in range(len(im)):
        i = im.point[r]
        num = np.empty((2, 6))
        for a in range(6):
            outs = []
            for s in (h, -h):
                if a < 3:
                    w = np.zeros(3)
                    w[a] = s
                    R = _rot(w) @ cam.R
                    c = cam.center
                else:
                    R, c = cam.R, cam.center.copy()
                    c[a - 3] += s
                moved = rc.Camera(cam.K, R, -R @ c, cam.dist, cam.size)
                outs.append(_follow(im.uv[r], im.branch[r], rc.project(pts[i], moved, tube)))
            num[:, a] = (outs[0] - outs[1]) / (2 * h)
        assert np.linalg.norm(dS[r] - num) <= 1e-6 * np.linalg.norm(dS[r])


def test_terms_match_reference_sweep():
    """cylinder_terms' phi and z equal the reference's Phi_Z on both branches, for either sign of Lambda."""
    from refractive_camera._reference.cylinder import Phi_Z
    n, R = [1.0, 1.49, 1.33], [0.100, 0.094]
    for lam, cz, far in [(0.05, 0.3, False), (-0.05, -0.2, False), (0.03, 0.1, True), (-0.02, 0.4, True)]:
        t = cylinder_terms(np.array(lam), np.array(cz), n, R, 0.35, 0.07, far)
        phi, z = Phi_Z(lam, cz, n, R, 0.35, 0.07, far)
        assert abs(float(t["phi"]) - phi) < 1e-14 and abs(float(t["z"]) - z) < 1e-15


def _cases():
    """(name, window, camera, points) for every window and side that image_jacobians covers."""
    rng = np.random.default_rng(7)
    out = []
    flat = rc.FlatWindow((0.0, 0.0, 1.0), (0.040, 0.050), (1.0, 1.49, 1.33))
    cam = rc.Camera.look_at((0.01, -0.02, 0.0), (0.0, 0.0, 0.5), f=800.0, size=(640, 512), dist=(-0.05, 0.01))
    X = np.c_[rng.uniform(-0.2, 0.2, 4), rng.uniform(-0.2, 0.2, 4), rng.uniform(0.1, 0.8, 4)]
    out.append(("flat", flat, cam, np.vstack([X, [[0.01, -0.02, 0.4]]])))      # the last straight ahead
    ball = rc.SphericalWindow((0.100, 0.094), (1.0, 1.49, 1.33), center=(0.01, 0.0, -0.02))
    cam = rc.Camera.look_at((0.3, 0.1, 0.05), (0, 0, 0), f=1500.0, size=(1280, 1024))
    v = rng.normal(size=(4, 3))
    out.append(("sphere outside", ball, cam, 0.06 * v / np.linalg.norm(v, axis=1, keepdims=True) + ball.center))
    for kind in ("cylinder", "sphere"):
        for idx, rp in (((1.33, 1.49, 1.0), 0.03), ((1.0, 1.49, 1.33), 0.06), ((1.0, 1.49, 1.33), 0.0)):
            if kind == "cylinder":
                win = rc.CylindricalWindow((0.100, 0.094), idx, axis_point=(0.0, 0.01, 0.0),
                                           axis_direction=(0.02, 0.0, 1.0))
                c = win.vectors_to_world(np.array([rp, 0.0, 0.01])) + win.axis_point
            else:
                win = rc.SphericalWindow((0.100, 0.094), idx)
                c = np.array([rp, 0.0, 0.0])
            cam = rc.Camera.look_at(c, c + np.array([1.0, 0.3, 0.1]), f=600.0, size=(1280, 1024))
            r = rng.uniform(0.15, 0.5, 4)
            a = rng.uniform(-0.7, 0.7, 4)
            X = np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-0.2, 0.2, 4)]
            out.append((f"{kind} inside {idx} r_P={rp}", win, cam, X))
    return out


@pytest.mark.parametrize("name,win,cam,pts", _cases(), ids=[c[0] for c in _cases()])
def test_derivatives_every_window(name, win, cam, pts):
    """d(uv)/dX and d(uv)/d(pose) against central differences of project(), for flat windows, spheres seen from
    outside, and cameras inside cylinders and spheres (off centre, and on the axis or centre, where only the
    derivatives that exist in closed form are returned and the others are NaN)."""
    im = rc.project(pts, cam, win)
    dP, dS = image_jacobians(pts, cam, win, im)
    h = 1e-7
    checked = 0
    for r in range(len(im)):
        if not np.all(np.isfinite(im.uv[r])):
            continue
        i = im.point[r]
        ana = np.concatenate([dP[r], dS[r]], -1)
        known = np.all(np.isfinite(ana), 0)
        num = np.full((2, 9), np.nan)
        for a in np.flatnonzero(known):
            outs = []
            for s in (h, -h):
                p, R, c = pts[i].copy(), cam.R, cam.center.copy()
                if a < 3:
                    p[a] += s
                elif a < 6:
                    w = np.zeros(3)
                    w[a - 3] = s
                    R = _rot(w) @ cam.R
                else:
                    c[a - 6] += s
                moved = rc.Camera(cam.K, R, -R @ c, cam.dist, cam.size)
                outs.append(_follow(im.uv[r], im.branch[r], rc.project(p, moved, win)))
            num[:, a] = (outs[0] - outs[1]) / (2 * h)
        assert known[:6].all()                                   # the point's and the rotation's always exist
        assert np.linalg.norm(ana[:, known] - num[:, known]) <= 1e-6 * np.linalg.norm(num[:, known])
        checked += 1
    assert checked >= 3
