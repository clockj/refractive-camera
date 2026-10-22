"""The scene renderer: straight rays without refraction (T1), and every hit consistent with project() (T2)."""
import numpy as np

import refractive_camera as rc
from refractive_camera.scene import Box, Cylinder, Disk, NoiseTexture, Sphere, cast, render_scene

TEX = NoiseTexture(0.01, [(0.2, 0.3, 0.4), (0.9, 0.8, 0.7)], spot=(0, 0, 0))


def _objects():
    return [Disk((0, 0, -0.06), 0.09, TEX),
            Sphere((0.0, 0.01, -0.02), 0.03, TEX),
            Box((-0.04, -0.03, -0.04), (0.02, 0.012, 0.01), TEX),
            Cylinder((0.06, 0.03, -0.03), 0.01, 0.03, TEX)]


def _pixels(cam, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([0, 0], cam.size, (n, 2)) - 0.5


def test_no_refraction_is_straight():
    """T1: with every index 1 the hits are those of straight rays from the camera."""
    flat_tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.0, 1.0))
    cam = rc.Camera.look_at((0.35, 0.02, 0.03), (0, 0, -0.02), f=900.0, size=(640, 512))
    uv = _pixels(cam)
    h = cast(cam, flat_tube, uv, _objects())
    o, d = cam.rays(uv)
    t = np.full(len(uv), np.inf)
    for ob in _objects():
        t = np.minimum(t, ob.intersect(o, d, 0.0, np.inf)[0])
    hit = np.isfinite(t)
    assert np.array_equal(hit, h.obj >= 0)
    straight = o[hit] + t[hit, None] * d[hit]
    assert np.max(np.linalg.norm(h.point[hit] - straight, axis=1)) < 1e-12
    assert np.max(np.abs(h.path[hit] - t[hit])) < 1e-12


def test_hits_project_to_their_pixels():
    """T2: every point a pixel's ray meets inside the tube has an image, by project(), at that pixel."""
    tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))
    cam = rc.Camera.look_at((0.33, -0.08, 0.04), (0, 0, -0.03), f=900.0, size=(640, 512), dist=(-0.05, 0.01))
    uv = _pixels(cam, 3000, 1)
    h = cast(cam, tube, uv, _objects())
    m = np.flatnonzero(h.obj >= 0)
    im = rc.project(h.point[m], cam, tube)
    worst = 0.0
    for j, i in enumerate(m):
        rows = im.rows_of(j)
        assert rows.size >= 1
        worst = max(worst, np.min(np.linalg.norm(im.uv[rows] - uv[i], axis=1)))
    assert worst < 1e-6


def test_render_shapes():
    tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))
    cam = rc.Camera.look_at((0.35, 0.0, 0.0), (0, 0, -0.02), f=300.0, size=(64, 48))
    rgb, centre = render_scene(cam, tube, _objects(), supersample=2)
    assert rgb.shape == (48, 64, 3) and centre.obj.shape == (48 * 64,)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0 and (centre.obj >= 0).any()
