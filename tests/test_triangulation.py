import numpy as np
from conftest import make_window, points_inside

import refractive_camera as rc


def rig(window):
    """Three cameras around the tube, at 0, 80 and 160 degrees, looking at its axis."""
    cams = []
    for az, h in ((0, 0.0), (80, 0.05), (160, -0.04)):
        a = np.radians(az)
        cams.append(rc.Camera.look_at((0.4 * np.cos(a), 0.4 * np.sin(a), h), (0, 0, 0), f=2200.0, size=(1280, 1024)))
    return cams


def observations(cams, window, X):
    """Every on-sensor image of X in every camera."""
    cam_list, uv = [], []
    for c in cams:
        im = rc.project(X, c, window)
        for k in np.flatnonzero(im.in_image):
            cam_list.append(c)
            uv.append(im.uv[k])
    return cam_list, np.array(uv)


def test_exact_pixels_give_the_point(kind):
    rng = np.random.default_rng(31)
    window = make_window(kind)
    cams = rig(window)
    for X in points_inside(window, 15, rng, zspan=0.04):
        cl, uv = observations(cams, window, X)
        if len(uv) < 2:
            continue
        P, info = rc.triangulate(cl, uv, window, refine=False)
        assert np.linalg.norm(P - X) < 1e-9


def test_refinement_reduces_the_reprojection_error():
    rng = np.random.default_rng(32)
    window = make_window("cylinder")
    cams = rig(window)
    gains, errors = [], []
    for X in points_inside(window, 12, rng, zspan=0.04):
        cl, uv = observations(cams, window, X)
        if len(uv) < 3:
            continue
        noisy = uv + rng.normal(scale=0.5, size=uv.shape)
        P0, i0 = rc.triangulate(cl, noisy, window, refine=False)
        P1, i1 = rc.triangulate(cl, noisy, window, refine=True)
        gains.append(i0["rms_px"] - i1["rms_px"])
        errors.append(np.linalg.norm(P1 - X))
    assert len(gains) >= 5
    assert min(gains) >= -1e-9                      # never worse
    assert max(errors) < 1e-3                       # half a pixel of noise: well under a millimetre here
