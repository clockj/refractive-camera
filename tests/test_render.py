import numpy as np
from conftest import make_window, reference_camera

import refractive_camera as rc


def centroid(img, region, background):
    u0, v0, u1, v1 = region
    w = np.clip(background - img, 0, None)
    vv, uu = np.mgrid[v0:v1, u0:u1]
    return np.array([(w * uu).sum() / w.sum(), (w * vv).sum() / w.sum()])


def test_a_small_dot_appears_where_it_projects(kind):
    window, cam = make_window(kind), reference_camera()
    for X in ([0.0, 0.0, 0.0], [0.02, 0.03, 0.01], [-0.03, -0.02, 0.0]):
        im = rc.project(X, cam, window)
        assert im.counts()[0] == 1
        u, v = np.round(im.uv[0]).astype(int)
        region = (u - 12, v - 12, u + 13, v + 13)
        img = rc.render_dots(cam, window, [X], 0.0005, supersample=6, region=region, outside=0.85)
        assert img.min() < 0.1                                  # the dot is there
        assert np.linalg.norm(centroid(img, region, 0.85) - im.uv[0]) < 0.1
