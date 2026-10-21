import numpy as np
from conftest import make_window

import refractive_camera as rc


def rays_into(window, rng, m, spread=0.25):
    """Rays from a point outside the window towards points near its axis or centre."""
    origin = np.array([0.4, 0.05, 0.03])
    target = rng.normal(scale=spread * window.radii[0], size=(m, 3))
    return np.repeat(origin[None], m, 0), target - origin


def test_snell_conserves_the_invariants(kind):
    """Across radial normals, C_z = n d_z and Lambda = n (x d_y - y d_x) (cylinder), or n x cross d (sphere), are
    the same in every medium: the paper's conserved quantities, here from the vector form of Snell's law."""
    rng = np.random.default_rng(21)
    window = make_window(kind, (0.1, 0.094, 0.05), (1.0, 1.49, 1.33, 1.6))
    o, d = rays_into(window, rng, 400)
    tr = rc.trace(window, o, d)
    assert tr.ok.sum() > 100
    n = np.array(window.indices)
    for k in range(len(window.radii)):
        h = tr.hits[tr.ok, k]
        before, after = n[k] * tr.segments[tr.ok, k], n[k + 1] * tr.segments[tr.ok, k + 1]
        if kind == "cylinder":
            np.testing.assert_allclose(after[:, 2], before[:, 2], atol=1e-12)
            lam = lambda v: h[:, 0] * v[:, 1] - h[:, 1] * v[:, 0]       # noqa: E731
            np.testing.assert_allclose(lam(after), lam(before), atol=1e-12)
        else:
            np.testing.assert_allclose(np.cross(h, after), np.cross(h, before), atol=1e-12)
        # every hit lies on its interface
        r = np.hypot(h[:, 0], h[:, 1]) if kind == "cylinder" else np.linalg.norm(h, axis=1)
        np.testing.assert_allclose(r, window.radii[k], rtol=1e-12)


def test_reflection_and_misses_are_reported():
    # From water through glass into air: rays inclined enough are totally reflected at the glass-air interface
    window = rc.CylindricalWindow((0.1, 0.09), (1.33, 1.5, 1.0))
    a = np.radians(np.linspace(0, 30, 61))
    d = np.c_[-np.cos(a), np.sin(a), np.zeros_like(a)]
    tr = rc.trace(window, np.repeat([[0.3, 0.0, 0.0]], len(a), 0), d)
    assert tr.ok[0] and not tr.ok[-1]
    assert set(tr.reason[~tr.ok]) <= {"tir", "miss"} and "tir" in set(tr.reason)
    # a ray pointing away from the window misses it
    away = rc.trace(window, [0.3, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert not away.ok[0] and away.reason[0] == "miss"


def test_paths_stay_inside_the_innermost_interface(kind):
    rng = np.random.default_rng(22)
    window = make_window(kind)
    o, d = rays_into(window, rng, 300)
    tr = rc.trace(window, o, d)
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = tr.entry[tr.ok] + (s * tr.length[tr.ok])[:, None] * tr.direction[tr.ok]
        assert np.all(window.radial_distance(p) <= window.radii[-1] * (1 + 1e-12))
