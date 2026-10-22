"""The scene of demos B and C: textured stones in the paper's reference tube, some of them close to the wall.

The tube is acrylic (outer radius 100 mm, wall 6 mm, n = 1.49) filled with water (n = 1.33), seen from air.  Lengths
are in metres.  The stones are spheres, boxes and capped cylinders with solid noise textures, on a gravel floor.
Several reach within 10 mm of the inner wall, where a camera sees some of their points twice and others not at all.
"""
import numpy as np

import refractive_camera as rc
from refractive_camera.scene import Box, Cylinder, Disk, NoiseTexture, Sphere

TUBE = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))
FLOOR_Z = -0.075


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])


def objects():
    """The scene's objects, all inside the water."""
    warm = [(0.35, 0.16, 0.08), (0.80, 0.45, 0.18), (0.98, 0.86, 0.62)]
    sea = [(0.05, 0.18, 0.35), (0.15, 0.55, 0.75), (0.85, 0.95, 0.98)]
    moss = [(0.10, 0.25, 0.08), (0.45, 0.65, 0.20), (0.92, 0.95, 0.55)]
    plum = [(0.28, 0.06, 0.25), (0.75, 0.30, 0.55), (0.98, 0.80, 0.85)]
    sand = [(0.30, 0.27, 0.22), (0.62, 0.56, 0.45), (0.90, 0.86, 0.74)]
    teal = [(0.02, 0.22, 0.20), (0.15, 0.62, 0.55), (0.80, 0.98, 0.90)]
    return [
        Disk((0.0, 0.0, FLOOR_Z), 0.0925, NoiseTexture(0.012, sand, spot=(0.10, 0.08, 0.06), seed=1)),
        Sphere((0.0, 0.005, -0.040), 0.032, NoiseTexture(0.020, warm, spot=(0.10, 0.04, 0.02), seed=2)),
        Sphere((-0.050, 0.036, -0.055), 0.018, NoiseTexture(0.012, sea, spot=(0.02, 0.05, 0.10), seed=3)),
        Sphere((0.046, -0.050, -0.060), 0.014, NoiseTexture(0.010, moss, spot=(0.95, 0.90, 0.30), seed=4)),
        Sphere((0.0, 0.077, -0.062), 0.012, NoiseTexture(0.008, plum, spot=(1.0, 0.95, 0.6), seed=5)),
        Sphere((0.022, 0.026, 0.012), 0.016, NoiseTexture(0.010, teal, spot=(0.95, 0.35, 0.10), seed=6)),
        Box((-0.034, -0.040, -0.064), (0.024, 0.014, 0.011), NoiseTexture(0.012, plum, spot=(0.98, 0.98, 0.9),
                                                                       seed=7), rotation=_rz(30)),
        Box((0.034, -0.012, -0.068), (0.032, 0.007, 0.004), NoiseTexture(0.010, warm, spot=(0.05, 0.05, 0.05),
                                                                        seed=8), rotation=_rz(-55) @ _rx(8)),
        Cylinder((0.068, 0.035, -0.035), 0.010, 0.040, NoiseTexture(0.010, teal, spot=(0.98, 0.85, 0.2), seed=9)),
        Cylinder((-0.076, -0.020, -0.045), 0.008, 0.030, NoiseTexture(0.008, sea, spot=(0.98, 0.5, 0.2), seed=10)),
        Cylinder((-0.020, -0.066, -0.040), 0.009, 0.035, NoiseTexture(0.009, moss, spot=(0.2, 0.1, 0.4), seed=11),
                 rotation=_rx(12)),
    ]


def max_radius(objs):
    """Largest distance from the axis reached by any object (a check that all stay inside the water)."""
    r = 0.0
    for ob in objs:
        c = np.hypot(*ob.center[:2])
        if isinstance(ob, Sphere):
            r = max(r, c + ob.radius)
        elif isinstance(ob, Box):
            r = max(r, c + np.linalg.norm(ob.half))
        elif isinstance(ob, Cylinder):
            r = max(r, c + np.hypot(ob.radius, ob.half_height * np.linalg.norm(ob.rotation[:2, 2])))
        elif isinstance(ob, Disk):
            r = max(r, c + ob.radius)
    return r


def ring_cameras(n_per_ring=20, rings=((-0.045, 0.34), (0.0, 0.40), (0.05, 0.36)), f=1500.0, size=(1280, 1024),
                 aim=(0.0, 0.0, -0.035), jitter=0.004, seed=0, phase=0.0):
    """Cameras on rings around the tube, (height, distance from the axis) per ring, looking near the axis."""
    rng = np.random.default_rng(seed)
    cams = []
    for j, (z, rad) in enumerate(rings):
        for i in range(n_per_ring):
            a = 2 * np.pi * (i + 0.5 * (j % 2)) / n_per_ring + phase
            pos = np.array([rad * np.cos(a), rad * np.sin(a), z])
            tgt = np.asarray(aim, float) + rng.normal(0, jitter, 3)
            cams.append(rc.Camera.look_at(pos, tgt, f=f, size=size))
    return cams
