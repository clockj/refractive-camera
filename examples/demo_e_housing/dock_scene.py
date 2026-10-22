"""Demo E's underwater scene: a dock marked by a dot board, on a sandy floor among rocks.

The scene is built in the world frame, the dock's: its board at the origin, facing the robot, which comes along +x.
`render` draws the robot camera's view through its housing by tracing every ray through the port
(`refractive_camera.scene.cast`), with the water's haze: a colour fades to the water's along its path in the water.
"""
import numpy as np

import housing as hz
from refractive_camera.scene import Box, Cylinder, NoiseTexture, Sphere, cast, shade

WATER_RGB = np.array([0.03, 0.20, 0.26])
HAZE = 3.5                                      # m: a path this long in the water keeps 1/e of its colour
FLOOR_Z = -0.75                                 # m, the sea floor below the board's centre
LIGHTS = ((np.array([0.15, 0.25, 1.0]), 0.65), (np.array([-0.6, 0.35, 0.45]), 0.25))     # light from the surface


class DotBoard:
    """The dock's target: white dots on a dark board with a light frame, in the board's own frame."""

    def __init__(self, dots, radius, half):
        self.dots, self.r2, self.half = np.asarray(dots, float)[:, :2], radius ** 2, np.asarray(half, float)

    def __call__(self, p):
        p = np.asarray(p, float)
        d2 = np.min(((p[..., None, :2] - self.dots) ** 2).sum(-1), -1)
        grey = np.where(d2 <= self.r2, 0.96, 0.08)
        edge = (np.abs(p[..., 0]) > self.half[0] - 0.018) | (np.abs(p[..., 1]) > self.half[1] - 0.018)
        grey = np.where(edge, 0.80, grey)
        return grey[..., None] * np.array([1.0, 0.96, 0.86])


def dock_rotation():
    """The board's axes in the world (columns): x to the robot's right (-y), y down (-z), z away from the robot."""
    return np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def objects_world(dots, dot_radius=0.024):
    """The scene's objects in the world frame; the board first."""
    Rb = dock_rotation()
    half = (0.235, 0.155, 0.006)
    sand = [(0.40, 0.36, 0.28), (0.55, 0.50, 0.39), (0.70, 0.64, 0.50)]
    rock = [(0.12, 0.12, 0.11), (0.30, 0.29, 0.25), (0.50, 0.48, 0.42)]
    weed = [(0.05, 0.18, 0.06), (0.18, 0.45, 0.15), (0.55, 0.75, 0.30)]
    rust = [(0.25, 0.10, 0.04), (0.60, 0.28, 0.10), (0.90, 0.62, 0.30)]
    steel = [(0.30, 0.32, 0.34), (0.55, 0.58, 0.60), (0.80, 0.82, 0.84)]
    up = np.eye(3)
    objs = [
        Box((0.0, 0.0, 0.0), half, DotBoard(dots, dot_radius, half), rotation=Rb),
        Cylinder((0.004, -0.17, (FLOOR_Z - 0.155) / 2), 0.015, (-FLOOR_Z - 0.155) / 2,
                 NoiseTexture(0.03, steel, seed=21), rotation=up),
        Cylinder((0.004, 0.17, (FLOOR_Z - 0.155) / 2), 0.015, (-FLOOR_Z - 0.155) / 2,
                 NoiseTexture(0.03, steel, seed=22), rotation=up),
        Box((0.05, 0.0, FLOOR_Z + 0.02), (0.25, 0.32, 0.02), NoiseTexture(0.05, rust, spot=(0.1, 0.05, 0.02),
                                                                         seed=23)),
        Box((-1.6, 0.0, FLOOR_Z - 0.05), (3.0, 3.0, 0.05), NoiseTexture(0.18, sand, spot=(0.30, 0.27, 0.21),
                                                                        spot_scale=0.04, octaves=5, seed=24)),
    ]
    rng = np.random.default_rng(5)
    for k in range(30):
        x, y = rng.uniform(-3.6, 0.8), rng.uniform(-2.0, 2.0)
        if abs(y) < 0.45 and x > -3.4:
            y += np.sign(y + 1e-9) * 0.45                              # keep the robot's path clear
        r = rng.uniform(0.05, 0.20)
        objs.append(Sphere((x, y, FLOOR_Z - 0.35 * r), r,
                           NoiseTexture(0.04, rock if k % 3 else weed, spot=(0.9, 0.85, 0.7), seed=30 + k)))
    return objs


def objects_in_housing(objs, R_wh, t_wh):
    """The same objects in the housing frame of a robot at pose (R_wh, t_wh) (housing -> world)."""
    out = []
    for ob in objs:
        new = object.__new__(type(ob))
        new.__dict__.update(ob.__dict__)
        new.center = R_wh.T @ (ob.center - t_wh)
        new.rotation = R_wh.T @ ob.rotation
        out.append(new)
    return out


def render(cam, window, objs, R_wh, scale=0.5, supersample=2, chunk=1 << 18):
    """The camera's view through its housing, at `scale` times its resolution: (H, W, 3) in [0, 1].

    `objs` and the camera are in the housing frame; R_wh turns the housing frame into the world's (for the light)."""
    s = float(scale)
    K = cam.K.copy()
    K[0, 0], K[1, 1] = K[0, 0] * s, K[1, 1] * s
    K[0, 2], K[1, 2] = (K[0, 2] + 0.5) * s - 0.5, (K[1, 2] + 0.5) * s - 0.5
    W, H = int(round(hz.W * s)), int(round(hz.H * s))
    small = hz.rc.Camera(K, cam.R, cam.t, None, (W, H))
    lights = tuple((R_wh.T @ d, p) for d, p in LIGHTS)
    S = int(supersample)
    off = (np.arange(S) + 0.5) / S - 0.5
    V, U = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    centres = np.c_[U.ravel(), V.ravel()]
    grid = np.stack(np.meshgrid(off, off, indexing="xy"), -1).reshape(1, -1, 2)
    rgb = np.zeros((H * W, 3))
    step = max(chunk // (S * S), 1)
    for k in range(0, len(centres), step):
        uv = (centres[k:k + step, None, :] + grid).reshape(-1, 2)
        hits = cast(small, window, uv, inside=objs)
        col = shade(objs, hits, background=WATER_RGB, lights=lights)
        _, d = small.rays(uv)
        up = (R_wh @ d.T).T[:, 2]                                     # brighter water towards the surface
        water = WATER_RGB * (0.75 + 0.9 * np.clip(up, 0.0, 1.0))[:, None]
        fade = np.exp(-np.nan_to_num(hits.path, posinf=1e9) / HAZE)[:, None]
        col = np.where((hits.obj >= 0)[:, None], col * fade + water * (1 - fade), water)
        rgb[k:k + step] = col.reshape(-1, S * S, 3).mean(1)
    return np.clip(rgb.reshape(H, W, 3), 0.0, 1.0)
