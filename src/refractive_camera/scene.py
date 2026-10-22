"""Textured scenes seen through a window, rendered by tracing: synthetic images with exact ground truth.

Each pixel's rays are traced from the camera through the window by vector Snell's law (`trace`), and their straight
path in the innermost medium is intersected with the objects placed there.  Objects may also stand in the camera's own
medium, outside the window; a ray meets those before it reaches the window or, when it misses the window, anywhere
along its straight path.  The renderer uses none of the projection's closed forms, so its images can test them.

Shading is view-independent: a solid texture (the colour at a point of the object, in the object's own frame) times
an ambient term plus Lambertian light from fixed directions.  It is meant for geometry: feature detectors, matching,
splatting, tracking.  It has no shadows, reflections or brightness changes at the interfaces.
"""
from dataclasses import dataclass

import numpy as np

from .tracing import trace


# ---- textures ----------------------------------------------------------------------------------------------------

def _hash01(ix, iy, iz, seed):
    """Pseudo-random numbers in [0, 1) from integer lattice coordinates."""
    with np.errstate(over="ignore"):                    # the products wrap modulo 2^64, as a hash should
        h = (ix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ^ iy.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
             ^ iz.astype(np.uint64) * np.uint64(0x165667B19E3779F9)
             ^ np.uint64((int(seed) * 0x27D4EB2F165667C5) & 0xFFFFFFFFFFFFFFFF))
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def value_noise(p, seed=0):
    """Smooth 3-D value noise in [0, 1] at points (..., 3)."""
    p = np.asarray(p, float)
    i = np.floor(p)
    f = p - i
    s = f * f * (3.0 - 2.0 * f)
    i = i.astype(np.int64)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((s[..., 0] if dx else 1 - s[..., 0]) * (s[..., 1] if dy else 1 - s[..., 1])
                     * (s[..., 2] if dz else 1 - s[..., 2]))
                out = out + w * _hash01(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


class NoiseTexture:
    """A solid texture: fractal value noise through a colour palette, with spots of another colour.

    Parameters
    ----------
    scale : float
        Size of the largest features, in the scene's length unit.
    colors : (k, 3) array_like
        Palette, RGB in [0, 1], spread evenly over the noise's range.
    spot : (3,) array_like, optional
        Colour of the spots, which give feature detectors corners and blobs; none by default.
    spot_scale : float, optional
        Size of the spots (default scale / 3).
    octaves : int
    seed : int
    """

    def __init__(self, scale, colors, spot=None, spot_scale=None, octaves=4, seed=0):
        self.scale, self.colors = float(scale), np.asarray(colors, float).reshape(-1, 3)
        self.spot = None if spot is None else np.asarray(spot, float)
        self.spot_scale = self.scale / 3 if spot_scale is None else float(spot_scale)
        self.octaves, self.seed = int(octaves), int(seed)

    def __call__(self, p):
        p = np.asarray(p, float) / self.scale
        v, a, tot = 0.0, 1.0, 0.0
        for o in range(self.octaves):
            v = v + a * value_noise(p * 2.0 ** o, self.seed + o)
            tot += a
            a *= 0.5
        v = np.clip((v / tot - 0.5) * 2.2 + 0.5, 0.0, 1.0)              # stretch the narrow middle of the sum
        x = v * (len(self.colors) - 1)
        k = np.minimum(np.floor(x).astype(int), len(self.colors) - 2) if len(self.colors) > 1 else np.zeros_like(x, int)
        t = (x - k)[..., None] if len(self.colors) > 1 else 0.0
        c = self.colors[k] * (1 - t) + self.colors[np.minimum(k + 1, len(self.colors) - 1)] * t
        if self.spot is not None:
            m = value_noise(np.asarray(p) * (self.scale / self.spot_scale), self.seed + 101)
            w = np.clip((m - 0.62) / 0.05, 0.0, 1.0)[..., None]
            c = c * (1 - w) + self.spot * w
        return c


class Uniform:
    """A plain colour."""

    def __init__(self, color):
        self.color = np.asarray(color, float)

    def __call__(self, p):
        return np.broadcast_to(self.color, np.shape(p)[:-1] + (3,)).copy()


# ---- objects -----------------------------------------------------------------------------------------------------

def _frame(rotation):
    return np.eye(3) if rotation is None else np.asarray(rotation, float)


class _Object:
    """An object with its own frame: X_world = rotation @ x_local + center."""

    def __init__(self, center, rotation, texture):
        self.center = np.asarray(center, float).reshape(3)
        self.rotation = _frame(rotation)
        self.texture = texture

    def _local(self, o, d):
        return (o - self.center) @ self.rotation, d @ self.rotation

    def color(self, p):
        return self.texture((np.asarray(p, float) - self.center) @ self.rotation)


class Sphere(_Object):
    def __init__(self, center, radius, texture, rotation=None):
        super().__init__(center, rotation, texture)
        self.radius = float(radius)

    def intersect(self, o, d, tmin, tmax):
        """Nearest hit of rays o + t d (unit d) with tmin < t < tmax: t (inf if none) and the world normal."""
        ol, dl = self._local(o, d)
        b = np.einsum("ij,ij->i", ol, dl)
        c = np.einsum("ij,ij->i", ol, ol) - self.radius ** 2
        disc = b * b - c
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = np.where(-b - sq > tmin, -b - sq, -b + sq)
        t = np.where((disc >= 0) & (t > tmin) & (t < tmax), t, np.inf)
        nl = ol + np.where(np.isfinite(t), t, 0.0)[:, None] * dl
        return t, (nl / self.radius) @ self.rotation.T


class Box(_Object):
    def __init__(self, center, half_sizes, texture, rotation=None):
        super().__init__(center, rotation, texture)
        self.half = np.asarray(half_sizes, float).reshape(3)

    def intersect(self, o, d, tmin, tmax):
        ol, dl = self._local(o, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-self.half - ol) / dl
            t2 = (self.half - ol) / dl
        lo, hi = np.fmin(t1, t2), np.fmax(t1, t2)
        tn, tf = np.max(lo, 1), np.min(hi, 1)
        t = np.where(tn > tmin, tn, tf)
        ok = (tn <= tf) & (t > tmin) & (t < tmax)
        t = np.where(ok, t, np.inf)
        axis = np.where(tn > tmin, np.argmax(lo, 1), np.argmin(hi, 1))
        p = ol + np.where(ok, t, 0.0)[:, None] * dl
        nl = np.zeros_like(ol)
        nl[np.arange(len(ol)), axis] = np.sign(p[np.arange(len(ol)), axis])
        return t, nl @ self.rotation.T


class Cylinder(_Object):
    """A capped cylinder along its local z axis."""

    def __init__(self, center, radius, half_height, texture, rotation=None):
        super().__init__(center, rotation, texture)
        self.radius, self.half_height = float(radius), float(half_height)

    def intersect(self, o, d, tmin, tmax):
        ol, dl = self._local(o, d)
        a = dl[:, 0] ** 2 + dl[:, 1] ** 2
        b = ol[:, 0] * dl[:, 0] + ol[:, 1] * dl[:, 1]
        c = ol[:, 0] ** 2 + ol[:, 1] ** 2 - self.radius ** 2
        disc = b * b - a * c
        sq = np.sqrt(np.maximum(disc, 0.0))
        best = np.full(len(ol), np.inf)
        normal = np.zeros_like(ol)
        with np.errstate(divide="ignore", invalid="ignore"):
            for ts in ((-b - sq) / a, (-b + sq) / a):
                z = ol[:, 2] + ts * dl[:, 2]
                ok = (disc >= 0) & (a > 0) & (ts > tmin) & (ts < tmax) & (np.abs(z) <= self.half_height) & (ts < best)
                best = np.where(ok, ts, best)
                p = ol + np.nan_to_num(ts)[:, None] * dl
                normal = np.where(ok[:, None], np.c_[p[:, :2] / self.radius, np.zeros(len(ol))], normal)
            for zc in (-self.half_height, self.half_height):
                ts = (zc - ol[:, 2]) / dl[:, 2]
                p = ol + np.nan_to_num(ts)[:, None] * dl
                ok = (ts > tmin) & (ts < tmax) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= self.radius ** 2) & (ts < best)
                best = np.where(ok, ts, best)
                normal = np.where(ok[:, None], np.array([0.0, 0.0, np.sign(zc)]), normal)
        return best, normal @ self.rotation.T


class Disk(_Object):
    """A flat disk in its local xy plane, seen from both sides."""

    def __init__(self, center, radius, texture, rotation=None):
        super().__init__(center, rotation, texture)
        self.radius = float(radius)

    def intersect(self, o, d, tmin, tmax):
        ol, dl = self._local(o, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -ol[:, 2] / dl[:, 2]
        p = ol + np.nan_to_num(t)[:, None] * dl
        ok = (t > tmin) & (t < tmax) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= self.radius ** 2)
        nl = np.zeros_like(ol)
        nl[:, 2] = np.where(dl[:, 2] > 0, -1.0, 1.0)
        return np.where(ok, t, np.inf), nl @ self.rotation.T


# ---- rendering ---------------------------------------------------------------------------------------------------

@dataclass
class Hits:
    """What the rays through given pixels meet.

    obj : (L,) int
        Index into ``inside + outside`` of the object met, -1 for none.
    point, normal : (L, 3)
        Where, and the object's outward normal there (NaN for none).
    path : (L,)
        Geometric length of the ray from the camera to the hit (inf for none).
    inner : (L,) bool
        The ray reached the window's innermost medium.
    """
    obj: np.ndarray
    point: np.ndarray
    normal: np.ndarray
    path: np.ndarray
    inner: np.ndarray


def _nearest(objects, o, d, tmin, tmax, ids):
    t = np.full(len(o), np.inf)
    nrm = np.full((len(o), 3), np.nan)
    obj = np.full(len(o), -1)
    for k, ob in zip(ids, objects):
        tk, nk = ob.intersect(o, d, tmin, tmax)
        better = tk < t
        t = np.where(better, tk, t)
        nrm = np.where(better[:, None], nk, nrm)
        obj = np.where(better, k, obj)
    return t, nrm, obj


def cast(camera, window, uv, inside=(), outside=()):
    """Trace the rays through pixels (L, 2) and find the objects they meet first.

    `inside` objects stand in the window's innermost medium, `outside` ones in the camera's medium.
    """
    inside, outside = list(inside), list(outside)
    origins, dirs = camera.rays(np.atleast_2d(np.asarray(uv, float)))
    L = len(origins)
    tr = trace(window, origins, dirs)
    first = tr.hits[:, 0] if tr.hits.shape[1] else tr.entry
    reach = np.where(np.isfinite(first).all(1), np.linalg.norm(first - origins, axis=1), np.inf)
    # outside objects, before the window (or anywhere along the ray if it misses the window)
    ids_out = range(len(inside), len(inside) + len(outside))
    t_o, n_o, k_o = _nearest(outside, origins, dirs, 0.0, reach, ids_out)
    obj, point, normal, path = k_o, origins + np.where(np.isfinite(t_o), t_o, np.nan)[:, None] * dirs, n_o, t_o
    # inside objects, along the ray's straight path in the innermost medium
    go = np.flatnonzero(tr.ok & ~np.isfinite(t_o))
    if go.size and inside:
        e, dd = tr.entry[go], tr.direction[go]
        t_i, n_i, k_i = _nearest(inside, e, dd, 0.0, tr.length[go], range(len(inside)))
        hit = np.isfinite(t_i)
        g = go[hit]
        obj[g], normal[g] = k_i[hit], n_i[hit]
        point[g] = e[hit] + t_i[hit, None] * dd[hit]
        pts = np.concatenate([origins[g, None], tr.hits[g], point[g, None]], 1)
        seg = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
        path[g] = seg
    return Hits(obj=obj, point=point, normal=normal, path=path, inner=tr.ok)


DEFAULT_LIGHTS = ((np.array([0.35, -0.5, 0.8]), 0.55), (np.array([-0.6, 0.4, 0.2]), 0.25))


def shade(objects, hits, *, background=(0.08, 0.08, 0.1), ambient=0.35, lights=DEFAULT_LIGHTS):
    """RGB in [0, 1] of hits: the texture times ambient plus Lambertian light from fixed world directions."""
    rgb = np.broadcast_to(np.asarray(background, float), (len(hits.obj), 3)).copy()
    for k, ob in enumerate(objects):
        m = hits.obj == k
        if not m.any():
            continue
        lum = np.full(m.sum(), ambient)
        for direction, power in lights:
            l = np.asarray(direction, float) / np.linalg.norm(direction)
            lum += power * np.abs(hits.normal[m] @ l)                 # two-sided: disks and thin parts
        rgb[m] = np.clip(ob.color(hits.point[m]) * lum[:, None], 0.0, 1.0)
    return rgb


def render_scene(camera, window, inside=(), outside=(), *, supersample=2, region=None, chunk=1 << 18, **shading):
    """Render objects behind (and in front of) the window as the camera sees them.

    Parameters
    ----------
    camera : Camera
    window : CylindricalWindow, SphericalWindow or FlatWindow
    inside, outside : sequences of objects (Sphere, Box, Cylinder, Disk)
        In the window's innermost medium, and in the camera's medium.
    supersample : int
        Rays per pixel along each axis, for the colour.
    region : (u0, v0, u1, v1), optional
        Pixel bounds, u0 <= u < u1 and v0 <= v < v1; the whole image by default.
    **shading
        ``background``, ``ambient``, ``lights``: see `shade`.

    Returns
    -------
    rgb : (H, W, 3) float array
    centre : Hits
        Of the ray through each pixel's centre, flattened row by row (index v * W + u within the region).
    """
    if region is None:
        if camera.size is None:
            raise ValueError("give `region`, or a camera with an image size")
        region = (0, 0, camera.size[0], camera.size[1])
    u0, v0, u1, v1 = (int(v) for v in region)
    W, H = u1 - u0, v1 - v0
    objects = list(inside) + list(outside)
    S = int(supersample)
    off = (np.arange(S) + 0.5) / S - 0.5
    V, U = np.meshgrid(np.arange(v0, v1, dtype=float), np.arange(u0, u1, dtype=float), indexing="ij")
    centres = np.c_[U.ravel(), V.ravel()]
    rgb = np.zeros((H * W, 3))
    grid = np.stack(np.meshgrid(off, off, indexing="xy"), -1).reshape(1, -1, 2)      # (1, S*S, 2) offsets (du, dv)
    step = max(chunk // (S * S), 1)                                                   # pixels per chunk
    for k in range(0, len(centres), step):
        sub = (centres[k:k + step, None, :] + grid).reshape(-1, 2)
        col = shade(objects, cast(camera, window, sub, inside, outside), **shading)
        rgb[k:k + step] = col.reshape(-1, S * S, 3).mean(1)
    parts = [cast(camera, window, centres[k:k + chunk], inside, outside) for k in range(0, len(centres), chunk)]
    centre = Hits(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("obj", "point", "normal", "path",
                                                                              "inner")))
    return rgb.reshape(H, W, 3), centre
