"""Refractive windows: coaxial cylindrical and concentric spherical interfaces.

The media are numbered from the camera inward, as in the paper: medium 1 holds the camera and medium N the points.
`radii` lists the N - 1 interfaces from the outermost in, strictly decreasing; `indices` lists the N refractive
indices from the camera's medium in.  A water-filled acrylic tube seen from air is

    CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))

The model needs the camera outside the outermost interface and the point strictly inside the innermost one.
"""
import numpy as np


class _Window:
    kind = ""

    def __init__(self, radii, indices):
        radii = np.atleast_1d(np.asarray(radii, float))
        indices = np.atleast_1d(np.asarray(indices, float))
        if radii.ndim != 1 or radii.size < 1:
            raise ValueError("a window needs at least one interface")
        if indices.ndim != 1 or indices.size != radii.size + 1:
            raise ValueError("one index per medium: len(indices) must be len(radii) + 1")
        if not (np.all(np.isfinite(radii)) and np.all(np.isfinite(indices))):
            raise ValueError("radii and indices must be finite")
        if np.any(indices <= 0):
            raise ValueError("indices must be positive")
        if np.any(np.diff(radii) >= 0) or radii[-1] <= 0:
            raise ValueError("radii must be positive and strictly decreasing, the outermost first")
        self.radii = tuple(float(r) for r in radii)
        self.indices = tuple(float(n) for n in indices)

    def outside(self, points):
        """Whether points (..., 3) lie strictly outside the outermost interface, where the camera must be."""
        return self.radial_distance(points) > self.radii[0]

    def inside(self, points):
        """Whether points (..., 3) lie strictly inside the innermost interface, where the model projects them."""
        return self.radial_distance(points) < self.radii[-1]


class CylindricalWindow(_Window):
    """Coaxial cylindrical interfaces: a tube, a pipe, a rod.

    Parameters
    ----------
    radii : sequence of float
        Interface radii, outermost first, strictly decreasing.
    indices : sequence of float
        Refractive indices from the camera's medium inward; one more than `radii`.
    axis_point, axis_direction : array_like
        A point on the common axis and its direction (any length).
    """

    kind = "cylinder"

    def __init__(self, radii, indices, axis_point=(0.0, 0.0, 0.0), axis_direction=(0.0, 0.0, 1.0)):
        super().__init__(radii, indices)
        a = np.asarray(axis_direction, float)
        if a.shape != (3,) or not np.linalg.norm(a) > 0:
            raise ValueError("axis_direction must be a non-zero 3-vector")
        a = a / np.linalg.norm(a)
        # The window frame: e3 along the axis.  For the axis z the frame is the world's own, exactly.
        h = np.eye(3)[int(np.argmin(np.abs(a)))]
        e1 = h - (h @ a) * a
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        self.axis_point = np.array(axis_point, float).reshape(3)
        self.axis_direction = a
        self._Q = np.column_stack([e1, e2, a])            # window -> world: X = Q @ x + axis_point

    def __repr__(self):
        return (f"CylindricalWindow(radii={self.radii}, indices={self.indices}, "
                f"axis_point={self.axis_point.tolist()}, axis_direction={self.axis_direction.tolist()})")

    def to_local(self, points):
        """World points (..., 3) in the window frame, whose z axis is the cylinders' axis."""
        return (np.asarray(points, float) - self.axis_point) @ self._Q

    def vectors_to_world(self, v):
        """Window-frame vectors (..., 3) in the world frame."""
        return np.asarray(v, float) @ self._Q.T

    def vectors_to_local(self, v):
        """World vectors (..., 3) in the window frame."""
        return np.asarray(v, float) @ self._Q

    def radial_distance(self, points):
        """Distance of world points (..., 3) from the axis."""
        x = self.to_local(points)
        return np.hypot(x[..., 0], x[..., 1])


class SphericalWindow(_Window):
    """Concentric spherical interfaces: a spherical tank or flask, seen from outside.

    Parameters
    ----------
    radii : sequence of float
        Interface radii, outermost first, strictly decreasing.
    indices : sequence of float
        Refractive indices from the camera's medium inward; one more than `radii`.
    center : array_like
        The common centre.
    """

    kind = "sphere"

    def __init__(self, radii, indices, center=(0.0, 0.0, 0.0)):
        super().__init__(radii, indices)
        self.center = np.array(center, float).reshape(3)

    def __repr__(self):
        return f"SphericalWindow(radii={self.radii}, indices={self.indices}, center={self.center.tolist()})"

    def to_local(self, points):
        """World points (..., 3) relative to the centre."""
        return np.asarray(points, float) - self.center

    def vectors_to_world(self, v):
        """Window-frame vectors (..., 3) in the world frame (the frames differ by a translation only)."""
        return np.asarray(v, float)

    def vectors_to_local(self, v):
        """World vectors (..., 3) in the window frame (the same vectors)."""
        return np.asarray(v, float)

    def radial_distance(self, points):
        """Distance of world points (..., 3) from the centre."""
        return np.linalg.norm(self.to_local(points), axis=-1)
