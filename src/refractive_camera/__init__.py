"""Camera models that see through cylindrical and spherical windows.

For a 3-D point behind a refractive window (a water-filled tube, a spherical tank), `project()` finds every pixel
where the point appears: none when the window hides it, one, or several.  The method is that of S. Zhong, S. Tan,
X. Zhang, Y. Li and R. Ni, "A complete forward projection algorithm through cylindrical and spherical refractive
windows", arXiv:2610.17413 (2026).

    >>> import refractive_camera as rc
    >>> tube = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))
    >>> cam = rc.Camera.look_at((0.35, 0.0, 0.0), (0.0, 0.0, 0.0), f=2000.0, size=(1280, 1024))
    >>> images = rc.project([[0.02, 0.03, 0.01]], cam, tube)
    >>> images.uv, images.branch
"""
from .camera import Camera
from .flat import FlatWindow
from .projection import Images, available_backends, project
from .render import render_dots
from .tracing import Trace, back_project, trace
from .triangulation import triangulate
from .windows import CylindricalWindow, SphericalWindow

__version__ = "0.1.0.dev0"

__all__ = ["Camera", "CylindricalWindow", "SphericalWindow", "FlatWindow", "Images", "Trace", "available_backends",
           "back_project", "project", "render_dots", "trace", "triangulate", "__version__"]
