"""Builds the optional compiled core, refractive_camera._core.

Without a C++17 compiler (or pybind11) the build of the extension is skipped and the package installs as pure
Python, with the slower reference backend only.  Metadata lives in pyproject.toml.
"""
from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    ext_modules, cmdclass = [], {}
else:
    ext_modules = [
        Pybind11Extension(
            "refractive_camera._core",
            ["src/cpp/bindings.cpp", "src/cpp/RefractCurved.cpp"],
            include_dirs=["src/cpp"],
            cxx_std=17,
            optional=True,
        )
    ]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
