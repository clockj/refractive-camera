// Python bindings of the compiled core: batch projection through coaxial cylinders and concentric spheres.
//
// RefractCurved.cpp and RefractCurved.h are the search of OpenLPT's curved-window cameras, unchanged; this file only
// loops it over points (optionally on several threads, with the GIL released) and returns one row per image, in the
// layout of the pure-Python backend (refractive_camera/_reference/__init__.py).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "RefractCurved.h"

namespace py = pybind11;
namespace rc = refract_curved;

namespace {

struct PointResult {
    std::vector<rc::Ray> rays;
    bool capped = false;
};

using Points = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Run one(i) for i < m on `threads` threads (0: every hardware thread), keeping the results in point order.
template <class F>
std::vector<PointResult> forEach(std::size_t m, int threads, F&& one) {
    std::vector<PointResult> res(m);
    std::size_t nt = threads > 0 ? static_cast<std::size_t>(threads)
                                 : std::max(1u, std::thread::hardware_concurrency());
    nt = std::max<std::size_t>(1, std::min(nt, m));
    if (nt == 1) {
        for (std::size_t i = 0; i < m; ++i) {
            res[i] = one(i);
        }
        return res;
    }
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < nt; ++t) {
        pool.emplace_back([&, t] {
            for (std::size_t i = t; i < m; i += nt) {
                res[i] = one(i);
            }
        });
    }
    for (auto& th : pool) {
        th.join();
    }
    return res;
}

py::dict rows(const std::vector<PointResult>& res) {
    std::size_t k = 0;
    for (const auto& r : res) {
        k += r.rays.size();
    }
    py::array_t<std::int64_t> point(k);
    py::array_t<std::int8_t> branch(k);
    py::array_t<double> lam(k), c_z(k);
    py::array_t<double> dir({k, static_cast<std::size_t>(3)});
    py::array_t<bool> ring(k), capped(res.size());
    auto P = point.mutable_unchecked<1>();
    auto B = branch.mutable_unchecked<1>();
    auto L = lam.mutable_unchecked<1>();
    auto C = c_z.mutable_unchecked<1>();
    auto D = dir.mutable_unchecked<2>();
    auto G = ring.mutable_unchecked<1>();
    auto K = capped.mutable_unchecked<1>();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::size_t j = 0;
    for (std::size_t i = 0; i < res.size(); ++i) {
        K(i) = res[i].capped;
        for (const rc::Ray& r : res[i].rays) {
            P(j) = static_cast<std::int64_t>(i);
            B(j) = r.branch == rc::Branch::Far ? 1 : 0;
            L(j) = r.lambda;
            C(j) = r.c_z;
            for (int c = 0; c < 3; ++c) {
                D(j, c) = r.is_ring ? nan : r.dir[c];
            }
            G(j) = r.is_ring;
            ++j;
        }
    }
    py::dict out;
    out["point"] = point;
    out["branch"] = branch;
    out["lam"] = lam;
    out["c_z"] = c_z;
    out["dir"] = dir;
    out["ring"] = ring;
    out["capped"] = capped;
    return out;
}

rc::Stack makeStack(const std::vector<double>& n, const std::vector<double>& R) {
    rc::Stack s;
    s.n = n;
    s.R = R;
    if (!s.isValid()) {
        throw std::invalid_argument("invalid window: one more index than radii, positive indices, radii "
                                    "strictly decreasing and positive");
    }
    return s;
}

const double* pointData(const Points& points) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (M, 3)");
    }
    return points.data();
}

py::dict projectCylinder(const std::vector<double>& n, const std::vector<double>& R,
                         const std::array<double, 3>& pinhole, const Points& points, double tol, int threads) {
    const rc::Stack s = makeStack(n, R);
    const double* X = pointData(points);
    const std::size_t m = static_cast<std::size_t>(points.shape(0));
    std::vector<PointResult> res;
    {
        py::gil_scoped_release release;
        res = forEach(m, threads, [&](std::size_t i) {
            PointResult pr;
            rc::CylinderCoverage cov;
            pr.rays = rc::projectCylinder(s, pinhole.data(), X + 3 * i, &cov, tol);
            pr.capped = cov.near_branch && !cov.far_branch;
            return pr;
        });
    }
    return rows(res);
}

py::dict projectSphere(const std::vector<double>& n, const std::vector<double>& R,
                       const std::array<double, 3>& pinhole, const Points& points, double tol, int threads) {
    const rc::Stack s = makeStack(n, R);
    const double* X = pointData(points);
    const std::size_t m = static_cast<std::size_t>(points.shape(0));
    const double centre[3] = {0.0, 0.0, 0.0};
    std::vector<PointResult> res;
    {
        py::gil_scoped_release release;
        res = forEach(m, threads, [&](std::size_t i) {
            PointResult pr;
            pr.rays = rc::projectSphere(s, centre, pinhole.data(), X + 3 * i, tol);
            return pr;
        });
    }
    return rows(res);
}

// Pairs (pinhole i, point i): the search run once per pair, on `threads` threads.  A camera inside the window is the
// point of the outside problem whose pinhole is each point it looks at (refractive_camera.projection).
std::size_t pairCount(const Points& pinholes, const Points& points) {
    pointData(pinholes);
    pointData(points);
    if (pinholes.shape(0) != points.shape(0)) {
        throw std::invalid_argument("one pinhole per point: pinholes and points must have the same shape");
    }
    return static_cast<std::size_t>(points.shape(0));
}

py::dict projectCylinderPairs(const std::vector<double>& n, const std::vector<double>& R, const Points& pinholes,
                              const Points& points, double tol, int threads) {
    const rc::Stack s = makeStack(n, R);
    const std::size_t m = pairCount(pinholes, points);
    const double* C = pinholes.data();
    const double* X = points.data();
    std::vector<PointResult> res;
    {
        py::gil_scoped_release release;
        res = forEach(m, threads, [&](std::size_t i) {
            PointResult pr;
            rc::CylinderCoverage cov;
            pr.rays = rc::projectCylinder(s, C + 3 * i, X + 3 * i, &cov, tol);
            pr.capped = cov.near_branch && !cov.far_branch;
            return pr;
        });
    }
    return rows(res);
}

py::dict projectSpherePairs(const std::vector<double>& n, const std::vector<double>& R, const Points& pinholes,
                            const Points& points, double tol, int threads) {
    const rc::Stack s = makeStack(n, R);
    const std::size_t m = pairCount(pinholes, points);
    const double* C = pinholes.data();
    const double* X = points.data();
    const double centre[3] = {0.0, 0.0, 0.0};
    std::vector<PointResult> res;
    {
        py::gil_scoped_release release;
        res = forEach(m, threads, [&](std::size_t i) {
            PointResult pr;
            pr.rays = rc::projectSphere(s, centre, C + 3 * i, X + 3 * i, tol);
            return pr;
        });
    }
    return rows(res);
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "The compiled search: every image of each point through coaxial cylinders or concentric spheres, in "
              "the window's own frame (axis along z through the origin; centre at the origin).";
    m.def("project_cylinder", &projectCylinder, py::arg("n"), py::arg("R"), py::arg("pinhole"), py::arg("points"),
          py::arg("tol") = 1e-9, py::arg("threads") = 1,
          "Rows (point, branch, lam, c_z, dir, ring) for every image of each point, and `capped` per point.");
    m.def("project_sphere", &projectSphere, py::arg("n"), py::arg("R"), py::arg("pinhole"), py::arg("points"),
          py::arg("tol") = 1e-9, py::arg("threads") = 1,
          "Rows (point, branch, lam, c_z, dir, ring) for every image of each point, and `capped` per point.");
    m.def("project_cylinder_pairs", &projectCylinderPairs, py::arg("n"), py::arg("R"), py::arg("pinholes"),
          py::arg("points"), py::arg("tol") = 1e-9, py::arg("threads") = 1,
          "project_cylinder for pairs (pinhole i, point i); `point` in the rows indexes the pairs.");
    m.def("project_sphere_pairs", &projectSpherePairs, py::arg("n"), py::arg("R"), py::arg("pinholes"),
          py::arg("points"), py::arg("tol") = 1e-9, py::arg("threads") = 1,
          "project_sphere for pairs (pinhole i, point i); `point` in the rows indexes the pairs.");
}
