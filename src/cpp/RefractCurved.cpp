#include "RefractCurved.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// Implementation of doc/refract_curved.tex.  Every routine here mirrors the
// reference implementation in test/refs/project_cylinder.py term for term, so
// that the two can be compared directly; where the reference uses a finite
// difference (dZ/dC_z) this does too, deliberately, rather than substituting a
// closed form the reference has not been checked against.

namespace refract_curved {
namespace {

constexpr double kPi = 3.14159265358979323846;

/** @brief At Lambda = Lmax the binding medium's b = Lambda/nu equals its radius,
  *        but rounding can put it a few ulp above; such a b is the grazing ray,
  *        not a ray that misses, so it is clamped to the radius. */
constexpr double kGraze = 8.0 * std::numeric_limits<double>::epsilon();

/** @brief Clamp to [-1, 1] before asin/acos, where rounding can push just past. */
inline double clamp1(double x) { return std::clamp(x, -1.0, 1.0); }

/** @brief q(a, L) = 1/sqrt(a^2 - L^2), the recurring factor of eq. (dPhi). */
inline double qFactor(double a, double lam) {
    const double d = a * a - lam * lam;
    return d > 0.0 ? 1.0 / std::sqrt(d) : std::numeric_limits<double>::infinity();
}

/** @brief Outer radius of medium k: R_{k-1}, with R_{-1} = r_P. */
inline double rOut(const Stack& s, double r_p, std::size_t k) {
    return k == 0 ? r_p : s.R[k - 1];
}

/** @brief Inner radius of medium k: R_k, with R_{N-1} = r_X. */
inline double rIn(const Stack& s, double r_x, std::size_t k) {
    return k + 1 == s.nMedia() ? r_x : s.R[k];
}

/** @brief The grid of far_turns(): 801 points on [0,1], dense at both ends. */
const std::vector<double>& arcGrid() {
    static const std::vector<double> grid = [] {
        constexpr int kN = 801;
        std::vector<double> g(kN);
        for (int i = 0; i < kN; ++i) {
            g[i] = 0.5 - 0.5 * std::cos(kPi * i / (kN - 1));
        }
        return g;
    }();
    return grid;
}

} // namespace

bool Stack::isValid() const {
    if (n.size() < 1 || R.size() + 1 != n.size()) {
        return false;
    }
    for (double v : n) {
        if (!(v > 0.0)) {
            return false;
        }
    }
    for (std::size_t k = 0; k + 1 < R.size(); ++k) {
        if (!(R[k] > R[k + 1])) {
            return false;
        }
    }
    return R.empty() || R.back() > 0.0;
}

std::vector<double> nus(const std::vector<double>& n, double c_z) {
    std::vector<double> out(n.size());
    for (std::size_t k = 0; k < n.size(); ++k) {
        const double d = n[k] * n[k] - c_z * c_z;
        out[k] = d > 0.0 ? std::sqrt(d) : 0.0;
    }
    return out;
}

LmaxKstar lmaxKstar(const Stack& s, double r_x, double c_z) {
    const std::vector<double> nu = nus(s.n, c_z);
    LmaxKstar out{std::numeric_limits<double>::infinity(), 0};
    for (std::size_t k = 0; k < s.nMedia(); ++k) {
        const double cap = nu[k] * rIn(s, r_x, k);
        if (cap < out.lmax) {
            out.lmax = cap;
            out.kstar = static_cast<int>(k) + 1;
        }
    }
    if (!std::isfinite(out.lmax)) {
        return LmaxKstar{0.0, 0};
    }
    return out;
}

std::optional<PhiZ> phiZ(double lam, double c_z, const Stack& s, double r_p,
                         double r_x, Branch b) {
    const std::vector<double> nu = nus(s.n, c_z);
    const std::size_t N = s.nMedia();
    PhiZ acc;
    for (std::size_t k = 0; k < N; ++k) {
        if (!(nu[k] > 0.0)) {
            return std::nullopt;
        }
        const double ro = rOut(s, r_p, k);
        const double ri = rIn(s, r_x, k);
        double bb = lam / nu[k];
        if (bb > ri) {
            if (bb > ri * (1.0 + kGraze)) {
                return std::nullopt; // the ray turns before reaching this interface
            }
            bb = ri; // Lambda = Lmax to rounding: the grazing ray of this medium
        }
        const double a1 = std::asin(clamp1(bb / ri));
        const double a2 = std::asin(clamp1(bb / ro));
        const double s1 = std::sqrt(std::max(ro * ro - bb * bb, 0.0));
        const double s2 = std::sqrt(std::max(ri * ri - bb * bb, 0.0));
        if (b == Branch::Far && k + 1 == N) {
            acc.phi += kPi - a1 - a2;
            acc.z += c_z * (s1 + s2) / nu[k];
        } else {
            acc.phi += a1 - a2;
            acc.z += c_z * (s1 - s2) / nu[k];
        }
    }
    return acc;
}

std::optional<double> radialAction(double lam, double c_z, const Stack& s,
                                   double r_p, double r_x, Branch b) {
    const std::vector<double> nu = nus(s.n, c_z);
    const std::size_t N = s.nMedia();
    // The elementary antiderivative of eq. (W). At bb = 0 the second term is 0,
    // its limit, also at r = 0, where bb / r would be 0/0: the far branch's
    // turning point at Lambda = 0 (the radial ray).
    auto F = [](double r, double bb, double v) {
        const double turn = bb == 0.0 ? 0.0 : bb * std::acos(clamp1(bb / r));
        return v * (std::sqrt(std::max(r * r - bb * bb, 0.0)) - turn);
    };
    double tot = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        if (!(nu[k] > 0.0)) {
            return std::nullopt;
        }
        const double ro = rOut(s, r_p, k);
        const double ri = rIn(s, r_x, k);
        double bb = lam / nu[k];
        if (bb > ri) {
            if (bb > ri * (1.0 + kGraze)) {
                return std::nullopt;
            }
            bb = ri; // as in phiZ
        }
        if (b == Branch::Far && k + 1 == N) {
            // Both legs of the turning point, each measured from r = b.
            tot += (F(ro, bb, nu[k]) - F(bb, bb, nu[k])) +
                   (F(ri, bb, nu[k]) - F(bb, bb, nu[k]));
        } else {
            tot += F(ro, bb, nu[k]) - F(ri, bb, nu[k]);
        }
    }
    return tot;
}

double dPhiDLam(double lam, double c_z, const Stack& s, double r_p, double r_x,
                Branch b) {
    const std::vector<double> nu = nus(s.n, c_z);
    const std::size_t N = s.nMedia();
    double sum = 0.0;
    for (std::size_t k = 0; k + 1 < N; ++k) {
        sum += qFactor(nu[k] * s.R[k], lam) - qFactor(nu[k + 1] * s.R[k], lam);
    }
    const double sign = (b == Branch::Far) ? -1.0 : 1.0;
    return sum + sign * qFactor(nu[N - 1] * r_x, lam) - qFactor(nu[0] * r_p, lam);
}

double dPhiDCz(double lam, double c_z, const Stack& s, double r_p, double r_x,
               Branch b) {
    const std::vector<double> nu = nus(s.n, c_z);
    const std::size_t N = s.nMedia();
    double sum = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        const double ro = rOut(s, r_p, k);
        const double ri = rIn(s, r_x, k);
        sum += (qFactor(nu[k] * ri, lam) - qFactor(nu[k] * ro, lam)) / (nu[k] * nu[k]);
    }
    double v = c_z * lam * sum;
    if (b == Branch::Far) {
        v -= 2.0 * c_z * lam * qFactor(nu[N - 1] * r_x, lam) / (nu[N - 1] * nu[N - 1]);
    }
    return v;
}

std::optional<double> dZDCz(double lam, double c_z, const Stack& s, double r_p,
                            double r_x, Branch b, double h) {
    const auto up = phiZ(lam, c_z + h, s, r_p, r_x, b);
    const auto dn = phiZ(lam, c_z - h, s, r_p, r_x, b);
    if (up && dn) {
        return (up->z - dn->z) / (2.0 * h);
    }
    // So close to Lmax one side leaves the domain (the tilt that lowers nu_k
    // lowers Lmax): a one-sided difference, None only if both sides do.
    const auto mid = phiZ(lam, c_z, s, r_p, r_x, b);
    if (!mid || (!up && !dn)) {
        return std::nullopt;
    }
    return up ? (up->z - mid->z) / h : (mid->z - dn->z) / h;
}

double z0(double c_z, const Stack& s, double r_p, double r_x) {
    const std::vector<double> nu = nus(s.n, c_z);
    double sum = 0.0;
    for (std::size_t k = 0; k < s.nMedia(); ++k) {
        if (!(nu[k] > 0.0)) {
            return std::numeric_limits<double>::infinity();
        }
        sum += (rOut(s, r_p, k) - rIn(s, r_x, k)) / nu[k];
    }
    return std::abs(c_z) * sum;
}

bool inDomain(double lam, double c_z, const Stack& s, double r_x) {
    const double n_min = *std::min_element(s.n.begin(), s.n.end());
    if (std::abs(c_z) >= n_min - 1e-12) {
        return false;
    }
    return lam >= 0.0 && lam < lmaxKstar(s, r_x, c_z).lmax;
}

// ------------------------------------------------------------- the near branch

std::optional<double> lamAt(double c_z, const Stack& s, double r_p, double r_x,
                            double target, int iters) {
    const double n_min = *std::min_element(s.n.begin(), s.n.end());
    if (std::abs(c_z) >= n_min) {
        return std::nullopt;
    }
    const double l_max = lmaxKstar(s, r_x, c_z).lmax;
    // The reach is taken at Lmax itself, and the bisection runs in s with
    // Lambda = Lmax (1 - s^2). Phi_near has a square-root edge at Lmax, so a
    // margin m in Lambda would cost about sqrt(m) in Phi and hide every tilt
    // whose reach exceeds the target by less; in s the edge is regular.
    const auto top = phiZ(l_max, c_z, s, r_p, r_x, Branch::Near);
    if (!top || top->phi < target) {
        return std::nullopt; // out of reach at this tilt
    }
    double lo = 0.0, hi = 1.0; // s: Phi_near falls as s grows
    for (int i = 0; i < iters; ++i) {
        const double mid = 0.5 * (lo + hi);
        const auto v =
            phiZ(l_max * (1.0 - mid * mid), c_z, s, r_p, r_x, Branch::Near);
        if (v && v->phi < target) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    const double sm = 0.5 * (lo + hi);
    // Strictly below Lmax, inside the domain of solveCylinder.
    return std::min(l_max * (1.0 - sm * sm), std::nextafter(l_max, 0.0));
}

double polish(double lam, double c_z, const Stack& s, double r_p, double r_x,
              double target, Branch b, int itmax, double tol) {
    const double l_max = lmaxKstar(s, r_x, c_z).lmax;
    double L = std::clamp(lam, 0.0, l_max * (1 - 1e-15));
    for (int i = 0; i < itmax; ++i) {
        const auto v = phiZ(L, c_z, s, r_p, r_x, b);
        if (!v) {
            return L;
        }
        const double g = v->phi - target;
        if (std::abs(g) < tol) {
            return L;
        }
        const double d = dPhiDLam(L, c_z, s, r_p, r_x, b);
        if (!std::isfinite(d) || d == 0.0) {
            return L;
        }
        double step = -g / d;
        double Ln = L + step;
        if (!(Ln >= 0.0 && Ln < l_max)) {
            // Halve towards the domain rather than leaving it (Remark newton).
            bool ok = false;
            for (int j = 0; j < 60; ++j) {
                step *= 0.5;
                Ln = L + step;
                if (Ln >= 0.0 && Ln < l_max) {
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                return L;
            }
        }
        L = Ln;
    }
    return L;
}

// -------------------------------------------------------------- the far branch

namespace {

/** @brief The (c_j, A_j) list of eq. (Flist): Phi_far' = sum_j c_j (A_j - t)^{-1/2}. */
struct FarTerms {
    std::vector<double> c;
    std::vector<double> A;
};

FarTerms farTerms(double c_z, const Stack& s, double r_p, double r_x) {
    const std::vector<double> nu = nus(s.n, c_z);
    const std::size_t N = s.nMedia();
    FarTerms t;
    t.c.reserve(2 * N);
    t.A.reserve(2 * N);
    for (std::size_t k = 0; k < N; ++k) { // inner radii
        t.c.push_back(k + 1 < N ? 1.0 : -1.0);
        const double a = nu[k] * rIn(s, r_x, k);
        t.A.push_back(a * a);
    }
    for (std::size_t k = 0; k < N; ++k) { // outer radii
        t.c.push_back(-1.0);
        const double a = nu[k] * rOut(s, r_p, k);
        t.A.push_back(a * a);
    }
    return t;
}

inline double farF(const FarTerms& t, double tt) {
    double sum = 0.0;
    for (std::size_t j = 0; j < t.c.size(); ++j) {
        sum += t.c[j] / std::sqrt(t.A[j] - tt);
    }
    return sum;
}

/**
  * @brief Eq. (intervalbound): bounds of Phi_far' over [t1, t2].
  *
  * Every term increases in t, so the low bound takes positive terms at t1 and
  * negative terms at t2, and the high bound the reverse.
  */
void farBounds(const FarTerms& t, double t1, double t2, double& lo, double& hi) {
    lo = hi = 0.0;
    for (std::size_t j = 0; j < t.c.size(); ++j) {
        const double v1 = t.c[j] / std::sqrt(t.A[j] - t1);
        const double v2 = t.c[j] / std::sqrt(t.A[j] - t2);
        if (t.c[j] > 0.0) {
            lo += v1;
            hi += v2;
        } else {
            lo += v2;
            hi += v1;
        }
    }
}

/** @brief Recursive subdivision of far_turns(), refining every sign change to tol. */
void farHalve(const FarTerms& t, double t1, double t2, double f1, double f2,
              double tol_abs, std::vector<double>& out) {
    double lo = 0.0, hi = 0.0;
    farBounds(t, t1, t2, lo, hi);
    if (lo > 0.0 || hi < 0.0) {
        return; // the bounds exclude zero: no turning point here
    }
    if (t2 - t1 <= tol_abs) {
        if ((f1 > 0.0) != (f2 > 0.0)) {
            out.push_back(0.5 * (t1 + t2));
        }
        return; // a closer pair is a tangency, whose arc sweeps nothing
    }
    const double tm = 0.5 * (t1 + t2);
    const double fm = farF(t, tm);
    farHalve(t, t1, tm, f1, fm, tol_abs, out);
    farHalve(t, tm, t2, fm, f2, tol_abs, out);
}

} // namespace

std::vector<double> farTurns(double c_z, const Stack& s, double r_p, double r_x,
                             double tol) {
    const FarTerms t = farTerms(c_z, s, r_p, r_x);
    const double A0 = *std::min_element(t.A.begin(), t.A.end());
    if (!(A0 > 0.0)) {
        return {};
    }

    // Terms sitting at the edge A_j = A0 dominate as t -> A0; past the cut-off
    // below they fix the sign, so no turning point can hide beyond it.
    double c0 = 0.0, M = 0.0;
    for (std::size_t j = 0; j < t.A.size(); ++j) {
        if (t.A[j] <= A0 * (1 + 1e-12)) {
            c0 += t.c[j];
        } else {
            M += std::abs(t.c[j]) / std::sqrt(t.A[j] - A0);
        }
    }
    double t_end = A0 * (1 - 1e-14);
    if (c0 != 0.0) {
        t_end = std::min(t_end, M > 0.0 ? A0 - (c0 / M) * (c0 / M) : -1.0);
    }
    if (t_end <= 0.0) {
        return {};
    }

    const std::vector<double>& arc = arcGrid();
    std::vector<double> grid(arc.size()), fg(arc.size());
    for (std::size_t i = 0; i < arc.size(); ++i) {
        grid[i] = t_end * arc[i] * arc[i];
        fg[i] = farF(t, grid[i]);
    }

    std::vector<double> roots_t;
    const double tol_abs = tol * A0;
    for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
        double lo = 0.0, hi = 0.0;
        farBounds(t, grid[i], grid[i + 1], lo, hi);
        if (lo > 0.0 || hi < 0.0) {
            continue;
        }
        farHalve(t, grid[i], grid[i + 1], fg[i], fg[i + 1], tol_abs, roots_t);
    }

    std::vector<double> out;
    out.reserve(roots_t.size());
    for (double v : roots_t) {
        out.push_back(std::sqrt(v));
    }
    std::sort(out.begin(), out.end());
    return out;
}

namespace {

/** @brief Bisection on a sign change of f in [lo, hi]; lo itself if f vanishes there. */
template <typename F>
double bisect(F&& f, double lo, double hi, int it = 70) {
    const double f0 = f(lo);
    if (f0 == 0.0) {
        return lo;
    }
    const bool up = f0 > 0.0;
    for (int i = 0; i < it; ++i) {
        const double m = 0.5 * (lo + hi);
        if ((f(m) > 0.0) == up) {
            lo = m;
        } else {
            hi = m;
        }
    }
    return 0.5 * (lo + hi);
}

} // namespace

std::vector<FarRoot> farRoots(double c_z, const Stack& s, double r_p, double r_x,
                              double target) {
    // The last arc ends at Lmax itself: a margin m there would drop the roots
    // of the targets within about sqrt(m) of Phi_far(Lmax), as in lamAt.
    const double l_max = lmaxKstar(s, r_x, c_z).lmax;
    if (!(l_max > 0.0)) {
        return {};
    }
    std::vector<double> cuts{0.0};
    for (double L : farTurns(c_z, s, r_p, r_x)) {
        if (L > 0.0 && L < l_max) {
            cuts.push_back(L);
        }
    }
    cuts.push_back(l_max);

    auto phi = [&](double L) {
        const auto v = phiZ(L, c_z, s, r_p, r_x, Branch::Far);
        return v ? v->phi : std::numeric_limits<double>::quiet_NaN();
    };

    std::vector<FarRoot> out;
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
        const double a = cuts[i], b = cuts[i + 1];
        const double fa = phi(a), fb = phi(b);
        if (!std::isfinite(fa) || !std::isfinite(fb)) {
            continue;
        }
        double L;
        if (fa == target) {
            L = a; // counted once, as the root of the arc it starts
        } else if (fb != target && ((fa > target) != (fb > target))) {
            L = bisect([&](double x) { return phi(x) - target; }, a, b);
        } else {
            continue;
        }
        const auto v = phiZ(L, c_z, s, r_p, r_x, Branch::Far);
        out.push_back(FarRoot{L, fb > fa ? 1 : -1, v ? v->z : 0.0});
    }
    return out;
}

// --------------------------------------------------------------------- spheres

namespace {

inline double dot3(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline double norm3(const double a[3]) { return std::sqrt(dot3(a, a)); }

} // namespace

std::optional<RayPlane> rayPlane(const double centre[3], const double pinhole[3],
                                 const double point[3]) {
    double u[3], v[3];
    for (int i = 0; i < 3; ++i) {
        u[i] = pinhole[i] - centre[i];
        v[i] = point[i] - centre[i];
    }
    RayPlane rp;
    rp.r_p = norm3(u);
    rp.r_x = norm3(v);
    if (!(rp.r_p > 0.0)) {
        return std::nullopt; // pinhole at the centre
    }
    for (int i = 0; i < 3; ++i) {
        rp.e1[i] = u[i] / rp.r_p;
    }
    if (!(rp.r_x > 0.0)) { // the point is at the centre: the radial ray
        rp.axial = true;
        rp.dphi = 0.0;
        return rp;
    }
    const double proj = dot3(v, rp.e1);
    double w[3];
    for (int i = 0; i < 3; ++i) {
        w[i] = v[i] - proj * rp.e1[i];
    }
    const double nw = norm3(w);
    if (nw <= 1e-12 * rp.r_x) { // collinear: Section ring
        rp.axial = true;
        rp.dphi = proj > 0.0 ? 0.0 : kPi;
        return rp;
    }
    for (int i = 0; i < 3; ++i) {
        rp.e2[i] = w[i] / nw;
    }
    rp.dphi = std::atan2(nw, proj);
    return rp;
}

double chordBound(const Stack& s, double r_p) {
    double sum = kPi;
    for (std::size_t k = 0; k < s.R.size(); ++k) {
        sum += std::acos(clamp1(s.R[k] / rOut(s, r_p, k)));
    }
    return sum;
}

Sweeps sweeps(const Stack& s, double r_p, double r_x) {
    const double l_max = lmaxKstar(s, r_x, 0.0).lmax;
    const double edge = l_max; // the reach at Lmax itself (phiZ clamps the grazing medium)
    Sweeps out;
    if (const auto near = phiZ(edge, 0.0, s, r_p, r_x, Branch::Near)) {
        out.near_reach = near->phi;
    }
    // Phi_far starts at pi and turns where farTurns says, so its supremum is the
    // largest of pi, the turning values and the value at the edge.
    double best = kPi;
    if (const auto far_edge = phiZ(edge, 0.0, s, r_p, r_x, Branch::Far)) {
        best = std::max(best, far_edge->phi);
    }
    for (double L : farTurns(0.0, s, r_p, r_x)) {
        if (L > 0.0 && L < edge) {
            if (const auto v = phiZ(L, 0.0, s, r_p, r_x, Branch::Far)) {
                best = std::max(best, v->phi);
            }
        }
    }
    out.m0 = std::max(best, out.near_reach);
    return out;
}

namespace {

/** @brief Step (3): (target, sense) for the short way and every wrap M(0) allows. */
std::vector<std::pair<double, double>> sphereTargets(double dphi, const Stack& s,
                                                     double r_p, double r_x) {
    const double eps = 8.0 * std::numeric_limits<double>::epsilon();
    const double cap = std::min(sweeps(s, r_p, r_x).m0, chordBound(s, r_p));
    std::vector<std::pair<double, double>> out;
    if (dphi > cap) {
        return out;
    }
    // At 0 and pi the two senses coincide, so each target is emitted once.
    const bool one_sense = dphi <= eps || std::abs(dphi - kPi) <= eps;
    out.emplace_back(dphi, 1.0);
    for (int m = 0;; ++m) {
        bool added = false;
        const double back = 2.0 * kPi * (m + 1) - dphi;
        const double fwd = dphi + 2.0 * kPi * (m + 1);
        if (back <= cap && !one_sense) {
            out.emplace_back(back, -1.0);
            added = true;
        }
        if (fwd <= cap) {
            out.emplace_back(fwd, 1.0);
            added = true;
        }
        if (!added) {
            break;
        }
    }
    return out;
}

/** @brief Step (4): the direction in the ray plane, at angle alpha to -e1. */
void sphereDirection(double lam, double n1, double r_p, const RayPlane& rp,
                     double out[3]) {
    const double sa = clamp1(lam / (n1 * r_p));
    const double ca = std::sqrt(std::max(1.0 - sa * sa, 0.0));
    for (int i = 0; i < 3; ++i) {
        out[i] = -ca * rp.e1[i] + (rp.axial ? 0.0 : sa * rp.e2[i]);
    }
}

/** @brief projectSphere() on a valid input centred at the origin, every length
  *        in units of 2^floor(log2 r_P). */
std::vector<Ray> projectSphereUnit(const Stack& s, const double pinhole[3],
                                   const double point[3], double tol, int itmax) {
    std::vector<Ray> out;
    const double centre[3] = {0.0, 0.0, 0.0};
    const auto rp_opt = rayPlane(centre, pinhole, point);
    if (!rp_opt) {
        return out;
    }
    const RayPlane& rp = *rp_opt;
    const double eps = 8.0 * std::numeric_limits<double>::epsilon();
    const int it_polish = itmax >= 0 ? itmax : 30;

    auto emit = [&](Branch b, double lam_signed, bool ring_flag) {
        Ray r;
        r.branch = b;
        r.lambda = lam_signed;
        r.c_z = 0.0;
        r.is_ring = ring_flag;
        sphereDirection(lam_signed, s.n[0], rp.r_p, rp, r.dir);
        out.push_back(r);
    };

    // The point must lie strictly inside the innermost interface. This comes
    // before the axial shortcuts, which would otherwise send the radial ray
    // towards a point beyond the pinhole or in the port wall.
    if (!s.R.empty() && !(s.R.back() > rp.r_x)) {
        return out;
    }
    if (rp.r_x <= 0.0) {
        emit(Branch::Near, 0.0, false); // the point at the centre: the radial ray
        return out;
    }
    if (rp.axial) {
        // Section ring: every plane through the axial line holds both points. The
        // radial ray (Delta phi = 0) or the ray through the centre (Delta phi = pi)
        // reaches the point in all of them at once, and so does a cone of rays for
        // each root Lambda > 0 of Phi = t, for every target t of step (3) on either
        // branch; each cone images as a ring. A single sphere has at most one ring;
        // nothing bounds a shell, so every root is returned (audit C5).
        emit(rp.dphi <= eps ? Branch::Near : Branch::Far, 0.0, false);
        const double near_reach = sweeps(s, rp.r_p, rp.r_x).near_reach;
        for (const auto& target_sense : sphereTargets(rp.dphi, s, rp.r_p, rp.r_x)) {
            const double target = target_sense.first;
            if (target <= near_reach) {
                const auto L = lamAt(0.0, s, rp.r_p, rp.r_x, target);
                if (L && *L > 1e-12) {
                    emit(Branch::Near, *L, true);
                }
            }
            for (const FarRoot& fr : farRoots(0.0, s, rp.r_p, rp.r_x, target)) {
                if (fr.lambda > 1e-12) {
                    emit(Branch::Far, fr.lambda, true);
                }
            }
        }
        return out;
    }

    const double near_reach = sweeps(s, rp.r_p, rp.r_x).near_reach;
    for (const auto& [target, sense] : sphereTargets(rp.dphi, s, rp.r_p, rp.r_x)) {
        if (target <= near_reach) {
            if (const auto L0 = lamAt(0.0, s, rp.r_p, rp.r_x, target)) {
                const double L = polish(*L0, 0.0, s, rp.r_p, rp.r_x, target,
                                        Branch::Near, it_polish);
                const auto v = phiZ(L, 0.0, s, rp.r_p, rp.r_x, Branch::Near);
                if (v && std::abs(v->phi - target) < tol) {
                    emit(Branch::Near, sense * L, false);
                }
            }
        }
        for (const FarRoot& fr : farRoots(0.0, s, rp.r_p, rp.r_x, target)) {
            const double L = polish(fr.lambda, 0.0, s, rp.r_p, rp.r_x, target,
                                    Branch::Far, it_polish);
            const auto v = phiZ(L, 0.0, s, rp.r_p, rp.r_x, Branch::Far);
            if (v && std::abs(v->phi - target) < tol) {
                emit(Branch::Far, sense * L, false);
            }
        }
    }
    return out;
}

} // namespace

std::vector<Ray> projectSphere(const Stack& s, const double centre[3],
                               const double pinhole[3], const double point[3],
                               double tol, int itmax) {
    const double u[3] = {pinhole[0] - centre[0], pinhole[1] - centre[1],
                         pinhole[2] - centre[2]};
    const double r_p = norm3(u);
    // The model needs the pinhole outside the outermost interface.
    if (!s.isValid() || !(r_p > 0.0) || !std::isfinite(r_p) ||
        !(s.R.empty() || r_p > s.R.front())) {
        return {};
    }
    // Lengths in units of 2^floor(log2 r_P), as in projectCylinder(): exact, and
    // every tolerance is then dimensionless.
    const double w = std::ldexp(1.0, std::ilogb(r_p));
    Stack unit = s;
    for (double& R : unit.R) {
        R /= w;
    }
    const double p[3] = {u[0] / w, u[1] / w, u[2] / w};
    const double x[3] = {(point[0] - centre[0]) / w, (point[1] - centre[1]) / w,
                         (point[2] - centre[2]) / w};
    std::vector<Ray> out = projectSphereUnit(unit, p, x, tol, itmax);
    for (Ray& r : out) {
        r.lambda *= w;
    }
    return out;
}

// ------------------------------------------------------------------- cylinders

std::vector<double> switchTilts(const Stack& s, double r_x, double chi) {
    const std::size_t N = s.nMedia();
    std::vector<double> caps(N);
    for (std::size_t k = 0; k < N; ++k) {
        caps[k] = rIn(s, r_x, k);
    }
    std::vector<double> out;
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = j + 1; k < N; ++k) {
            const double den = caps[j] * caps[j] - caps[k] * caps[k];
            if (den == 0.0) {
                continue;
            }
            const double c2 = (s.n[j] * s.n[j] * caps[j] * caps[j] -
                               s.n[k] * s.n[k] * caps[k] * caps[k]) /
                              den;
            if (c2 > 0.0 && c2 < chi * chi) {
                out.push_back(std::sqrt(c2));
            }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<double> reachTerms(double c_z, const Stack& s, double r_p, double r_x) {
    const std::vector<double> nu = nus(s.n, c_z);
    const std::size_t N = s.nMedia();
    double l_max = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < N; ++k) {
        l_max = std::min(l_max, nu[k] * rIn(s, r_x, k));
    }
    std::vector<double> terms(N);
    for (std::size_t k = 0; k < N; ++k) {
        const double ri = rIn(s, r_x, k);
        const double ro = rOut(s, r_p, k);
        terms[k] = std::asin(std::min(l_max / (nu[k] * ri), 1.0)) -
                   std::asin(clamp1(l_max / (nu[k] * ro)));
    }
    return terms;
}

std::vector<Component> components(const Stack& s, double r_p, double r_x,
                                  double dphi, double chi, double tol) {
    const double n_min = *std::min_element(s.n.begin(), s.n.end());
    if (chi <= 0.0) {
        chi = n_min * (1 - 1e-9);
    }
    std::vector<double> cuts{0.0};
    for (double t : switchTilts(s, r_x, chi)) {
        cuts.push_back(t);
    }
    cuts.push_back(chi);

    auto sumMin = [](const std::vector<double>& a, const std::vector<double>& b) {
        double t = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            t += std::min(a[i], b[i]);
        }
        return t;
    };
    auto sumMax = [](const std::vector<double>& a, const std::vector<double>& b) {
        double t = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            t += std::max(a[i], b[i]);
        }
        return t;
    };

    struct Cell {
        double a, b;
        std::vector<double> ta, tb;
    };
    std::vector<Component> kept;
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
        std::vector<Cell> stack{
            Cell{cuts[i], cuts[i + 1], reachTerms(cuts[i], s, r_p, r_x),
                 reachTerms(cuts[i + 1], s, r_p, r_x)}};
        while (!stack.empty()) {
            Cell c = std::move(stack.back());
            stack.pop_back();
            if (sumMax(c.ta, c.tb) < dphi) {
                continue; // the upper bound is below the target: no tilt here reaches
            }
            if (sumMin(c.ta, c.tb) >= dphi || c.b - c.a < tol) {
                kept.push_back(Component{c.a, c.b});
                continue; // an undecided sliver is kept, so no component is lost
            }
            const double m = 0.5 * (c.a + c.b);
            std::vector<double> tm = reachTerms(m, s, r_p, r_x);
            stack.push_back(Cell{m, c.b, tm, c.tb});
            stack.push_back(Cell{c.a, m, c.ta, std::move(tm)});
        }
    }

    std::sort(kept.begin(), kept.end(),
              [](const Component& x, const Component& y) { return x.lo < y.lo; });
    std::vector<Component> out;
    for (const Component& c : kept) {
        if (!out.empty() && c.lo <= out.back().hi) {
            out.back().hi = std::max(out.back().hi, c.hi);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::optional<double> zetaAt(double c_z, const Stack& s, double r_p, double r_x,
                             double dphi) {
    const auto L = lamAt(c_z, s, r_p, r_x, dphi, 60);
    if (!L) {
        return std::nullopt;
    }
    const auto v = phiZ(*L, c_z, s, r_p, r_x, Branch::Near);
    if (!v) {
        return std::nullopt;
    }
    return v->z;
}

std::optional<Seed> classifyNear(const Stack& s, double r_p, double r_x,
                                 double dphi, double dz, int iters) {
    const double sgn = dz >= 0.0 ? 1.0 : -1.0; // zeta is odd in C_z
    const double target = std::abs(dz);

    // zeta at c. At an end of a component where the reach is dphi, Lambda =
    // Lmax: the ray is the grazing ray and zeta is its Z(Lmax, C_z) in closed
    // form, so the bisection brackets the whole component. (Approaching the
    // ends from inside lost the rays of the tilts next to an end, within about
    // 1e-10 of grazing.) The same value serves at any tilt within rounding of
    // such an end, where lamAt finds the reach a hair below dphi.
    auto zeta = [&](double c) -> std::optional<double> {
        if (const auto v = zetaAt(c, s, r_p, r_x, dphi)) {
            return v;
        }
        const auto pz =
            phiZ(lmaxKstar(s, r_x, c).lmax, c, s, r_p, r_x, Branch::Near);
        if (pz && pz->phi >= dphi - 1e-9) {
            return pz->z;
        }
        return std::nullopt;
    };
    // First finite zeta walking from c0 towards c1: the fallback for an end
    // where zeta() has no value. C_z = 0 is a regular point, used as it stands.
    auto zin = [&](double c0, double c1, double& c_out,
                   double& z_out) -> bool {
        static const double kSteps[] = {1e-12, 1e-10, 1e-8, 1e-6, 1e-4,
                                        1e-3,  1e-2,  3e-2, 1e-1};
        if (c0 == 0.0) {
            if (const auto v = zeta(0.0)) {
                c_out = 0.0;
                z_out = *v;
                return true;
            }
        }
        for (double t : kSteps) {
            const double c = c0 + t * (c1 - c0);
            if (const auto v = zeta(c)) {
                c_out = c;
                z_out = *v;
                return true;
            }
        }
        return false;
    };

    for (const Component& comp : components(s, r_p, r_x, dphi)) {
        if (comp.hi <= comp.lo) {
            continue;
        }
        double a = comp.lo, za = 0.0, b = comp.hi, zb = 0.0;
        if (const auto v = zeta(a)) {
            za = *v;
        } else if (!zin(comp.lo, comp.hi, a, za)) {
            continue;
        }
        if (const auto v = zeta(b)) {
            zb = *v;
        } else if (!zin(comp.hi, comp.lo, b, zb)) {
            continue;
        }
        if (b <= a) {
            continue;
        }
        if (!(std::min(za, zb) <= target && target <= std::max(za, zb))) {
            continue;
        }
        for (int i = 0; i < iters; ++i) {
            const double mm = 0.5 * (a + b);
            const auto v = zeta(mm);
            if (!v) {
                break;
            }
            if ((*v - target) * (za - target) > 0.0) {
                a = mm;
            } else {
                b = mm;
            }
        }
        const double c = 0.5 * (a + b);
        if (const auto L = lamAt(c, s, r_p, r_x, dphi)) {
            return Seed{*L, sgn * c};
        }
        // c is within rounding of the end: the grazing ray.
        const double l_max = lmaxKstar(s, r_x, c).lmax;
        const auto pz = phiZ(l_max, c, s, r_p, r_x, Branch::Near);
        if (!pz || pz->phi < dphi - 1e-9) {
            continue;
        }
        return Seed{std::nextafter(l_max, 0.0), sgn * c};
    }
    return std::nullopt;
}

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();

/**
  * @brief The 2N terms of eq. (Flist), as far_box_jacobian and _far_point use them.
  *
  * Term 2k is the inner radius of medium k and term 2k+1 the outer one, so
  * `med` runs 0,0,1,1,..., `rad` alternates Rin[k], Rout[k], and `sig` is
  * (+1 except -1 in the last medium, then -1) per pair.
  */
struct FarStack {
    std::vector<double> n, Rin, Rout; // N each
    std::vector<int> med;             // 2N
    std::vector<double> rad, sig;     // 2N
    std::size_t nMed() const { return n.size(); }
    std::size_t nTerms() const { return med.size(); }
};

FarStack farStack(const Stack& s, double r_p, double r_x) {
    const std::size_t N = s.nMedia();
    FarStack st;
    st.n = s.n;
    st.Rin.resize(N);
    st.Rout.resize(N);
    for (std::size_t k = 0; k < N; ++k) {
        st.Rin[k] = rIn(s, r_x, k);
        st.Rout[k] = rOut(s, r_p, k);
    }
    for (std::size_t k = 0; k < N; ++k) {
        st.med.push_back(static_cast<int>(k));
        st.rad.push_back(st.Rin[k]);
        st.sig.push_back(k + 1 < N ? 1.0 : -1.0);
        st.med.push_back(static_cast<int>(k));
        st.rad.push_back(st.Rout[k]);
        st.sig.push_back(-1.0);
    }
    return st;
}

/** @brief An interval; the whole line stands for "unknown", as NaN does in the reference. */
struct Iv {
    double lo = 0.0;
    double hi = 0.0;
};

/** @brief Interval product, taking a NaN corner to mean the product is unbounded. */
Iv ivMul(Iv a, Iv b) {
    const double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    for (double v : p) {
        if (std::isnan(v)) {
            return Iv{-kInf, kInf};
        }
    }
    return Iv{std::min(std::min(p[0], p[1]), std::min(p[2], p[3])),
              std::max(std::max(p[0], p[1]), std::max(p[2], p[3]))};
}

/** @brief One box of the far-branch search, in (theta*, C_z). */
struct Box {
    double ta = 0.0, tb = 0.0; ///< theta* range; Lambda = Lmax(C_z) sin(theta*).
    double ca = 0.0, cb = 0.0; ///< tilt range.
    double pa = 0.0, pb = 0.0; ///< the piece between cuts (eq. cuts) this box lies in.
    int ks = 0;                ///< k*, 0-based, fixed within a piece.
};

struct BoxBounds {
    double plo, phi, zlo, zhi;
};

/**
  * @brief Eq. (farbox): bounds on Phi_far and Z_far over a box.
  *
  * b_k = R_kstar (nu_kstar / nu_k) sin(theta*) increases with theta* and is
  * monotone in C_z between cuts; each medium's sweep and chord increase with
  * b_k, except that medium N's decrease on the far branch.
  */
BoxBounds boxBounds(const FarStack& st, const Box& b) {
    const std::size_t N = st.nMed();
    std::vector<double> nua(N), nub(N), blo(N), bhi(N);
    for (std::size_t k = 0; k < N; ++k) {
        nua[k] = std::sqrt(std::max(st.n[k] * st.n[k] - b.ca * b.ca, 0.0));
        nub[k] = std::sqrt(std::max(st.n[k] * st.n[k] - b.cb * b.cb, 0.0));
    }
    const double sa = std::sin(b.ta), sb = std::sin(b.tb);
    for (std::size_t k = 0; k < N; ++k) {
        const double ra = st.Rin[b.ks] * nua[b.ks] / nua[k];
        const double rb = st.Rin[b.ks] * nub[b.ks] / nub[k];
        blo[k] = std::min(sa * std::min(ra, rb), st.Rin[k]);
        bhi[k] = std::min(sb * std::max(ra, rb), st.Rin[k]);
    }
    auto sweep = [&](double bb, std::size_t k) {
        return std::asin(clamp1(bb / st.Rin[k])) - std::asin(clamp1(bb / st.Rout[k]));
    };
    auto chord = [&](double bb, std::size_t k) {
        return std::sqrt(std::max(st.Rout[k] * st.Rout[k] - bb * bb, 0.0)) -
               std::sqrt(std::max(st.Rin[k] * st.Rin[k] - bb * bb, 0.0));
    };
    auto fsweep = [&](double bb) {
        return kPi - std::asin(clamp1(bb / st.Rin[N - 1])) -
               std::asin(clamp1(bb / st.Rout[N - 1]));
    };
    auto fchord = [&](double bb) {
        return std::sqrt(std::max(st.Rout[N - 1] * st.Rout[N - 1] - bb * bb, 0.0)) +
               std::sqrt(std::max(st.Rin[N - 1] * st.Rin[N - 1] - bb * bb, 0.0));
    };

    double psum_lo = 0.0, psum_hi = 0.0, zlo = 0.0, zhi = 0.0, amp = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        double plo_k, phi_k, slo_k, shi_k;
        if (k + 1 == N) { // the far branch reverses the last medium's monotonicity
            plo_k = fsweep(bhi[k]);
            phi_k = fsweep(blo[k]);
            slo_k = fchord(bhi[k]);
            shi_k = fchord(blo[k]);
        } else {
            plo_k = sweep(blo[k], k);
            phi_k = sweep(bhi[k], k);
            slo_k = chord(blo[k], k);
            shi_k = chord(bhi[k], k);
        }
        psum_lo += plo_k;
        psum_hi += phi_k;
        zlo += b.ca / nua[k] * slo_k;
        zhi += b.cb / nub[k] * shi_k;
        amp = std::max(amp, st.n[k] * st.n[k] / (nub[k] * nub[k]));
    }
    const double pad = 64.0 * kEps * (1.0 + amp);
    return BoxBounds{psum_lo - pad * 10.0, psum_hi + pad * 10.0,
                     zlo * (1.0 - pad) - pad * 1e-3,
                     zhi * (1.0 + pad) + pad * 1e-3};
}

struct Jac2 {
    double lo[2][2];
    double hi[2][2];
};

/**
  * @brief Eq. (farjac): bounds on the Jacobian of (Phi_far, Z_far) in (theta*, C_z).
  *
  * Every factor but c_j and kappa_j' is nonnegative and monotone on a box, so
  * each is bounded by its end values. The grazing term, kappa_j = 1, is written
  * in theta* directly to keep accuracy as theta* -> pi/2.
  */
Jac2 farBoxJacobian(const FarStack& st, double ta, double tb, double ca, double cb,
                    int ks) {
    const std::size_t T = st.nTerms();
    const double nsa = std::sqrt(std::max(st.n[ks] * st.n[ks] - ca * ca, 0.0));
    const double nsb = std::sqrt(std::max(st.n[ks] * st.n[ks] - cb * cb, 0.0));
    const double Rs = st.Rin[ks];
    const double sa = std::sin(ta), sb = std::sin(tb);
    const double co_lo = std::cos(tb), co_hi = std::cos(ta);

    Iv acc[2][2];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            acc[r][c] = Iv{0.0, 0.0};
        }
    }
    bool bad[2][2] = {{false, false}, {false, false}};

    for (std::size_t j = 0; j < T; ++j) {
        const int m = st.med[j];
        const double nk = st.n[m];
        const double nua = std::sqrt(std::max(nk * nk - ca * ca, 0.0));
        const double nub = std::sqrt(std::max(nk * nk - cb * cb, 0.0));
        const double radj = st.rad[j];
        const double ma = Rs * nsa / (nua * radj);
        const double mb = Rs * nsb / (nub * radj);
        const double mlo = std::min(ma, mb), mhi = std::max(ma, mb);
        const double kc = Rs / radj * (st.n[ks] * st.n[ks] - nk * nk);
        const double fa = ca / (nsa * nua * nua * nua);
        const double fb = cb / (nsb * nub * nub * nub);
        const Iv d = kc >= 0.0 ? Iv{kc * fa, kc * fb} : Iv{kc * fb, kc * fa};

        const double xlo = std::min(mlo * sa, 1.0);
        const double xhi = std::min(mhi * sb, 1.0);
        const double Dlo = std::sqrt(std::max(1.0 - xhi * xhi, 0.0));
        const double Dhi = std::sqrt(std::max(1.0 - xlo * xlo, 0.0));
        const double iDlo = 1.0 / Dhi; // reciprocal swaps the ends
        const double iDhi = 1.0 / Dlo;
        const double glo = ca / nua, ghi = cb / nub;
        const double hlo = nk * nk / (nua * nua * nua);
        const double hhi = nk * nk / (nub * nub * nub);

        const bool graze = (m == ks) && (j % 2 == 0);
        Iv pt, pc, zt, zc;
        if (graze) {
            pt = Iv{1.0, 1.0};
            pc = Iv{0.0, 0.0};
            zt = Iv{radj * glo * sa, radj * ghi * sb};
            zc = Iv{radj * hlo * co_lo, radj * hhi * co_hi};
        } else {
            pt = Iv{mlo * co_lo * iDlo, mhi * co_hi * iDhi};
            pc = ivMul(d, Iv{sa * iDlo, sb * iDhi});
            zt = Iv{radj * glo * mlo * mlo * sa * co_lo * iDlo,
                    radj * ghi * mhi * mhi * sb * co_hi * iDhi};
            const Iv q = ivMul(d, Iv{glo * mlo * sa * sa * iDlo,
                                     ghi * mhi * sb * sb * iDhi});
            zc = Iv{radj * (hlo * Dlo - q.hi), radj * (hhi * Dhi - q.lo)};
        }

        const double sg = st.sig[j];
        const Iv cell[2][2] = {{pt, pc}, {zt, zc}};
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                // (1,1) carries -sig; the others carry +sig.
                const double s = (r == 1 && c == 1) ? -sg : sg;
                const Iv v = cell[r][c];
                const double add_lo = s > 0.0 ? v.lo : -v.hi;
                const double add_hi = s > 0.0 ? v.hi : -v.lo;
                if (std::isnan(add_lo) || std::isnan(add_hi)) {
                    bad[r][c] = true;
                } else {
                    acc[r][c].lo += add_lo;
                    acc[r][c].hi += add_hi;
                }
            }
        }
    }

    Jac2 out;
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            if (bad[r][c] || std::isnan(acc[r][c].lo) || std::isnan(acc[r][c].hi)) {
                out.lo[r][c] = -kInf;
                out.hi[r][c] = kInf;
                continue;
            }
            const double grow =
                64.0 * kEps * (std::abs(acc[r][c].lo) + std::abs(acc[r][c].hi));
            out.lo[r][c] = acc[r][c].lo - grow;
            out.hi[r][c] = acc[r][c].hi + grow;
        }
    }
    return out;
}

struct FarPoint {
    double phi = 0.0, z = 0.0;
    double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double ephi = 0.0, ez = 0.0; ///< Bounds on the rounding error of phi and z.
};

/** @brief Phi_far, Z_far, their Jacobian and the rounding error of each value. */
FarPoint farPoint(const FarStack& st, double th, double C, int ks) {
    const std::size_t N = st.nMed(), T = st.nTerms();
    std::vector<double> nu(N), amp(N);
    for (std::size_t k = 0; k < N; ++k) {
        nu[k] = std::sqrt(std::max(st.n[k] * st.n[k] - C * C, 0.0));
        amp[k] = (st.n[k] * st.n[k] + C * C) / (nu[k] * nu[k]);
    }
    const double lam_edge = nu[ks] * st.Rin[ks];
    const double sth = std::sin(th);

    FarPoint out;
    double sum_asn_abs = 0.0, sum_grow_ex = 0.0, sum_zabs = 0.0, sum_z_err = 0.0;
    out.phi = kPi;
    for (std::size_t j = 0; j < T; ++j) {
        const int m = st.med[j];
        const double x = std::min(lam_edge * sth / (nu[m] * st.rad[j]), 1.0);
        const bool graze = (m == ks) && (j % 2 == 0);
        // Writing the grazing term in theta* keeps full accuracy as theta* -> pi/2.
        const double asn = graze ? th : std::asin(clamp1(x));
        const double root = graze ? std::cos(th) : std::sqrt(std::max(1.0 - x * x, 0.0));
        const double zj = -st.sig[j] * st.rad[j] * (C / nu[m]) * root;
        out.phi += st.sig[j] * asn;
        out.z += zj;

        const double ex = 6.0 + amp[ks] + amp[m];
        double grow = 0.0;
        if (!graze) {
            grow = root > 0.0 ? x / root : 1e300;
            if (!std::isfinite(grow)) {
                grow = 1e300;
            }
        }
        sum_asn_abs += std::abs(asn);
        sum_grow_ex += grow * ex;
        sum_zabs += std::abs(zj);
        sum_z_err += std::abs(zj) * (grow * grow * ex + 6.0 + amp[m]);
    }
    const double terms = static_cast<double>(2 * T + 2);
    out.ephi = kEps * (sum_grow_ex + terms * (kPi + sum_asn_abs));
    out.ez = kEps * (sum_z_err + terms * sum_zabs);

    const Jac2 jb = farBoxJacobian(st, th, th, C, C, ks);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            out.J[r][c] = 0.5 * (jb.lo[r][c] + jb.hi[r][c]);
        }
    }
    return out;
}

/** @brief Newton on (Phi_far - dphi, Z_far - dz) in (theta*, C_z), clamped into [lo, hi]. */
struct NewtonOut {
    double th, c;
    double res_phi, res_z;
    double ephi, ez;
};

NewtonOut farNewton(const FarStack& st, double th, double c, int ks, double dphi,
                    double dz, const double lo[2], const double hi[2],
                    const double Y[2][2], bool have_Y) {
    double x0 = th, x1 = c;
    double f0 = 0.0, f1 = 0.0, e0 = 0.0, e1 = 0.0;
    for (int it = 0; it < 60; ++it) {
        const FarPoint fp = farPoint(st, x0, x1, ks);
        f0 = fp.phi - dphi;
        f1 = fp.z - dz;
        e0 = fp.ephi;
        e1 = fp.ez;
        if (std::abs(f0) <= e0 && std::abs(f1) <= e1) {
            break; // the residual is within the rounding error of the values
        }
        const double det = fp.J[0][0] * fp.J[1][1] - fp.J[0][1] * fp.J[1][0];
        double y0, y1;
        if (std::isfinite(det) && det != 0.0) {
            y0 = x0 - (fp.J[1][1] * f0 - fp.J[0][1] * f1) / det;
            y1 = x1 - (-fp.J[1][0] * f0 + fp.J[0][0] * f1) / det;
        } else {
            y0 = x0;
            y1 = x1;
        }
        if (!(y0 >= lo[0] && y0 <= hi[0] && y1 >= lo[1] && y1 <= hi[1])) {
            if (have_Y) {
                // The fixed Y of Krawczyk's test converges on the box.
                y0 = x0 - (Y[0][0] * f0 + Y[0][1] * f1);
                y1 = x1 - (Y[1][0] * f0 + Y[1][1] * f1);
            } else {
                y0 = std::clamp(y0, lo[0], hi[0]);
                y1 = std::clamp(y1, lo[1], hi[1]);
            }
        }
        if (std::abs(y0 - x0) <= 4.0 * kEps * std::max(1.0, std::abs(x0)) &&
            std::abs(y1 - x1) <= 4.0 * kEps * std::max(1.0, std::abs(x1))) {
            break;
        }
        x0 = y0;
        x1 = y1;
    }
    return NewtonOut{x0, x1, std::abs(f0), std::abs(f1), e0, e1};
}

// ------------------------------------------------ double-double arithmetic
// A value is an unevaluated sum hi + lo of two doubles, about 32 significant
// digits (Dekker 1971; Hida, Li and Bailey's QD library). Each operation below
// has a relative error of a few 2^-104 (4.9e-32).

struct DD {
    double hi = 0.0;
    double lo = 0.0;
};

inline DD twoSum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return DD{s, (a - (s - bb)) + (b - bb)};
}

inline DD fastTwoSum(double a, double b) { // requires |a| >= |b|
    const double s = a + b;
    return DD{s, b - (s - a)};
}

inline DD twoProd(double a, double b) {
    const double p = a * b;
    return DD{p, std::fma(a, b, -p)};
}

inline DD operator+(DD a, DD b) {
    DD s = twoSum(a.hi, b.hi);
    const DD t = twoSum(a.lo, b.lo);
    s.lo += t.hi;
    s = fastTwoSum(s.hi, s.lo);
    s.lo += t.lo;
    return fastTwoSum(s.hi, s.lo);
}

inline DD operator-(DD a) { return DD{-a.hi, -a.lo}; }

inline DD operator-(DD a, DD b) { return a + (-b); }

inline DD operator*(DD a, DD b) {
    DD p = twoProd(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return fastTwoSum(p.hi, p.lo);
}

inline DD operator*(DD a, double b) {
    DD p = twoProd(a.hi, b);
    p.lo += a.lo * b;
    return fastTwoSum(p.hi, p.lo);
}

inline DD operator/(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    DD r = a - b * q1;
    const double q2 = r.hi / b.hi;
    r = r - b * q2;
    const double q3 = r.hi / b.hi;
    return fastTwoSum(q1, q2) + DD{q3, 0.0};
}

/** @brief sqrt of a non-negative value; 0 for a value rounding made negative. */
DD ddSqrt(DD a) {
    if (!(a.hi > 0.0)) {
        return DD{};
    }
    const double x = 1.0 / std::sqrt(a.hi);
    const double ax = a.hi * x;
    return twoSum(ax, (a - twoProd(ax, ax)).hi * (0.5 * x));
}

/** @brief sin and cos of a double with |t| <= pi/2, by their Taylor series. */
void ddSinCos(double t, DD& s, DD& c) {
    const DD t2 = twoProd(t, t);
    DD st{t, 0.0}, ct{1.0, 0.0}; // t^(2k+1)/(2k+1)! and t^(2k)/(2k)!
    s = st;
    c = ct;
    for (int k = 1; k <= 24; ++k) {
        ct = ct * t2 / DD{static_cast<double>((2 * k - 1) * (2 * k)), 0.0};
        st = st * t2 / DD{static_cast<double>((2 * k) * (2 * k + 1)), 0.0};
        s = (k % 2) ? s - st : s + st;
        c = (k % 2) ? c - ct : c + ct;
        if (std::abs(ct.hi) < 1e-36) {
            break;
        }
    }
}

/** @brief asin of |x| <= 1/2: one Newton step on sin from the double-precision value. */
DD ddAsinSmall(DD x) {
    const double a0 = std::asin(x.hi);
    DD s, c;
    ddSinCos(a0, s, c);
    return DD{a0, 0.0} - (s - x) / c;
}

/** @brief asin of x in [0, 1]; above 1/2 through asin x = pi/2 - 2 asin sqrt((1 - x)/2),
  *        which stays accurate where asin is steep (a grazing ray). */
DD ddAsin(DD x) {
    if (x.hi <= 0.5) {
        return ddAsinSmall(x);
    }
    const DD half_pi{1.5707963267948966, 6.123233995736766e-17};
    return half_pi - ddAsinSmall(ddSqrt((DD{1.0, 0.0} - x) * 0.5)) * 2.0;
}

/**
  * @brief (Phi_far - dphi, Z_far - dz) at one point in double-double, and a bound
  *        on the error of each, as the reference's _far_residual_mp computes the
  *        residual in 34 digits.
  *
  * The inputs are doubles and exact; each of the 2N terms passes through at most
  * about a dozen operations, so every value is within 16 (2N + 2) 2^-104 times
  * the sum of the magnitudes of its terms.
  */
void farResidualDD(const FarStack& st, double th, double C, int ks, double dphi,
                   double dz, DD F[2], double bound[2]) {
    const std::size_t N = st.nMed(), T = st.nTerms();
    std::vector<DD> nu(N);
    const DD C2 = twoProd(C, C);
    for (std::size_t k = 0; k < N; ++k) {
        nu[k] = ddSqrt(twoProd(st.n[k], st.n[k]) - C2);
    }
    const DD lam_edge = nu[ks] * st.Rin[ks];
    DD sth, cth;
    ddSinCos(th, sth, cth);

    DD phi{3.141592653589793, 1.2246467991473532e-16}, z{};
    double sum_phi = kPi, sum_z = 0.0;
    for (std::size_t j = 0; j < T; ++j) {
        const int m = st.med[j];
        DD a, root;
        if (m == ks && j % 2 == 0) { // the grazing term, written in theta* as in farPoint
            a = DD{th, 0.0};
            root = cth;
        } else {
            DD x = lam_edge * sth / (nu[m] * st.rad[j]);
            if (x.hi > 1.0 || (x.hi == 1.0 && x.lo > 0.0)) {
                x = DD{1.0, 0.0};
            }
            a = ddAsin(x);
            root = ddSqrt(DD{1.0, 0.0} - x * x);
        }
        const DD zj = (DD{C, 0.0} / nu[m]) * root * (st.sig[j] * st.rad[j]);
        phi = phi + a * st.sig[j];
        z = z - zj;
        sum_phi += std::abs(a.hi);
        sum_z += std::abs(zj.hi);
    }
    F[0] = phi - DD{dphi, 0.0};
    F[1] = z - DD{dz, 0.0};
    const double rel = 16.0 * static_cast<double>(T + 2) * 4.93e-32;
    bound[0] = rel * (sum_phi + std::abs(dphi));
    bound[1] = rel * (sum_z + std::abs(dz));
}

} // namespace

std::vector<Seed> classifyFar(const Stack& s, double r_p, double r_x, double dphi,
                              double dz, double tol, FarSearchStats* stats,
                              int maxboxes) {
    const double sgn = dz >= 0.0 ? 1.0 : -1.0;
    dz = std::abs(dz);

    std::vector<Seed> out;
    if (dz == 0.0) { // level with the pinhole: the tilt is zero and Lambda alone solves
        for (const FarRoot& fr : farRoots(0.0, s, r_p, r_x, dphi)) {
            out.push_back(Seed{fr.lambda, 0.0});
        }
        return out;
    }

    const FarStack st = farStack(s, r_p, r_x);
    const double n_min = *std::min_element(s.n.begin(), s.n.end());
    // No ray steeper than C_max reaches the point (Property zbound).
    const double cmax = bisect(
        [&](double C) { return z0(C, s, r_p, r_x) - dz; }, 0.0, n_min * (1 - 1e-15),
        200);
    const double top = std::min(cmax * (1 + 1e-12) + 1e-15, n_min * (1 - 1e-15));

    std::vector<double> cuts{0.0};
    for (double t : switchTilts(s, r_x, top)) {
        cuts.push_back(t);
    }
    cuts.push_back(top);

    auto kstarAt = [&](double C) {
        int best = 0;
        double bestv = kInf;
        for (std::size_t k = 0; k < st.nMed(); ++k) {
            const double v =
                std::sqrt(std::max(st.n[k] * st.n[k] - C * C, 0.0)) * st.Rin[k];
            if (v < bestv) {
                bestv = v;
                best = static_cast<int>(k);
            }
        }
        return best;
    };

    std::vector<Box> work;
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
        Box b;
        b.ta = 0.0;
        b.tb = 0.5 * kPi;
        b.ca = cuts[i];
        b.cb = cuts[i + 1];
        b.pa = cuts[i];
        b.pb = cuts[i + 1];
        b.ks = kstarAt(0.5 * (b.ca + b.cb));
        work.push_back(b);
    }

    struct Found {
        double tm, cm;
        double Ta, Tb, Ca, Cb;
        int ks;
        double Y[2][2];
    };
    struct Loose {
        double tm, cm, pa, pb;
        int ks;
    };
    std::vector<Found> found;
    std::vector<Loose> loose;
    int examined = 0, precise = 0;
    bool capped = false;

    while (!work.empty()) {
        examined += static_cast<int>(work.size());
        if (examined > maxboxes) { // never expected; keep every box rather than drop it
            for (const Box& b : work) {
                loose.push_back(Loose{0.5 * (b.ta + b.tb), 0.5 * (b.ca + b.cb), b.pa,
                                      b.pb, b.ks});
            }
            capped = true;
            break;
        }

        std::vector<Box> next;
        for (const Box& b : work) {
            const BoxBounds bb = boxBounds(st, b);
            if (!(bb.plo <= dphi && dphi <= bb.phi && bb.zlo <= dz && dz <= bb.zhi)) {
                continue; // eq. (farbox) excludes the target: no ray here
            }

            const double tm = 0.5 * (b.ta + b.tb), cm = 0.5 * (b.ca + b.cb);
            const double wt = b.tb - b.ta, wc = b.cb - b.ca;
            // The test is applied to the box grown by 10%, so a ray on an edge
            // between two boxes lies inside one of them.
            const double Ta = std::max(b.ta - 0.1 * wt, 0.0);
            const double Tb = std::min(b.tb + 0.1 * wt, 0.5 * kPi);
            const double Ca = std::max(b.ca - 0.1 * wc, b.pa);
            const double Cb = std::min(b.cb + 0.1 * wc, b.pb);

            const FarPoint fp = farPoint(st, tm, cm, b.ks);
            const double det = fp.J[0][0] * fp.J[1][1] - fp.J[0][1] * fp.J[1][0];

            double Y[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            bool Y_ok = false;
            if (std::isfinite(det) && det != 0.0) {
                Y[0][0] = fp.J[1][1] / det;
                Y[0][1] = -fp.J[0][1] / det;
                Y[1][0] = -fp.J[1][0] / det;
                Y[1][1] = fp.J[0][0] / det;
                Y_ok = std::isfinite(Y[0][0]) && std::isfinite(Y[0][1]) &&
                       std::isfinite(Y[1][0]) && std::isfinite(Y[1][1]);
            }
            if (!Y_ok) {
                Y[0][0] = Y[0][1] = Y[1][0] = Y[1][1] = 0.0;
            }

            const double F[2] = {fp.phi - dphi, fp.z - dz};
            double v[2], err[2];
            for (int r = 0; r < 2; ++r) {
                v[r] = -(Y[r][0] * F[0] + Y[r][1] * F[1]);
                err[r] = std::abs(Y[r][0]) * fp.ephi + std::abs(Y[r][1]) * fp.ez;
            }
            // Next to a caustic Y is large, and the rounding of the residual can
            // swamp the box: recompute the residual in double-double there, as
            // the reference does in 34 digits.
            if (std::isfinite(det) && det != 0.0 &&
                (err[0] > 0.125 * 0.5 * (Tb - Ta) || err[1] > 0.125 * 0.5 * (Cb - Ca))) {
                DD Fd[2];
                double eF[2];
                farResidualDD(st, tm, cm, b.ks, dphi, dz, Fd, eF);
                const double x_mid[2] = {tm, cm};
                for (int r = 0; r < 2; ++r) {
                    v[r] = -(Fd[0] * Y[r][0] + Fd[1] * Y[r][1]).hi;
                    err[r] = 4.0 * kEps * (std::abs(v[r]) + std::abs(x_mid[r])) +
                             std::abs(Y[r][0]) * eF[0] + std::abs(Y[r][1]) * eF[1];
                }
                ++precise;
            }

            // Krawczyk's test, eq. (krawczyk).
            const Jac2 jb = farBoxJacobian(st, Ta, Tb, Ca, Cb, b.ks);
            double Klo[2] = {tm + v[0] - err[0], cm + v[1] - err[1]};
            double Khi[2] = {tm + v[0] + err[0], cm + v[1] + err[1]};
            const double lo_off[2] = {Ta - tm, Ca - cm};
            const double hi_off[2] = {Tb - tm, Cb - cm};
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 2; ++c) {
                    const Iv y0 = ivMul(Iv{Y[r][0], Y[r][0]},
                                        Iv{jb.lo[0][c], jb.hi[0][c]});
                    const Iv y1 = ivMul(Iv{Y[r][1], Y[r][1]},
                                        Iv{jb.lo[1][c], jb.hi[1][c]});
                    const double slack =
                        8.0 * kEps * (std::abs(y0.lo) + std::abs(y0.hi) +
                                      std::abs(y1.lo) + std::abs(y1.hi));
                    const double ident = (r == c) ? 1.0 : 0.0;
                    const Iv k = ivMul(Iv{ident - (y0.hi + y1.hi) - slack,
                                          ident - (y0.lo + y1.lo) + slack},
                                       Iv{lo_off[c], hi_off[c]});
                    Klo[r] += k.lo;
                    Khi[r] += k.hi;
                }
            }

            const double Blo[2] = {Ta, Ca}, Bhi[2] = {Tb, Cb};
            const bool fine = std::isfinite(Klo[0]) && std::isfinite(Klo[1]) &&
                              std::isfinite(Khi[0]) && std::isfinite(Khi[1]) &&
                              std::isfinite(det) && det != 0.0;
            bool one = fine, none = false;
            for (int r = 0; r < 2; ++r) {
                one = one && (Klo[r] > Blo[r]) && (Khi[r] < Bhi[r]);
                if (fine && (Khi[r] < Blo[r] || Klo[r] > Bhi[r])) {
                    none = true;
                }
            }

            if (one) {
                Found f{tm, cm, Ta, Tb, Ca, Cb, b.ks, {{0, 0}, {0, 0}}};
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 2; ++c) {
                        f.Y[r][c] = Y[r][c];
                    }
                }
                found.push_back(f);
                continue;
            }
            if (none) {
                continue; // certified empty
            }
            if (wt < tol && wc < tol) {
                loose.push_back(Loose{tm, cm, b.pa, b.pb, b.ks});
                continue; // on an edge, or within rounding of a caustic
            }

            // Halve across the side along which the Jacobian changes F more.
            const double zs = std::max(dz, 1e-9);
            double s_t = wt * (std::abs(fp.J[0][0]) + std::abs(fp.J[1][0]) / zs);
            double s_c = wc * (std::abs(fp.J[0][1]) + std::abs(fp.J[1][1]) / zs);
            if (!std::isfinite(s_t)) {
                s_t = wt;
            }
            if (!std::isfinite(s_c)) {
                s_c = wc;
            }
            const bool by_t = ((s_t >= s_c) && (wt >= tol)) || (wc < tol);
            Box lo_box = b, hi_box = b;
            if (by_t) {
                lo_box.tb = tm;
                hi_box.ta = tm;
            } else {
                lo_box.cb = cm;
                hi_box.ca = cm;
            }
            next.push_back(lo_box);
            next.push_back(hi_box);
        }
        work.swap(next);
    }

    // Step (4): polish each certified box, then each undecided one.
    struct RayBox {
        double x[2];
        int ks;
        double lo[2], hi[2];
    };
    std::vector<RayBox> rays;
    for (const Found& f : found) {
        const double lo[2] = {f.Ta, f.Ca}, hi[2] = {f.Tb, f.Cb};
        const NewtonOut nw =
            farNewton(st, f.tm, f.cm, f.ks, dphi, dz, lo, hi, f.Y, true);
        // Each of two boxes holds exactly one ray, so rays lying in each other's
        // boxes are the same ray.
        bool dup = false;
        for (const RayBox& r : rays) {
            const bool x_in_r = nw.th >= r.lo[0] && nw.th <= r.hi[0] &&
                                nw.c >= r.lo[1] && nw.c <= r.hi[1];
            const bool r_in_x = r.x[0] >= lo[0] && r.x[0] <= hi[0] &&
                                r.x[1] >= lo[1] && r.x[1] <= hi[1];
            if (x_in_r && r_in_x) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            rays.push_back(RayBox{{nw.th, nw.c}, f.ks, {lo[0], lo[1]}, {hi[0], hi[1]}});
        }
    }

    int extra = 0;
    for (const Loose& l : loose) {
        const double lo[2] = {0.0, l.pa}, hi[2] = {0.5 * kPi, l.pb};
        const double noY[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
        const NewtonOut nw =
            farNewton(st, l.tm, l.cm, l.ks, dphi, dz, lo, hi, noY, false);
        if (nw.res_phi <= std::max(64.0 * nw.ephi, 1e-13) &&
            nw.res_z <= std::max(64.0 * nw.ez, 1e-13 * std::max(dz, 1e-3))) {
            bool dup = false;
            for (const RayBox& r : rays) {
                if (std::abs(nw.th - r.x[0]) <= 1e-12 + 1e-9 * std::abs(r.x[0]) &&
                    std::abs(nw.c - r.x[1]) <= 1e-12 + 1e-9 * std::abs(r.x[1])) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                rays.push_back(RayBox{{nw.th, nw.c},
                                      l.ks,
                                      {nw.th, nw.c},
                                      {nw.th, nw.c}});
                ++extra;
            }
        }
    }

    if (stats) {
        stats->boxes = examined;
        stats->precise = precise;
        stats->undecided = static_cast<int>(loose.size());
        stats->from_undecided = extra;
        stats->capped = capped;
    }

    for (const RayBox& r : rays) {
        const double nuk =
            std::sqrt(std::max(st.n[r.ks] * st.n[r.ks] - r.x[1] * r.x[1], 0.0));
        out.push_back(Seed{nuk * st.Rin[r.ks] * std::sin(r.x[0]), sgn * r.x[1]});
    }
    return out;
}

SolveResult solveFar(const Stack& s, double r_p, double r_x, double dphi, double dz,
                     double lam0, double cz0, int itmax, double tol) {
    SolveResult res;
    double L = lam0, C = cz0;
    auto residual = [&](double LL, double CC, double& g0, double& g1) {
        const auto pz = phiZ(LL, CC, s, r_p, r_x, Branch::Far);
        if (!pz) {
            return false;
        }
        g0 = pz->phi - dphi;
        g1 = pz->z - dz;
        return true;
    };

    double g0 = 0.0, g1 = 0.0;
    bool have = residual(L, C, g0, g1);
    int it = 0; // the iterations taken, also after an early break
    for (; it < itmax; ++it) {
        if (!have) {
            res.iters = it;
            res.residual = kInf;
            res.lambda = L;
            res.c_z = C;
            return res;
        }
        const double gn = std::hypot(g0, g1);
        if (gn < tol) {
            return SolveResult{true, L, C, gn, it};
        }
        const double j11 = dPhiDLam(L, C, s, r_p, r_x, Branch::Far);
        const double j12 = dPhiDCz(L, C, s, r_p, r_x, Branch::Far);
        const auto j22o = dZDCz(L, C, s, r_p, r_x, Branch::Far);
        if (!j22o) {
            break;
        }
        const double det = j11 * *j22o - j12 * j12;
        if (!std::isfinite(det) || det == 0.0) {
            break;
        }
        const double p0 = -(*j22o * g0 - j12 * g1) / det;
        const double p1 = -(-j12 * g0 + j11 * g1) / det;

        // No convex program on this branch, so backtrack on the residual itself.
        double t = 1.0;
        bool moved = false;
        while (t > 1e-12) {
            const double Ln = L + t * p0, Cn = C + t * p1;
            double h0 = 0.0, h1 = 0.0;
            if (inDomain(Ln, Cn, s, r_x) && residual(Ln, Cn, h0, h1) &&
                std::hypot(h0, h1) < gn) {
                L = Ln;
                C = Cn;
                g0 = h0;
                g1 = h1;
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if (!moved) {
            break;
        }
    }
    res.lambda = L;
    res.c_z = C;
    res.iters = it;
    res.residual = have ? std::hypot(g0, g1) : kInf;
    res.ok = res.residual < tol;
    return res;
}

namespace {

/** @brief projectCylinder() on a valid input, every length in units of 2^floor(log2 r_P). */
std::vector<Ray> projectCylinderUnit(const Stack& s, const double pinhole[3],
                                     const double point[3], CylinderCoverage& cov,
                                     double tol, int itmax) {
    std::vector<Ray> out;
    const double r_p = std::hypot(pinhole[0], pinhole[1]);
    const double r_x = std::hypot(point[0], point[1]);
    const double dz = point[2] - pinhole[2];
    // Step (4) converges to the projection's tolerance when that is the stricter.
    const double stol = std::min(1e-10, tol);
    const int it_near = itmax >= 0 ? itmax : 80;
    const int it_far = itmax >= 0 ? itmax : 40;

    auto emit = [&](Branch b, double lam, double c_z) {
        Ray r;
        r.branch = b;
        r.lambda = lam;
        r.c_z = c_z;
        const std::vector<double> nu = nus(s.n, c_z);
        directionFromInvariants(pinhole, lam, c_z, s.n[0], nu[0], r.dir);
        out.push_back(r);
    };

    // Step (1): the signed azimuth from the pinhole to the point.
    const double d = std::remainder(std::atan2(point[1], point[0]) -
                                        std::atan2(pinhole[1], pinhole[0]),
                                    2.0 * kPi);

    if (r_x < 1e-15) { // step (1): on the axis, the answer is the radial ray
        const double n_min = *std::min_element(s.n.begin(), s.n.end());
        const double c = bisect(
            [&](double cc) { return z0(cc, s, r_p, r_x) - std::abs(dz); }, 0.0,
            n_min * (1 - 1e-15), 200);
        // Its branch is the limit r_X -> 0 of the general case: far when the
        // point lies beyond the axis as seen from the pinhole. Exactly on the
        // axis the azimuth is undefined and either label holds.
        const bool beyond = (point[0] != 0.0 || point[1] != 0.0) && std::abs(d) > 0.5 * kPi;
        emit(beyond ? Branch::Far : Branch::Near, 0.0, (dz >= 0.0 ? 1.0 : -1.0) * c);
        return out;
    }
    const double turn = d >= 0.0 ? 1.0 : -1.0;
    double dphi = std::abs(d);

    // Within 4 ulp of pi the two senses coincide and the rounding of the azimuth
    // alone decides which side of the domain edge the radial ray falls on.
    const double eps = std::numeric_limits<double>::epsilon();
    const bool opposite = std::abs(dphi - kPi) <= 8.0 * eps;

    // Property chord caps how far any ray can sweep, and with it how many
    // wrapped targets are worth solving.
    double top = kPi;
    for (std::size_t k = 0; k < s.R.size(); ++k) {
        top += std::acos(clamp1(s.R[k] / rOut(s, r_p, k)));
    }

    if (opposite) { // step (1): the far branch's radial ray, recorded once
        dphi = kPi;
        const double n_min = *std::min_element(s.n.begin(), s.n.end());
        const double c = bisect(
            [&](double cc) {
                const auto v = phiZ(0.0, cc, s, r_p, r_x, Branch::Far);
                return (v ? v->z : 0.0) - std::abs(dz);
            },
            0.0, n_min * (1 - 1e-15), 200);
        emit(Branch::Far, 0.0, (dz >= 0.0 ? 1.0 : -1.0) * c);
    }

    // Both senses at Delta phi = pi, otherwise the sense the target implies.
    const double senses_both[2] = {1.0, -1.0};

    for (int m = 0; dphi + 2.0 * kPi * m < top; ++m) {
        const double targets[2] = {dphi + 2.0 * kPi * m,
                                   2.0 * kPi * (m + 1) - dphi};
        const double target_sense[2] = {1.0, -1.0};
        for (int i = 0; i < 2; ++i) {
            const double target = targets[i];
            const double sense = target_sense[i];
            if (target >= top || (opposite && sense < 0.0)) {
                continue;
            }

            if (const auto seed = classifyNear(s, r_p, r_x, target, dz)) {
                const SolveResult sol =
                    solveCylinder(s, r_p, r_x, target, dz, seed->lambda, seed->c_z,
                                  Branch::Near, it_near, stol);
                // As the Python reference: an iterate is judged by its residual,
                // converged to the solver's own tolerance or not (next to
                // grazing no Newton step may be possible, yet the seed is good).
                if (sol.residual < tol && inDomain(sol.lambda, sol.c_z, s, r_x)) {
                    if (opposite) {
                        for (double sg : senses_both) {
                            emit(Branch::Near, turn * sg * sol.lambda, sol.c_z);
                        }
                    } else {
                        emit(Branch::Near, turn * sense * sol.lambda, sol.c_z);
                    }
                }
            }

            FarSearchStats fst;
            for (const Seed& sd : classifyFar(s, r_p, r_x, target, dz, 1e-13, &fst)) {
                const SolveResult sol = solveFar(s, r_p, r_x, target, dz, sd.lambda,
                                                 sd.c_z, it_far, stol);
                // As the Python reference: judged by its residual (audit A2).
                if (!(sol.residual < tol) || !inDomain(sol.lambda, sol.c_z, s, r_x)) {
                    continue;
                }
                if (opposite &&
                    std::abs(sol.lambda) <=
                        1e-9 * lmaxKstar(s, r_x, sol.c_z).lmax) {
                    continue; // the radial ray, already recorded in step (1)
                }
                if (opposite) {
                    for (double sg : senses_both) {
                        emit(Branch::Far, turn * sg * sol.lambda, sol.c_z);
                    }
                } else {
                    emit(Branch::Far, turn * sense * sol.lambda, sol.c_z);
                }
            }
            if (fst.capped) {
                cov.far_branch = false; // not certified: a ray may be missing
            }
        }
    }
    return out;
}

} // namespace

std::vector<Ray> projectCylinder(const Stack& s, const double pinhole[3],
                                 const double point[3],
                                 CylinderCoverage* coverage, double tol, int itmax) {
    CylinderCoverage cov; // both false: "not searched" until the model applies
    std::vector<Ray> out;
    const double r_p = std::hypot(pinhole[0], pinhole[1]);
    const double r_x = std::hypot(point[0], point[1]);
    // The model needs the pinhole outside the outermost interface and the point
    // strictly inside the innermost one. Otherwise nothing is searched, and the
    // coverage says so rather than certifying "no image".
    const bool applies =
        s.isValid() && r_p > 0.0 && std::isfinite(r_p) && std::isfinite(r_x) &&
        std::isfinite(point[2]) && std::isfinite(pinhole[2]) &&
        (s.R.empty() || (r_p > s.R.front() && r_x < s.R.back()));
    if (applies) {
        // Lengths in units of u = 2^floor(log2 r_P), so that every tolerance is
        // dimensionless. Dividing by a power of two is exact: the problem solved
        // is the one given to the last bit. (Dividing by r_P itself would round
        // every length, and next to grazing that alone costs ~eps/sqrt(delta) in Phi.)
        const double u = std::ldexp(1.0, std::ilogb(r_p));
        Stack unit = s;
        for (double& R : unit.R) {
            R /= u;
        }
        const double p[3] = {pinhole[0] / u, pinhole[1] / u, pinhole[2] / u};
        const double x[3] = {point[0] / u, point[1] / u, point[2] / u};
        cov.near_branch = true;
        cov.far_branch = true;
        out = projectCylinderUnit(unit, p, x, cov, tol, itmax);
        for (Ray& r : out) {
            r.lambda *= u;
        }
    }
    if (coverage) {
        *coverage = cov;
    }
    return out;
}

bool singleImage(const Stack& s, double r_p, double r_x, double dz) {
    const std::size_t N = s.nMedia();
    if (!s.isValid() || N < 2 || !(r_x > 0.0) || !(r_p > s.R.front()) ||
        !(r_x < s.R.back()) || !std::isfinite(dz)) {
        return false;
    }
    // Property onedens: one densifying interface at most.
    int densifying = 0;
    for (std::size_t k = 0; k + 1 < N; ++k) {
        if (s.n[k] < s.n[k + 1]) {
            ++densifying;
        }
    }
    if (densifying > 1) {
        return false;
    }
    // k* = N while nu_k R_k > nu_N r_X for every k < N, and
    // nu_k^2 R_k^2 - nu_N^2 r_X^2 = (n_k^2 R_k^2 - n_N^2 r_X^2) - C_z^2 (R_k^2 - r_X^2),
    // which falls with C_z^2 and reaches 0 at C_z^2 = (n_k^2 R_k^2 - n_N^2 r_X^2) / (R_k^2 - r_X^2).
    const double nN = s.n[N - 1];
    double c1sq = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k + 1 < N; ++k) {
        const double R = s.R[k];
        const double g0 = s.n[k] * s.n[k] * R * R - nN * nN * r_x * r_x;
        const double den = R * R - r_x * r_x;
        if (!(g0 > 1e-9 * s.n[k] * s.n[k] * R * R) || !(den > 0.0)) {
            return false; // r_X >= r_c(0), to the margin
        }
        c1sq = std::min(c1sq, g0 / den);
    }
    const double n_min = *std::min_element(s.n.begin(), s.n.end());
    const double c1 = std::sqrt(std::min(c1sq, n_min * n_min));
    // Property zbound; z0 is infinite when C1 reaches the least index.
    return std::abs(dz) < (1.0 - 1e-9) * z0(c1, s, r_p, r_x);
}

std::optional<Ray> projectCylinderFrom(const Stack& s, const double pinhole[3],
                                       const double point[3], double lam0,
                                       double cz0, bool far0, double tol,
                                       int itmax) {
    constexpr int kHintIters = 30;
    const double r_p = std::hypot(pinhole[0], pinhole[1]);
    const double r_x = std::hypot(point[0], point[1]);
    const bool applies = s.isValid() && !s.R.empty() && r_p > 0.0 &&
                         std::isfinite(r_p) && std::isfinite(r_x) &&
                         std::isfinite(point[2]) && std::isfinite(pinhole[2]) &&
                         std::isfinite(lam0) && std::isfinite(cz0) &&
                         r_p > s.R.front() && r_x < s.R.back();
    if (!applies) {
        return std::nullopt;
    }
    // The units of projectCylinder(), so that `tol` means the same there and here.
    const double u = std::ldexp(1.0, std::ilogb(r_p));
    Stack unit = s;
    for (double& R : unit.R) {
        R /= u;
    }
    const double p[3] = {pinhole[0] / u, pinhole[1] / u, pinhole[2] / u};
    const double x[3] = {point[0] / u, point[1] / u, point[2] / u};
    const double rp = std::hypot(p[0], p[1]);
    const double rx = std::hypot(x[0], x[1]);
    const double dz = x[2] - p[2];
    if (rx < 1e-15) {
        return std::nullopt; // on the axis: step (1) of the full search
    }
    // Step (1): the signed azimuth from the pinhole to the point, as there.
    const double d = std::remainder(std::atan2(x[1], x[0]) - std::atan2(p[1], p[0]),
                                    2.0 * kPi);
    const double dphi = std::abs(d);
    const double turn = d >= 0.0 ? 1.0 : -1.0;
    if (std::abs(dphi - kPi) <= 8.0 * std::numeric_limits<double>::epsilon()) {
        return std::nullopt; // opposite the pinhole: both senses, left to the full search
    }
    if (!singleImage(unit, rp, rx, dz)) {
        return std::nullopt;
    }

    // The start, inside the domain.
    const double n_min = *std::min_element(unit.n.begin(), unit.n.end());
    const double c0 = std::abs(cz0) < n_min - 1e-12 ? cz0 : 0.0;
    const double lmax0 = lmaxKstar(unit, rx, c0).lmax;
    const double l0 = std::min(std::abs(lam0) / u, lmax0 * (1.0 - 1e-9));
    // The near branch reaches dphi at this tilt iff its reach does (it is
    // increasing in Lambda, Property convexnear).
    bool far_first = far0;
    if (const auto reach = phiZ(lmax0, c0, unit, rp, rx, Branch::Near)) {
        far_first = dphi > reach->phi;
    }
    // From a start this close, Newton stops at its first residual below its
    // target, which a step later would be at rounding; projectCylinder()'s
    // longer solves usually end there. Aim lower, so that both give the same
    // ray to rounding (F1b: a stop at 1e-10 alone is 2e-6 px along the axis).
    const double stol = std::min(1e-13, tol);
    const int it = itmax >= 0 ? std::min(itmax, kHintIters) : kHintIters;
    for (int pass = 0; pass < 2; ++pass) {
        const bool far = (pass == 0) ? far_first : !far_first;
        const SolveResult sol =
            far ? solveFar(unit, rp, rx, dphi, dz, l0, c0, it, stol)
                : solveCylinder(unit, rp, rx, dphi, dz, l0, c0, Branch::Near, it, stol);
        // The acceptance test of projectCylinder().
        if (sol.residual < tol && inDomain(sol.lambda, sol.c_z, unit, rx)) {
            Ray r;
            r.branch = far ? Branch::Far : Branch::Near;
            r.c_z = sol.c_z;
            const double lam = turn * sol.lambda;
            const std::vector<double> nu = nus(unit.n, sol.c_z);
            directionFromInvariants(p, lam, sol.c_z, unit.n[0], nu[0], r.dir);
            r.lambda = lam * u;
            return r;
        }
    }
    return std::nullopt;
}

void directionFromInvariants(const double pinhole[3], double lam, double c_z,
                             double n1, double nu1, double out[3]) {
    const double r_p = std::hypot(pinhole[0], pinhole[1]);
    const double rhat[3] = {pinhole[0] / r_p, pinhole[1] / r_p, 0.0};
    const double phat[3] = {-pinhole[1] / r_p, pinhole[0] / r_p, 0.0};
    const double sa = clamp1(lam / (nu1 * r_p));
    const double ca = std::sqrt(std::max(1.0 - sa * sa, 0.0));
    const double scale = nu1 / n1;
    for (int i = 0; i < 3; ++i) {
        out[i] = scale * (-ca * rhat[i] + sa * phat[i]);
    }
    out[2] += c_z / n1;
}

SolveResult solveCylinder(const Stack& s, double r_p, double r_x, double dphi,
                          double dz, double lam0, double cz0, Branch b,
                          int itmax, double tol) {
    SolveResult res;
    // G = -W - Lambda dphi - C_z dz has gradient (Phi - dphi, Z - dz), and is
    // strictly convex on the near branch (Theorem convex).
    auto G = [&](double L, double C) -> std::optional<double> {
        const auto w = radialAction(L, C, s, r_p, r_x, b);
        if (!w) {
            return std::nullopt;
        }
        return -*w - L * dphi - C * dz;
    };

    double L = lam0, C = cz0;
    if (!inDomain(L, C, s, r_x)) {
        const double l_max = lmaxKstar(s, r_x, C).lmax;
        L = std::clamp(L, 0.0, 0.5 * l_max);
    }

    for (int it = 0; it < itmax; ++it) {
        const auto pz = phiZ(L, C, s, r_p, r_x, b);
        if (!pz) {
            res.iters = it;
            res.residual = std::numeric_limits<double>::infinity();
            return res;
        }
        const double g0v = pz->phi - dphi;
        const double g1v = pz->z - dz;
        const double gnorm = std::hypot(g0v, g1v);
        if (gnorm < tol) {
            return SolveResult{true, L, C, gnorm, it};
        }

        const double j11 = dPhiDLam(L, C, s, r_p, r_x, b);
        const double j12 = dPhiDCz(L, C, s, r_p, r_x, b); // = dZ/dLambda, Property symmetry
        const auto j22o = dZDCz(L, C, s, r_p, r_x, b);
        if (!j22o) { // no Newton step: the caller judges the iterate by its residual
            res.iters = it;
            res.residual = gnorm;
            res.lambda = L;
            res.c_z = C;
            return res;
        }
        const double j22 = *j22o;
        const double det = j11 * j22 - j12 * j12;
        if (!std::isfinite(det) || det == 0.0) {
            res.iters = it;
            res.residual = gnorm;
            res.lambda = L;
            res.c_z = C;
            return res;
        }
        const double p0 = -(j22 * g0v - j12 * g1v) / det;
        const double p1 = -(-j12 * g0v + j11 * g1v) / det;

        const auto gcur = G(L, C);
        if (!gcur) {
            res.iters = it;
            res.residual = gnorm;
            res.lambda = L;
            res.c_z = C;
            return res;
        }
        const double slope = p0 * g0v + p1 * g1v;

        // G is stationary at the solution, so below this floor an Armijo test on
        // it is meaningless and the full Newton step is taken.
        if (std::abs(slope) < 1e-14 * (1.0 + std::abs(*gcur))) {
            const double Ln = L + p0, Cn = C + p1;
            if (inDomain(Ln, Cn, s, r_x)) {
                L = Ln;
                C = Cn;
                continue;
            }
        }

        double step = 1.0;
        bool moved = false;
        for (int j = 0; j < 60; ++j) {
            const double Ln = L + step * p0, Cn = C + step * p1;
            if (inDomain(Ln, Cn, s, r_x)) {
                const auto gn = G(Ln, Cn);
                if (gn && *gn <= *gcur + 1e-4 * step * slope) {
                    moved = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if (!moved) {
            res.iters = it;
            res.residual = gnorm;
            res.lambda = L;
            res.c_z = C;
            return res;
        }
        L += step * p0;
        C += step * p1;
    }

    const auto pz = phiZ(L, C, s, r_p, r_x, b);
    res.lambda = L;
    res.c_z = C;
    res.iters = itmax;
    res.residual = pz ? std::hypot(pz->phi - dphi, pz->z - dz)
                      : std::numeric_limits<double>::infinity();
    res.ok = res.residual < tol;
    return res;
}

} // namespace refract_curved
