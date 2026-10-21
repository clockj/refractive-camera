"""Forward projection through coaxial cylindrical interfaces: the reference implementation, in pure Python.

The algorithm of the paper (S. Zhong et al., arXiv:2610.17413, Section "Projection algorithm"): the near branch is
classified by the reach and zeta and solved as a convex program; the far branch is searched by boxes in
(theta*, C_z), each certified by Krawczyk's test, and polished by Newton; the targets of rays that sweep past pi are
searched too.  project() returns every ray as (branch, Lambda, C_z, direction at the pinhole), in the frame whose
z axis is the cylinders' axis.

Every function below is copied verbatim from the authors' reference implementation, whose comments cite the
sections and equations of their derivation notes; the paper presents the same results.  The compiled backend
(_core, from RefractCurved.cpp) mirrors these functions term for term and is the default; this module is the
readable reference and the fallback when no compiler is available.
"""
import math

import numpy as np

nus = lambda n, Cz: np.sqrt(np.asarray(n, float) ** 2 - Cz ** 2)


q = lambda a, L: 1.0 / np.sqrt(a * a - L * L)


# At Lambda = Lmax the binding medium's impact parameter b = Lambda/nu equals its radius, but rounding can put it a few
# ulp above; such a b is the grazing ray, not a ray that misses, so it is clamped to the radius (see Phi_Z).
_GRAZE = 8 * np.finfo(float).eps


def Lmax_kstar(n, R, rX, Cz):
    nu = nus(n, Cz)
    v = [nu[k] * r for k, r in enumerate(list(R) + [rX])]
    return min(v), int(np.argmin(v)) + 1


def Phi_Z(Lam, Cz, n, R, rP, rX, far=False):
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    phi = z = 0.0
    for k in range(len(n)):
        b = Lam / nu[k]
        if b > Rin[k]:
            if b > Rin[k] * (1 + _GRAZE):
                return None
            b = Rin[k]                            # Lambda = Lmax to rounding: the grazing ray of this medium
        a1 = np.arcsin(np.clip(b / Rin[k], -1, 1))
        a2 = np.arcsin(np.clip(b / Rout[k], -1, 1))
        s1 = np.sqrt(max(Rout[k] ** 2 - b * b, 0.0))
        s2 = np.sqrt(max(Rin[k] ** 2 - b * b, 0.0))
        if far and k == len(n) - 1:
            phi += np.pi - a1 - a2
            z += Cz * (s1 + s2) / nu[k]
        else:
            phi += a1 - a2
            z += Cz * (s1 - s2) / nu[k]
    return phi, z


def W_closed(Lam, Cz, n, R, rP, rX, far=False):
    """Radial action in elementary form."""
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    # At b = 0 the arccos term is 0, its limit, also at r = b = 0 (the far turning point of the radial ray), where
    # b / r would be 0/0.
    F = lambda r, b, v: v * (np.sqrt(max(r * r - b * b, 0.0)) - (b * np.arccos(np.clip(b / r, -1, 1)) if b else 0.0))
    tot = 0.0
    for k in range(len(n)):
        b = Lam / nu[k]
        if b > Rin[k]:
            if b > Rin[k] * (1 + _GRAZE):
                return None
            b = Rin[k]                            # as in Phi_Z
        if far and k == len(n) - 1:
            tot += (F(Rout[k], b, nu[k]) - F(b, b, nu[k])) + (F(Rin[k], b, nu[k]) - F(b, b, nu[k]))
        else:
            tot += F(Rout[k], b, nu[k]) - F(Rin[k], b, nu[k])
    return tot


def dPhi_dLam(Lam, Cz, n, R, rP, rX, far=False):
    nu = nus(n, Cz)
    S = sum(q(nu[k] * R[k], Lam) - q(nu[k + 1] * R[k], Lam) for k in range(len(R)))
    return S + (-1.0 if far else 1.0) * q(nu[-1] * rX, Lam) - q(nu[0] * rP, Lam)


def dPhi_dCz(Lam, Cz, n, R, rP, rX, far=False):
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    v = Cz * Lam * sum((q(nu[k] * Rin[k], Lam) - q(nu[k] * Rout[k], Lam)) / nu[k] ** 2
                       for k in range(len(n)))
    if far:
        v -= 2 * Cz * Lam * q(nu[-1] * rX, Lam) / nu[-1] ** 2
    return v


def dZ_dCz(Lam, Cz, n, R, rP, rX, far=False, h=1e-7):
    """Central difference; one-sided where Lambda is so close to Lmax that one side leaves the domain (the tilt that
    lowers nu_k lowers Lmax), None only if both sides do."""
    a = Phi_Z(Lam, Cz + h, n, R, rP, rX, far)
    b = Phi_Z(Lam, Cz - h, n, R, rP, rX, far)
    if a is not None and b is not None:
        return (a[1] - b[1]) / (2 * h)
    c = Phi_Z(Lam, Cz, n, R, rP, rX, far)
    if c is None or (a is None and b is None):
        return None
    return (a[1] - c[1]) / h if a is not None else (c[1] - b[1]) / h


def in_domain(Lam, Cz, n, R, rX):
    if abs(Cz) >= min(n) - 1e-12:
        return False
    Lm, _ = Lmax_kstar(n, R, rX, Cz)
    return 0.0 <= Lam < Lm


def solve(n, R, rP, rX, dphi, dz, Lam0, Cz0, far=False, itmax=80, tol=1e-10):
    """Damped Newton on G = -W - Lam*dphi - Cz*dz  (grad G = (Phi-dphi, Z-dz))."""
    def G(L, C):
        w = W_closed(L, C, n, R, rP, rX, far)
        return None if w is None else -w - L * dphi - C * dz

    L, C = Lam0, Cz0
    if not in_domain(L, C, n, R, rX):
        Lm, _ = Lmax_kstar(n, R, rX, C)
        L = min(max(L, 0.0), 0.5 * Lm)
    for it in range(itmax):
        pz = Phi_Z(L, C, n, R, rP, rX, far)
        if pz is None:
            return None, it, np.inf
        g = np.array([pz[0] - dphi, pz[1] - dz])
        if np.linalg.norm(g) < tol:
            return (L, C), it, np.linalg.norm(g)
        j11 = dPhi_dLam(L, C, n, R, rP, rX, far)
        j12 = dPhi_dCz(L, C, n, R, rP, rX, far)
        j22 = dZ_dCz(L, C, n, R, rP, rX, far)
        if j22 is None:                           # no Newton step: the caller judges the iterate by its residual
            return (L, C), it, np.linalg.norm(g)
        J = np.array([[j11, j12], [j12, j22]])
        try:
            p = -np.linalg.solve(J, g)
        except np.linalg.LinAlgError:
            return (L, C), it, np.linalg.norm(g)
        g0 = G(L, C)
        slope = np.dot(p, g)
        # G is stationary at the solution, so its value saturates while |g| is
        # still ~sqrt(eps); below that floor an Armijo test on G is meaningless
        # and the full Newton step is taken.
        if abs(slope) < 1e-14 * (1.0 + abs(g0)):
            Ln, Cn = L + p[0], C + p[1]
            if in_domain(Ln, Cn, n, R, rX):
                L, C = Ln, Cn
                continue
        s = 1.0
        for _ in range(60):                       # backtracking, staying inside D
            Ln, Cn = L + s * p[0], C + s * p[1]
            if in_domain(Ln, Cn, n, R, rX):
                gn = G(Ln, Cn)
                if gn is not None and gn <= g0 + 1e-4 * s * np.dot(p, g):
                    break
            s *= 0.5
        else:
            return (L, C), it, np.linalg.norm(g)
        L, C = L + s * p[0], C + s * p[1]
    pz = Phi_Z(L, C, n, R, rP, rX, far)
    return (L, C), itmax, (np.inf if pz is None else
                           np.linalg.norm([pz[0] - dphi, pz[1] - dz]))


def direction_from_invariants(P0, Lam, Cz, n1, nu1):
    """Step (5) of the algorithm: rebuild d at the pinhole."""
    rP = np.hypot(P0[0], P0[1])
    rhat = np.array([P0[0] / rP, P0[1] / rP, 0.0])
    phat = np.array([-P0[1] / rP, P0[0] / rP, 0.0])
    sa = np.clip(Lam / (nu1 * rP), -1.0, 1.0)
    ca = np.sqrt(max(1.0 - sa * sa, 0.0))
    dperp = (nu1 / n1) * (-ca * rhat + sa * phat)
    return dperp + np.array([0.0, 0.0, Cz / n1])


def lam_at(Cz, n, R, rP, rX, dphi, iters=90):
    """Bisection for Phi_near(Lambda; Cz) = dphi.  None if out of reach at this Cz.

    Relies only on Property convexnear: Phi_near is strictly increasing in Lambda.  The reach is taken at Lmax itself,
    and the bisection runs in s with Lambda = Lmax (1 - s^2).  Phi_near has a square-root edge at Lmax, so a margin of
    m in Lambda would cost about sqrt(m) in Phi (4.5e-7 rad for m = 1e-13 at the reference port) and hide every tilt
    whose reach exceeds dphi by less; in s the edge is regular.  The result stays strictly below Lmax, inside the domain
    of solve()."""
    if abs(Cz) >= min(n):
        return None
    Lm, _ = Lmax_kstar(n, R, rX, Cz)
    top = Phi_Z(Lm, Cz, n, R, rP, rX)
    if top is None or top[0] < dphi:
        return None
    lo, hi = 0.0, 1.0                             # s; Phi_near falls as s grows, from the reach at s = 0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        v = Phi_Z(Lm * (1 - m * m), Cz, n, R, rP, rX)
        lo, hi = (lo, m) if (v is not None and v[0] < dphi) else (m, hi)
    s = 0.5 * (lo + hi)
    return min(Lm * (1 - s * s), np.nextafter(Lm, 0.0))


def switch_tilts(n, R, rX, chi):
    """Tilts in (0, chi) at which two caps nu_k R_k of (adm) are equal (R_N = r_X); k* is fixed between them."""
    caps = list(R) + [rX]
    out = set()
    for j in range(len(n)):
        for k in range(j + 1, len(n)):
            den = caps[j] ** 2 - caps[k] ** 2
            if den != 0.0:
                c2 = (n[j] ** 2 * caps[j] ** 2 - n[k] ** 2 * caps[k] ** 2) / den
                if 0.0 < c2 < chi * chi:
                    out.add(float(np.sqrt(c2)))
    return sorted(out)


def reach_terms(Cz, n, R, rP, rX):
    """The reach Phi_near(Lmax(C_z); C_z) medium by medium; each term is monotone in C_z while k* is fixed."""
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    Lm = min(nu[k] * Rin[k] for k in range(len(n)))
    return np.array([np.arcsin(min(Lm / (nu[k] * Rin[k]), 1.0)) - np.arcsin(Lm / (nu[k] * Rout[k]))
                     for k in range(len(n))])


def components(n, R, rP, rX, dphi, chi=None, tol=1e-13):
    """Connected pieces of {C_z in [0, chi] : reach(C_z) >= dphi}, found exactly (Property reachpieces).

    The range is cut at the tilts where k* can switch.  On a piece every medium's term of the reach is monotone
    in C_z, so on [a, b] the reach lies between sum_k min(T_k(a), T_k(b)) and sum_k max(T_k(a), T_k(b)): an
    interval is dropped when the upper bound is below dphi, kept when the lower bound reaches it, and halved
    otherwise.  An interval still undecided at width tol is kept, so no component is lost; a spurious sliver
    only costs classify() a failed look for zeta."""
    chi = min(n) * (1 - 1e-9) if chi is None else chi
    cuts = [0.0] + switch_tilts(n, R, rX, chi) + [chi]
    kept = []
    for a0, b0 in zip(cuts[:-1], cuts[1:]):
        stack = [(a0, b0, reach_terms(a0, n, R, rP, rX), reach_terms(b0, n, R, rP, rX))]
        while stack:
            a, b, ta, tb = stack.pop()
            if np.maximum(ta, tb).sum() < dphi:
                continue
            if np.minimum(ta, tb).sum() >= dphi or b - a < tol:
                kept.append((a, b))
                continue
            m = 0.5 * (a + b)
            tm = reach_terms(m, n, R, rP, rX)
            stack += [(m, b, tm, tb), (a, m, ta, tm)]
    out = []
    for a, b in sorted(kept):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def classify(n, R, rP, rX, dphi, dz, iters=60):
    """Step (3): is (dphi, dz) reachable on the near branch?  Seed (Lam, Cz) or None.

    zeta(Cz) = Z(Lambda(Cz), Cz) is strictly increasing on each component
    (Property zeta: zeta' = det J / Phi_Lambda > 0), so one bisection per component
    is both sound and complete there.  At a component end where the reach is dphi,
    Lambda(Cz) = Lmax: the ray is the grazing ray, and zeta there is its axial offset
    Z(Lmax, Cz) in closed form.  The bisection therefore brackets the whole component.
    (Approaching the ends from inside instead lost the rays of tilts next to an end,
    within about 1e-10 of grazing: evidence E1 of the paper, and the tests in
    code/tests/classify_grazing/.)"""
    sgn = 1.0 if dz >= 0 else -1.0                # zeta is odd in C_z
    target = abs(dz)

    def zeta(c):
        """zeta at c; at a tilt whose reach is dphi to rounding, the grazing ray's Z(Lmax, c)."""
        v = zeta_at(c, n, R, rP, rX, dphi)
        if v is None:
            pz = Phi_Z(Lmax_kstar(n, R, rX, c)[0], c, n, R, rP, rX)
            if pz is not None and pz[0] >= dphi - 1e-9:
                v = pz[1]
        return v

    def zin(c0, c1):
        """First finite zeta walking from c0 towards c1: the fallback for an end where zeta() has no value.

        C_z = 0 is a regular point with zeta(0) = 0, so it is used as it stands."""
        steps = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
        if c0 == 0.0:
            steps = (0.0,) + steps
        for t in steps:
            c = c0 + t * (c1 - c0)
            v = zeta(c)
            if v is not None:
                return c, v
        return None, None

    for (c1, c2) in components(n, R, rP, rX, dphi):
        if c2 <= c1:
            continue
        a, za = c1, zeta(c1)
        if za is None:
            a, za = zin(c1, c2)
        b, zb = c2, zeta(c2)
        if zb is None:
            b, zb = zin(c2, c1)
        if a is None or b is None or b <= a:
            continue
        if not (min(za, zb) <= target <= max(za, zb)):
            continue
        for _ in range(iters):
            mm = 0.5 * (a + b)
            v = zeta(mm)
            if v is None:
                break
            a, b = (mm, b) if (v - target) * (za - target) > 0 else (a, mm)
        c = 0.5 * (a + b)
        L = lam_at(c, n, R, rP, rX, dphi)
        if L is None:                             # c is within rounding of the end: the grazing ray
            Lm = Lmax_kstar(n, R, rX, c)[0]
            pz = Phi_Z(Lm, c, n, R, rP, rX)
            if pz is None or pz[0] < dphi - 1e-9:
                continue
            L = np.nextafter(Lm, 0.0)
        return L, sgn * c
    return None


def zeta_at(Cz, n, R, rP, rX, dphi):
    L = lam_at(Cz, n, R, rP, rX, dphi, iters=60)
    if L is None:
        return None
    v = Phi_Z(L, Cz, n, R, rP, rX)
    return None if v is None else v[1]


def Z0(Cz, n, R, rP, rX):
    """Eq. (z0): the axial offset of the radial ray; no ray of tilt C_z has |Z| below it (Property zbound)."""
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    return abs(Cz) * sum((Rout[k] - Rin[k]) / nu[k] for k in range(len(n)))


def _bisect(f, lo, hi, it=70):
    """Bisection on a sign change of f in [lo, hi]; lo itself if f vanishes there."""
    f0 = f(lo)
    if f0 == 0:
        return lo
    up = f0 > 0
    for _ in range(it):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if (f(m) > 0) == up else (lo, m)
    return 0.5 * (lo + hi)


_ARC = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, 801))    # Lambda/Lmax, dense at both ends


def far_turns(Cz, n, R, rP, rX, tol=1e-14):
    """Every turning point of Phi_far(Lambda; C_z) on (0, Lmax), none missed (Section fold, eq. intervalbound).

    With t = Lambda^2, Phi_far' = F(t) = sum_j c_j (a_j^2 - t)^(-1/2) with the (c_j, a_j) of (Flist), and every
    term increases in t.  On [t1, t2], F lies between (positive terms at t1) + (negative terms at t2) and the
    reverse, so a cell whose bounds exclude 0 holds no turning point.  Beyond Lmax^2 - c_0^2/M^2 the terms at
    a_j = Lmax dominate and fix the sign.  Cells of a grid dense at both ends are decided at once; the rest are
    halved, and every sign change is refined to tol Lmax^2.  A pair of turning points closer than that is a
    tangency of Phi_far' with 0, whose arc sweeps nothing."""
    N = len(n)
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    c = np.array([1.0 if k < N - 1 else -1.0 for k in range(N)] + [-1.0] * N)
    A = np.array([(nu[k] * Rin[k]) ** 2 for k in range(N)] + [(nu[k] * Rout[k]) ** 2 for k in range(N)])
    A0 = A.min()
    edge = A <= A0 * (1 + 1e-12)
    c0, M = c[edge].sum(), np.sum(np.abs(c[~edge]) / np.sqrt(A[~edge] - A0))
    t_end = A0 * (1 - 1e-14)
    if c0 != 0.0:
        t_end = min(t_end, A0 - (c0 / M) ** 2 if M > 0 else -1.0)
    if t_end <= 0.0:
        return []
    pos, neg = c > 0, c < 0
    Fv = lambda t: float(np.sum(c / np.sqrt(A - t)))

    def bounds(v1, v2):
        return v1[pos].sum() + v2[neg].sum(), v2[pos].sum() + v1[neg].sum()

    def halve(t1, t2, f1, f2):
        lo, hi = bounds(c / np.sqrt(A - t1), c / np.sqrt(A - t2))
        if lo > 0 or hi < 0:
            return []
        if t2 - t1 <= tol * A0:
            return [0.5 * (t1 + t2)] if (f1 > 0) != (f2 > 0) else []
        tm = 0.5 * (t1 + t2)
        fm = Fv(tm)
        return halve(t1, tm, f1, fm) + halve(tm, t2, fm, f2)

    t = t_end * _ARC ** 2
    V = c[None, :] / np.sqrt(A[None, :] - t[:, None])
    Fg = V.sum(1)
    lo = V[:-1][:, pos].sum(1) + V[1:][:, neg].sum(1)
    hi = V[1:][:, pos].sum(1) + V[:-1][:, neg].sum(1)
    out = []
    for i in np.flatnonzero(~((lo > 0) | (hi < 0))):
        out += halve(t[i], t[i + 1], Fg[i], Fg[i + 1])
    return [float(np.sqrt(x)) for x in out]


def far_roots(Cz, n, R, rP, rX, target):
    """Roots of Phi_far(Lambda; C_z) = target as (Lambda, slope sign of their arc, Z_far), by increasing Lambda.

    [0, Lmax) is split at the turning points of Phi_far, all of them located by far_turns(); an arc whose end
    values bracket the target holds exactly one root.  A root exactly at an end of an arc, such as Lambda = 0 at
    target pi, is counted once, as the root of the arc it starts.  The last arc ends at Lmax itself: a margin of m
    there would drop the roots of the targets within about sqrt(m) of Phi_far(Lmax), as in lam_at()."""
    Lm = Lmax_kstar(n, R, rX, Cz)[0]
    cuts = [0.0] + [L for L in far_turns(Cz, n, R, rP, rX) if 0.0 < L < Lm] + [Lm]
    phi = lambda L: Phi_Z(L, Cz, n, R, rP, rX, far=True)[0]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        fa, fb = phi(a), phi(b)
        if fa == target:
            L = a
        elif fb != target and (fa > target) != (fb > target):
            L = _bisect(lambda x: phi(x) - target, a, b)
        else:
            continue
        out.append((L, 1 if fb > fa else -1, Phi_Z(L, Cz, n, R, rP, rX, far=True)[1]))
    return out


# Lambda = Lmax(C_z) sin(theta*), eq. (thetastar).  Between two cuts (eq. cuts) k* is fixed, and every term of
# Phi_far and Z_far is monotone in theta* and in C_z, or a product of two nonnegative such factors, so the values at
# the corners of a box bound them over it (Property farbox).  The Jacobian in (theta*, C_z) is bounded term by term
# as well, and Krawczyk's test (eq. krawczyk) then shows that a box holds exactly one ray, or none.
_EPS = np.finfo(float).eps


def _far_stack(n, R, rP, rX):
    """Indices, inner and outer radii of the N media, and the medium k_j, radius rho_j and sign c_j of the 2N terms of
    (Flist)."""
    n = np.asarray(n, float)
    N = len(n)
    Rin, Rout = np.array(list(R) + [rX], float), np.array([rP] + list(R), float)
    med = np.repeat(np.arange(N), 2)
    rad = np.column_stack([Rin, Rout]).ravel()
    sig = np.column_stack([np.where(np.arange(N) < N - 1, 1.0, -1.0), -np.ones(N)]).ravel()
    return n, Rin, Rout, med, rad, sig


def box_bounds(st, ta, tb, ca, cb, ks, far=True):
    """Eq. (farbox): lower and upper bounds on Phi and Z of one branch over the boxes [ta, tb] x [ca, cb], k* = ks.

    b_k = R_k* (nu_k*/nu_k) sin(theta*) increases with theta* and is monotone in C_z between cuts; medium k's sweep
    and chord increase with b_k, except that medium N's decrease on the far branch; C_z/nu_k increases with C_z."""
    n, Rin, Rout = st[:3]
    i = np.arange(ca.size)
    nua, nub = np.sqrt(n ** 2 - ca[:, None] ** 2), np.sqrt(n ** 2 - cb[:, None] ** 2)
    ra, rb = Rin[ks][:, None] * nua[i, ks][:, None] / nua, Rin[ks][:, None] * nub[i, ks][:, None] / nub
    blo = np.minimum(np.sin(ta)[:, None] * np.minimum(ra, rb), Rin)
    bhi = np.minimum(np.sin(tb)[:, None] * np.maximum(ra, rb), Rin)
    sweep = lambda b: np.arcsin(b / Rin) - np.arcsin(b / Rout)
    chord = lambda b: np.sqrt(Rout ** 2 - b * b) - np.sqrt(np.maximum(Rin ** 2 - b * b, 0.0))
    plo, phi, slo, shi = sweep(blo), sweep(bhi), chord(blo), chord(bhi)
    fsweep = lambda b: np.pi - np.arcsin(b / Rin[-1]) - np.arcsin(b / Rout[-1])
    fchord = lambda b: np.sqrt(Rout[-1] ** 2 - b * b) + np.sqrt(np.maximum(Rin[-1] ** 2 - b * b, 0.0))
    if far:
        plo[:, -1], phi[:, -1] = fsweep(bhi[:, -1]), fsweep(blo[:, -1])
        slo[:, -1], shi[:, -1] = fchord(bhi[:, -1]), fchord(blo[:, -1])
    zlo, zhi = (ca[:, None] / nua * slo).sum(1), (cb[:, None] / nub * shi).sum(1)
    amp = 1.0 + (n ** 2 / nub ** 2).max(1)                  # rounding grows as nu -> 0
    pad = 64 * _EPS * amp
    return plo.sum(1) - pad * 10, phi.sum(1) + pad * 10, zlo * (1 - pad) - pad * 1e-3, zhi * (1 + pad) + pad * 1e-3


def _imul(alo, ahi, blo, bhi):
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    bad = np.isnan(p).any(0)
    return np.where(bad, -np.inf, p.min(0)), np.where(bad, np.inf, p.max(0))


def far_box_jacobian(st, ta, tb, ca, cb, ks):
    """Eq. (farjac): bounds on [[dPhi/dtheta*, dPhi/dC_z], [dZ/dtheta*, dZ/dC_z]] of the far branch over the boxes.

    Term j of (Flist), with radius rho_j in medium k_j, kappa_j = Lmax/a_j and g_j = sqrt(1 - kappa_j^2 sin^2), gives
        dPhi/dtheta*:  c_j kappa_j cos/g_j                   dPhi/dC_z:  c_j kappa_j' sin/g_j
        dZ/dtheta*:    c_j rho_j (C_z/nu) kappa_j^2 sin cos/g_j
        dZ/dC_z:      -c_j rho_j [(n^2/nu^3) g_j - (C_z/nu) kappa_j kappa_j' sin^2/g_j],   n, nu of medium k_j,
    with kappa_j' = (R_k*/rho_j) C_z (n_k*^2 - n^2)/(nu_k* nu^3).  Every factor but c_j and kappa_j' is nonnegative
    and monotone on a box, so each is bounded by its end values.  The term that grazes, kappa_j = 1, gives c_j, 0,
    c_j rho_j (C_z/nu) sin and -c_j rho_j (n^2/nu^3) cos.  Returns lo, hi of shape (boxes, 2, 2)."""
    n, Rin, Rout, med, rad, sig = st
    K = ca.size
    nua, nub = np.sqrt(n ** 2 - ca[:, None] ** 2)[:, med], np.sqrt(n ** 2 - cb[:, None] ** 2)[:, med]
    nsa, nsb = np.sqrt(n[ks] ** 2 - ca ** 2)[:, None], np.sqrt(n[ks] ** 2 - cb ** 2)[:, None]
    Rs, nk = Rin[ks][:, None], n[med]
    ma, mb = Rs * nsa / (nua * rad), Rs * nsb / (nub * rad)
    mlo, mhi = np.minimum(ma, mb), np.maximum(ma, mb)
    kc = Rs / rad * (n[ks][:, None] ** 2 - nk ** 2)           # kappa' = kc C_z/(nu_k* nu^3); C_z/(nu_k* nu^3) increases
    fa, fb = ca[:, None] / (nsa * nua ** 3), cb[:, None] / (nsb * nub ** 3)
    dlo, dhi = np.where(kc >= 0, kc * fa, kc * fb), np.where(kc >= 0, kc * fb, kc * fa)
    sa, sb = np.sin(ta)[:, None], np.sin(tb)[:, None]
    co_lo, co_hi = np.cos(tb)[:, None], np.cos(ta)[:, None]
    xlo, xhi = np.minimum(mlo * sa, 1.0), np.minimum(mhi * sb, 1.0)
    Dlo, Dhi = np.sqrt(1.0 - xhi ** 2), np.sqrt(1.0 - xlo ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iDlo, iDhi = 1.0 / Dhi, 1.0 / Dlo
        glo, ghi = ca[:, None] / nua, cb[:, None] / nub
        hlo, hhi = nk ** 2 / nua ** 3, nk ** 2 / nub ** 3
        pt = (mlo * co_lo * iDlo, mhi * co_hi * iDhi)
        pc = _imul(dlo, dhi, sa * iDlo, sb * iDhi)
        zt = (rad * glo * mlo ** 2 * sa * co_lo * iDlo, rad * ghi * mhi ** 2 * sb * co_hi * iDhi)
        q = _imul(dlo, dhi, glo * mlo * sa * sa * iDlo, ghi * mhi * sb * sb * iDhi)
        zc = (rad * (hlo * Dlo - q[1]), rad * (hhi * Dhi - q[0]))
    graze = (med[None, :] == ks[:, None]) & (np.arange(med.size)[None, :] % 2 == 0)
    pt = (np.where(graze, 1.0, pt[0]), np.where(graze, 1.0, pt[1]))
    pc = (np.where(graze, 0.0, pc[0]), np.where(graze, 0.0, pc[1]))
    zt = (np.where(graze, rad * glo * sa, zt[0]), np.where(graze, rad * ghi * sb, zt[1]))
    zc = (np.where(graze, rad * hlo * co_lo, zc[0]), np.where(graze, rad * hhi * co_hi, zc[1]))
    lo, hi = np.empty((K, 2, 2)), np.empty((K, 2, 2))
    for (r, c), (a, b), s in (((0, 0), pt, sig), ((0, 1), pc, sig), ((1, 0), zt, sig), ((1, 1), zc, -sig)):
        with np.errstate(invalid="ignore"):
            l_, h_ = np.where(s > 0, a, -b).sum(1), np.where(s > 0, b, -a).sum(1)
        grow = 64 * _EPS * (np.abs(l_) + np.abs(h_))
        lo[:, r, c] = np.where(np.isnan(l_), -np.inf, l_ - grow)
        hi[:, r, c] = np.where(np.isnan(h_), np.inf, h_ + grow)
    return lo, hi


def _far_point(st, th, C, ks):
    """Phi_far, Z_far and their Jacobian in (theta*, C_z) at points, with a bound on the rounding error of each value.

    The term that grazes is evaluated as theta* and cos(theta*), which keeps full accuracy as theta* -> pi/2.  The
    argument x_j = kappa_j sin(theta*) carries a relative error of a few eps, more where nu is small, and arcsin and
    sqrt magnify it by x_j/g_j."""
    n, Rin, Rout, med, rad, sig = st
    nu = np.sqrt(n ** 2 - C[:, None] ** 2)
    x = np.minimum(nu[np.arange(C.size), ks][:, None] * Rin[ks][:, None] * np.sin(th)[:, None] / (nu[:, med] * rad), 1.0)
    graze = (med[None, :] == ks[:, None]) & (np.arange(med.size)[None, :] % 2 == 0)
    asn = np.where(graze, th[:, None], np.arcsin(x))
    root = np.where(graze, np.cos(th)[:, None], np.sqrt(1.0 - x * x))
    zj = -sig * rad * (C[:, None] / nu[:, med]) * root
    phi = np.pi + (sig * asn).sum(1)
    z = zj.sum(1)
    amp = (n ** 2 + C[:, None] ** 2) / nu ** 2
    ex = 6.0 + amp[np.arange(C.size), ks][:, None] + amp[:, med]      # relative error of x_j, in units of eps
    with np.errstate(divide="ignore", invalid="ignore"):
        grow = np.nan_to_num(np.where(graze, 0.0, x / root), posinf=1e300)
    terms = 2 * med.size + 2
    ephi = _EPS * ((grow * ex).sum(1) + terms * (np.pi + np.abs(asn).sum(1)))
    ez = _EPS * ((np.abs(zj) * (grow ** 2 * ex + 6.0 + amp[:, med])).sum(1) + terms * np.abs(zj).sum(1))
    lo, hi = far_box_jacobian(st, th, th, C, C, ks)
    return phi, z, 0.5 * (lo + hi), ephi, ez


def _far_residual_mp(st, th, C, k, dphi, dz):
    """(Phi_far - dphi, Z_far - dz) at one point, in 34-digit arithmetic, for a box whose Krawczyk test is limited by
    the rounding of the double-precision value (near a caustic, where the inverse Jacobian is large)."""
    import mpmath as mpm
    n, Rin, Rout, med, rad, sig = st
    with mpm.workdps(34):
        th_, C_ = (v if isinstance(v, mpm.mpf) else mpm.mpf(float(v)) for v in (th, C))
        nu = [mpm.sqrt(mpm.mpf(float(v)) ** 2 - C_ ** 2) for v in n]
        Lm = nu[k] * mpm.mpf(float(Rin[k]))
        phi, z = mpm.pi, mpm.mpf(0)
        for j in range(med.size):
            nj, Rj = nu[med[j]], mpm.mpf(float(rad[j]))
            if med[j] == k and j % 2 == 0:
                a, r = th_, mpm.cos(th_)
            else:
                x = Lm * mpm.sin(th_) / (nj * Rj)
                a, r = mpm.asin(x), mpm.sqrt(1 - x * x)
            phi += float(sig[j]) * a
            z -= float(sig[j]) * Rj * (C_ / nj) * r
        return [phi - mpm.mpf(float(dphi)), z - mpm.mpf(float(dz))]


def classify_far(n, R, rP, rX, dphi, dz, tol=1e-13, stats=None, maxboxes=200000, box_trace=None):
    """Steps (3) and (4), far branch: every (Lambda, C_z) with Phi_far = dphi and Z_far = dz, polished.

    Boxes in (theta*, C_z), Lambda = Lmax(C_z) sin(theta*), cover [0, pi/2] x [0, C_max] between the cuts of (cuts); no
    steeper ray reaches the point (Property zbound).  A box is dropped when the bounds (farbox) exclude the target,
    or when Krawczyk's test (krawczyk) on the box grown by 10% shows it holds no ray; it yields a ray when the test
    shows it holds exactly one, and Newton polishes that ray inside it; otherwise it is halved.  The test carries a
    bound on the rounding error of the residual at the centre, and where that bound is not small against the box,
    as next to a caustic, the residual is recomputed to 34 digits, so no box is dropped on rounding.  A box still
    undecided at width tol either holds a ray on the edge of its piece (Lambda = 0 at dphi = pi, or a cut) or lies
    within rounding of a caustic; Newton is run from its centre and a ray it converges to is kept unless already found.
    stats, if a dict, receives the numbers of boxes examined, of residuals recomputed to 34 digits, of undecided boxes
    and of rays found from them.  If supplied, box_trace records each examined box as
    (theta_lo, theta_hi, C_lo, C_hi, outcome) for the paper's classification figure."""
    sgn, dz = (1.0 if dz >= 0 else -1.0), abs(dz)
    if dz == 0.0:
        return [(L, 0.0) for L, _, _ in far_roots(0.0, n, R, rP, rX, dphi)]
    st = _far_stack(n, R, rP, rX)
    nmin = st[0].min()
    cmax = _bisect(lambda C: Z0(C, n, R, rP, rX) - dz, 0.0, nmin * (1 - 1e-15), it=200)
    top = min(cmax * (1 + 1e-12) + 1e-15, nmin * (1 - 1e-15))
    cuts = [0.0] + switch_tilts(n, R, rX, top) + [top]
    ca, cb = np.array(cuts[:-1]), np.array(cuts[1:])
    ta, tb = np.zeros(ca.size), np.full(ca.size, 0.5 * np.pi)
    pa, pb = ca.copy(), cb.copy()                               # the piece between cuts that each box lies in
    ks = np.argmin(np.sqrt(st[0] ** 2 - (0.5 * (ca + cb))[:, None] ** 2) * st[1], 1)
    found, loose, examined, precise = [], [], 0, 0
    while ta.size:
        examined += ta.size
        if examined > maxboxes:                                 # never expected; keep every box rather than drop it
            loose += [(0.5 * (a + b), 0.5 * (c + d), p, q, k) for a, b, c, d, p, q, k in zip(ta, tb, ca, cb, pa, pb, ks)]
            break
        plo, phi, zlo, zhi = box_bounds(st, ta, tb, ca, cb, ks)
        keep = (plo <= dphi) & (dphi <= phi) & (zlo <= dz) & (dz <= zhi)
        if box_trace is not None:
            box_trace.extend((float(ta[k]), float(tb[k]), float(ca[k]), float(cb[k]), "bounds")
                             for k in np.flatnonzero(~keep))
        ta, tb, ca, cb, pa, pb, ks = (a[keep] for a in (ta, tb, ca, cb, pa, pb, ks))
        if not ta.size:
            break
        tm, cm, wt, wc = 0.5 * (ta + tb), 0.5 * (ca + cb), tb - ta, cb - ca
        Ta, Tb = np.maximum(ta - 0.1 * wt, 0.0), np.minimum(tb + 0.1 * wt, 0.5 * np.pi)
        Ca, Cb = np.maximum(ca - 0.1 * wc, pa), np.minimum(cb + 0.1 * wc, pb)
        ph, z, J, ephi, ez = _far_point(st, tm, cm, ks)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Y = np.stack([J[:, 1, 1], -J[:, 0, 1], -J[:, 1, 0], J[:, 0, 0]], 1).reshape(-1, 2, 2) / det[:, None, None]
            Y = np.where(np.isfinite(Y).all((1, 2))[:, None, None], Y, 0.0)
            v = -np.einsum("kij,kj->ki", Y, np.stack([ph - dphi, z - dz], 1))
            err = np.abs(Y[:, :, 0]) * ephi[:, None] + np.abs(Y[:, :, 1]) * ez[:, None]
            half = 0.5 * np.stack([Tb - Ta, Cb - Ca], 1)
            for k in np.flatnonzero((err > 0.125 * half).any(1) & (det != 0) & np.isfinite(det)):
                F_ = _far_residual_mp(st, tm[k], cm[k], ks[k], dphi, dz)
                v[k] = [-float(float(Y[k, i, 0]) * F_[0] + float(Y[k, i, 1]) * F_[1]) for i in range(2)]
                err[k] = 4 * _EPS * (np.abs(v[k]) + np.array([tm[k], cm[k]])) + 1e-30 * np.abs(Y[k]).sum(1)
                precise += 1
            Jlo, Jhi = far_box_jacobian(st, Ta, Tb, Ca, Cb, ks)
            Klo = np.stack([tm, cm], 1) + v - err
            Khi = np.stack([tm, cm], 1) + v + err
            lo_off, hi_off = np.stack([Ta - tm, Ca - cm], 1), np.stack([Tb - tm, Cb - cm], 1)
            for r in range(2):
                for c in range(2):
                    y0lo, y0hi = _imul(Y[:, r, 0], Y[:, r, 0], Jlo[:, 0, c], Jhi[:, 0, c])
                    y1lo, y1hi = _imul(Y[:, r, 1], Y[:, r, 1], Jlo[:, 1, c], Jhi[:, 1, c])
                    slack = 8 * _EPS * (np.abs(y0lo) + np.abs(y0hi) + np.abs(y1lo) + np.abs(y1hi))
                    klo, khi = _imul(float(r == c) - (y0hi + y1hi) - slack, float(r == c) - (y0lo + y1lo) + slack,
                                     lo_off[:, c], hi_off[:, c])
                    Klo[:, r] += klo
                    Khi[:, r] += khi
            Blo, Bhi = np.stack([Ta, Ca], 1), np.stack([Tb, Cb], 1)
            fine = np.isfinite(Klo).all(1) & np.isfinite(Khi).all(1) & (det != 0)
            one = fine & ((Klo > Blo) & (Khi < Bhi)).all(1)
            none = fine & ((Khi < Blo) | (Klo > Bhi)).any(1)
        for k in np.flatnonzero(one):
            found.append((tm[k], cm[k], Ta[k], Tb[k], Ca[k], Cb[k], ks[k], Y[k]))
        open_ = ~one & ~none
        small = open_ & (wt < tol) & (wc < tol)
        if box_trace is not None:
            outcomes = np.where(one, "ray", np.where(none, "krawczyk", np.where(small, "limit", "split")))
            box_trace.extend((float(a), float(b), float(c), float(d), str(outcome))
                             for a, b, c, d, outcome in zip(ta, tb, ca, cb, outcomes))
        loose += [(tm[k], cm[k], pa[k], pb[k], ks[k]) for k in np.flatnonzero(small)]
        split = open_ & ~small
        ta, tb, ca, cb, pa, pb, ks, tm, cm, wt, wc, J = (a[split] for a in (ta, tb, ca, cb, pa, pb, ks, tm, cm, wt, wc, J))
        zs = max(dz, 1e-9)
        with np.errstate(invalid="ignore", over="ignore"):
            st_ = wt * (np.abs(J[:, 0, 0]) + np.abs(J[:, 1, 0]) / zs)
            sc_ = wc * (np.abs(J[:, 0, 1]) + np.abs(J[:, 1, 1]) / zs)
        st_, sc_ = np.where(np.isfinite(st_), st_, wt), np.where(np.isfinite(sc_), sc_, wc)
        by_t = ((st_ >= sc_) & (wt >= tol)) | (wc < tol)
        ta, tb, ca, cb = (np.concatenate([ta, np.where(by_t, tm, ta)]), np.concatenate([np.where(by_t, tm, tb), tb]),
                          np.concatenate([ca, np.where(by_t, ca, cm)]), np.concatenate([np.where(by_t, cb, cm), cb]))
        pa, pb, ks = np.concatenate([pa, pa]), np.concatenate([pb, pb]), np.concatenate([ks, ks])

    def newton(x, lo, hi, k, Y=None):
        """Newton on (Phi_far - dphi, Z_far - dz) in (theta*, C_z) inside [lo, hi]; the fixed Y when a step leaves it.
        Stops once the residual is within the rounding error of the values."""
        for _ in range(60):
            p_, z_, J_, e1, e2 = _far_point(st, x[:1], x[1:], np.array([k]))
            F_ = np.array([p_[0] - dphi, z_[0] - dz])
            if abs(F_[0]) <= e1[0] and abs(F_[1]) <= e2[0]:
                break
            try:
                y = x - np.linalg.solve(J_[0], F_)
            except np.linalg.LinAlgError:
                y = x
            if not ((lo <= y) & (y <= hi)).all():
                y = x - Y @ F_ if Y is not None else np.clip(y, lo, hi)
            if np.all(np.abs(y - x) <= 4 * _EPS * np.maximum(1.0, np.abs(x))):
                break
            x = y
        return x, np.abs(F_), (e1[0], e2[0])

    rays = []                                                   # (theta and C_z, k*, box)
    for tm_, cm_, Ta_, Tb_, Ca_, Cb_, k_, Y_ in found:
        lo, hi = np.array([Ta_, Ca_]), np.array([Tb_, Cb_])
        x, _, _ = newton(np.array([tm_, cm_]), lo, hi, k_, Y_)
        # each of two boxes holds exactly one ray, so rays lying in each other's boxes are the same ray
        if not any(((r[2] <= x) & (x <= r[3])).all() and ((lo <= r[0]) & (r[0] <= hi)).all() for r in rays):
            rays.append((x, k_, lo, hi))
    extra = 0
    for tm_, cm_, pa_, pb_, k_ in loose:
        x, res, (e1, e2) = newton(np.array([tm_, cm_]), np.array([0.0, pa_]), np.array([0.5 * np.pi, pb_]), k_)
        if (res[0] <= max(64 * e1, 1e-13) and res[1] <= max(64 * e2, 1e-13 * max(dz, 1e-3))
                and not any(np.all(np.abs(x - r[0]) <= 1e-12 + 1e-9 * np.abs(r[0])) for r in rays)):
            rays.append((x, k_, x, x))
            extra += 1
    if stats is not None:
        stats.update(boxes=examined, precise=precise, undecided=len(loose), from_undecided=extra)
    nu = lambda C, k: np.sqrt(st[0][k] ** 2 - C * C)
    return [(float(nu(x[1], k_) * st[1][k_] * np.sin(x[0])), sgn * float(x[1])) for x, k_, _, _ in rays]


def solve_far(n, R, rP, rX, dphi, dz, Lam0, Cz0, itmax=40, tol=1e-10):
    """Step (4), far branch: Newton on (Phi_far - dphi, Z_far - dz) with the Jacobian of Property symmetry, from a
    step-(3) seed, backtracking on the residual and staying inside D (this branch has no convex program)."""
    def residual(L, C):
        pz = Phi_Z(L, C, n, R, rP, rX, far=True)
        return None if pz is None else np.array([pz[0] - dphi, pz[1] - dz])

    L, C = Lam0, Cz0
    g = residual(L, C)
    for it in range(itmax):
        if g is None:
            return None, it, np.inf
        if np.linalg.norm(g) < tol:
            return (L, C), it, np.linalg.norm(g)
        j12 = dPhi_dCz(L, C, n, R, rP, rX, far=True)
        j22 = dZ_dCz(L, C, n, R, rP, rX, far=True)
        if j22 is None:
            break
        try:
            p = -np.linalg.solve(np.array([[dPhi_dLam(L, C, n, R, rP, rX, far=True), j12], [j12, j22]]), g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-12:
            Ln, Cn = L + t * p[0], C + t * p[1]
            gn = residual(Ln, Cn) if in_domain(Ln, Cn, n, R, rX) else None
            if gn is not None and np.linalg.norm(gn) < np.linalg.norm(g):
                L, C, g = Ln, Cn, gn
                break
            t *= 0.5
        else:
            break
    return (L, C), itmax, (np.inf if g is None else np.linalg.norm(g))


def project(n, R, P0, X, tol=1e-9):
    """Steps (1)-(5): every ray from the pinhole P0 to the point X, as (branch, Lambda, C_z, direction at P0).

    The sign of Lambda is the sense of rotation about the axis.  Beside Delta phi the targets
    2 pi - Delta phi, Delta phi + 2 pi, ... are searched while they stay below the bound
    sum_{k<N} arccos(R_k/R_{k-1}) + pi on any sweep (Property chord); test (wraptest) of the notes would
    let them be skipped when |Delta z| < Z0(C_pi), and the box search of the far branch discards them quickly.

    A point opposite the pinhole, Delta phi = pi, is reached by the radial ray of the far branch, Lambda = 0, which
    sweeps pi at every tilt.  Its root lies on the edge Lambda = 0 of the far search for both senses, where the
    rounding of pi alone can put it just outside for both or just inside for both, so it is added here, once.  The
    targets pi and 2 pi - pi coincide there, so each is searched once and every other ray is returned in both senses.

    Every length is divided on entry by u = 2^floor(log2 r_P), the power of two in (r_P/2, r_P], and Lambda is
    multiplied by u on exit.  The tolerances, tol here and those of the solvers, act on (Delta phi, Delta z), so they
    then act on Delta z / u, and a rig gives the same rays in metres and in millimetres.  In millimetres a tolerance
    on Delta z itself would be 1000 times stricter than in metres, beyond what the closed forms resolve next to
    grazing.  Dividing by a power of two is exact, so the problem solved is the one given to the last bit; dividing
    by r_P itself would round every length, and next to grazing that rounding alone costs about eps/sqrt(delta) in
    Phi (audit A1, code/tests/openlpt_audit/)."""
    P0, X = np.asarray(P0, float), np.asarray(X, float)
    unit = math.ldexp(1.0, math.frexp(np.hypot(P0[0], P0[1]))[1] - 1)     # u = 2^floor(log2 r_P), exact scaling
    R, P0, X = [float(r) / unit for r in R], P0 / unit, X / unit
    rP, rX = np.hypot(P0[0], P0[1]), np.hypot(X[0], X[1])
    dz = X[2] - P0[2]
    # Step (1): the signed azimuth, reduced by the IEEE remainder as in the C++ port; (d + pi) % 2 pi - pi rounds twice
    # and flushes |d| below about 4e-16 to 0.
    d = math.remainder(float(np.arctan2(X[1], X[0]) - np.arctan2(P0[1], P0[0])), 2 * math.pi)
    if rX < 1e-15:                                   # step (1): the radial ray
        C = np.sign(dz) * _bisect(lambda c: Z0(c, n, R, rP, rX) - abs(dz), 0.0, min(n) * (1 - 1e-15), it=200)
        # Its branch is the limit r_X -> 0 of the general case: far when X lies beyond the axis as seen from the
        # pinhole.  Exactly on the axis the azimuth is undefined and either label holds.
        beyond = (X[0] != 0.0 or X[1] != 0.0) and abs(d) > 0.5 * np.pi
        return [("far" if beyond else "near", 0.0, C, direction_from_invariants(P0, 0.0, C, n[0], nus(n, C)[0]))]
    turn, dphi = (1.0 if d >= 0 else -1.0), abs(d)
    opposite = abs(dphi - np.pi) <= 8 * _EPS                 # within 4 ulp of pi, the rounding of the azimuths
    top = sum(np.arccos(ri / ro) for ro, ri in zip([rP] + list(R[:-1]), R)) + np.pi
    rays, m = [], 0
    if opposite:                                     # step (1): the far branch's radial ray, Z_far(0, C_z) increasing
        dphi = np.pi
        C = np.sign(dz) * _bisect(lambda c: Phi_Z(0.0, c, n, R, rP, rX, far=True)[1] - abs(dz), 0.0,
                                  min(n) * (1 - 1e-15), it=200)
        rays.append(("far", 0.0, C))
    senses = (1.0, -1.0) if opposite else None
    while dphi + 2 * np.pi * m < top:
        for target, sense in ((dphi + 2 * np.pi * m, 1.0), (2 * np.pi * (m + 1) - dphi, -1.0)):
            if target >= top or (opposite and sense < 0):
                continue
            seed = classify(n, R, rP, rX, target, dz)
            if seed is not None:
                sol, _, res = solve(n, R, rP, rX, target, dz, seed[0], seed[1])
                if sol is not None and res < tol:
                    rays += [("near", turn * s * sol[0], sol[1]) for s in (senses or (sense,))]
            for s0 in classify_far(n, R, rP, rX, target, dz):
                sol, _, res = solve_far(n, R, rP, rX, target, dz, s0[0], s0[1])
                if sol is not None and res < tol:
                    if opposite and abs(sol[0]) <= 1e-9 * Lmax_kstar(n, R, rX, sol[1])[0]:
                        continue                     # the radial ray, already added
                    rays += [("far", turn * s * sol[0], sol[1]) for s in (senses or (sense,))]
        m += 1
    return [(b, L * unit, C, direction_from_invariants(P0, L, C, n[0], nus(n, C)[0])) for b, L, C in rays]
