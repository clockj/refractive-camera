"""Where does a point appear?  Counts and concrete cases for three camera models through the reference tube.

For points inside the water-filled tube, the truth is every image found by a brute-force search that uses tracing
alone (`tracing.search_images`), none of the paper's equations.  Three models answer the same question:
  * `project()` (ours): every image;
  * AFRT (Belden 2013): one image from one start, or none when the iteration fails;
  * a pinhole camera with OpenCV distortion (the camera model of OpenCV, COLMAP and ROS), calibrated through the
    tube on points filling the water (its best case): always exactly one image.
An answer is right when it has as many images as the truth, each within 1 px of a true one; a hidden point's right
answer is "no image".

    python benchmarks/cases.py          # about 5 minutes; writes cases_out.txt, cases.json and cases.png
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import refractive_camera as rc  # noqa: E402
from refractive_camera.baselines import afrt  # noqa: E402
from refractive_camera.tracing import search_images  # noqa: E402

TUBE = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))
CAM = rc.Camera.look_at((0.35, 0.0, 0.0), (0.0, 0.0, 0.0), f=2000.0, size=(1280, 1024))
TOL_PX = 1.0
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def fit_pinhole(rng, r_max=0.090, z_max=0.050, noise_px=0.1):
    """The pinhole + distortion camera, fitted through the tube to points filling the water (single images)."""
    from scipy.optimize import least_squares
    n = 6000
    r = r_max * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(-np.pi, np.pi, n)
    P = np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-z_max, z_max, n)]
    im = rc.project(P, CAM, TUBE)
    one = np.flatnonzero(im.counts() == 1)
    rows = np.array([im.rows_of(i)[0] for i in one])
    keep = im.in_image[rows]
    P, uv = P[one[keep]], im.uv[rows[keep]] + rng.normal(0, noise_px, (keep.sum(), 2))

    def build(p):
        return rc.Camera([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], rotvec(p[9:12]) @ CAM.R, p[12:15], p[4:9],
                         CAM.size)

    p0 = np.r_[CAM.K[0, 0], CAM.K[1, 1], CAM.K[0, 2], CAM.K[1, 2], np.zeros(5), np.zeros(3), CAM.t]
    fit = least_squares(lambda p: np.nan_to_num((build(p).project_points(P)[0] - uv).ravel(), nan=1e3), p0,
                        method="lm", x_scale="jac", max_nfev=5000)
    model = build(fit.x)
    rms = float(np.sqrt(np.mean(np.sum((model.project_points(P)[0] - uv) ** 2, 1))))
    return model, rms, len(P)


def point_sets(rng):
    def water(m, r_lo=0.0, r_hi=0.0935, z=0.05):
        r = np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2, m))
        a = rng.uniform(-np.pi, np.pi, m)
        return np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-z, z, m)]

    pool = water(40000, 0.060)
    c = rc.project(pool, CAM, TUBE).counts()
    r = np.hypot(pool[:, 0], pool[:, 1])
    inner = pool * ((r - 1e-4) / r)[:, None]
    inner[:, 2] = pool[:, 2]
    c_in = rc.project(inner, CAM, TUBE).counts()
    return {"ordinary": water(100), "seen twice": pool[c == 2][:50], "hidden": pool[(c == 0)][:50],
            "edge": pool[(c != c_in) & (c > 0)][:50], "near the wall": water(100, 0.085)}


def truth(x):
    """Every ray that reaches the point, as the benchmark defines it: the candidates of the brute-force search, of
    project() and of AFRT, each kept only if tracing it passes within 1e-9 r_P of the point, duplicates merged.  A
    sampled search alone can miss an image where two images merge or a ray grazes the wall."""
    cand = [search_images(TUBE, CAM.center, x, grid=(700, 400))]
    im = rc.project(x, CAM, TUBE)
    cand.append(im.direction[np.isfinite(im.direction).all(1)])
    d, _, _ = afrt(TUBE, CAM.center, x)
    if d is not None:
        cand.append(np.asarray(d)[None])
    D = np.vstack([c for c in cand if len(c)]) if any(len(c) for c in cand) else np.zeros((0, 3))
    if len(D):
        tr = rc.trace(TUBE, np.broadcast_to(CAM.center, D.shape), D)
        ok = tr.ok & (tr.miss_distance(np.broadcast_to(x, D.shape)) < 1e-9 * np.hypot(*CAM.center[:2]))
        D = D[ok]
        keep = []
        for k, dk in enumerate(D):
            if all(np.linalg.norm(dk - D[j]) > 1e-7 for j in keep):
                keep.append(k)
        D = D[keep]
    if not len(D):
        return np.zeros((0, 2))
    uv, front = CAM.pixels_from_directions(D @ CAM.R.T)
    return uv[front]


def ours(x):
    im = rc.project(x, CAM, TUBE)
    return im.uv[np.isfinite(im.uv).all(1)]


def afrt_one(x):
    d, status, _ = afrt(TUBE, CAM.center, x)
    if d is None:
        return np.zeros((0, 2))
    tr = rc.trace(TUBE, CAM.center, d)
    if not tr.ok[0] or tr.miss_distance(x[None])[0] > 1e-9:
        return np.zeros((0, 2))                     # it returned a ray that does not reach the point: no answer
    uv, front = CAM.pixels_from_directions((d @ CAM.R.T)[None])
    return uv[front]


def outcome(pred, true):
    if len(true) == 0:
        return "right" if len(pred) == 0 else "sees a hidden point"
    if len(pred) == 0:
        return "says hidden"
    near_true = [np.min(np.linalg.norm(true - p, axis=1)) <= TOL_PX for p in pred]
    found = [np.min(np.linalg.norm(pred - t, axis=1)) <= TOL_PX for t in true]
    if all(near_true) and all(found) and len(pred) == len(true):
        return "right"
    if not all(near_true):
        return "wrong place"
    return "misses an image"


def main():
    rng = np.random.default_rng(2026)
    pin, rms, n = fit_pinhole(rng)
    say(f"Pinhole + distortion fitted through the tube on {n} points filling the water: {rms:.2f} px rms, "
        f"fx = {pin.K[0, 0]:.0f}, fy = {pin.K[1, 1]:.0f}, k1 = {pin.dist[0]:.3f} (true camera: f = 2000, no distortion)")
    sets = point_sets(rng)
    methods = {"refractive (ours)": ours, "AFRT, one start": afrt_one,
               "pinhole + distortion": lambda x: pin.project_points(x)[0].reshape(-1, 2)}
    table, cases = {}, {}
    for name, X in sets.items():
        table[name] = {m: {} for m in methods}
        err_pin = []
        for x in X:
            t = truth(x)
            for m, f in methods.items():
                p = f(x)
                o = outcome(p, t)
                table[name][m][o] = table[name][m].get(o, 0) + 1
                cases.setdefault((name, m, o), []).append((x.tolist(), t.tolist(), p.tolist()))
            if len(t):
                err_pin.append(np.min(np.linalg.norm(t - pin.project_points(x)[0], axis=1)))
        table[name]["pinhole error px (median, to the nearest true image)"] = float(np.median(err_pin)) if err_pin \
            else None
        say(f"\n{name} ({len(X)} points):")
        for m in methods:
            right = table[name][m].get("right", 0)
            say(f"  {m:22s} right at {right:3d} of {len(X)} ({100 * right / len(X):5.1f} %)   "
                + ", ".join(f"{k}: {v}" for k, v in sorted(table[name][m].items()) if k != "right"))
        if err_pin:
            say(f"  pinhole + distortion: median distance to the nearest true image {np.median(err_pin):.1f} px")
    (HERE / "cases.json").write_text(json.dumps({"pinhole_fit_rms_px": rms, "table": table,
                                                 "cases": {" | ".join(k): v for k, v in cases.items()}}, indent=1))
    (HERE / "cases_out.txt").write_text("\n".join(OUT) + "\n")
    np.savez(HERE / "_cases_pinhole.npz", K=pin.K, R=pin.R, t=pin.t, dist=pin.dist)
    figure(pin, cases)


def figure(pin, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def n_true(ex):
        return len(np.array(ex[1]).reshape(-1, 2))

    def separation(ex):
        t_ = np.array(ex[1]).reshape(-1, 2)
        return np.linalg.norm(t_[0] - t_[1]) if len(t_) == 2 else np.inf

    def pin_err(ex):
        t_, p_ = np.array(ex[1]).reshape(-1, 2), np.array(ex[2]).reshape(-1, 2)
        return np.min(np.linalg.norm(t_ - p_[0], axis=1)) if len(t_) and len(p_) else np.inf

    picks = [  # (set, title, method whose failure is shown, its outcomes, how to choose among the examples)
        ("seen twice", "a point seen twice", "AFRT, one start", ("misses an image",),
         lambda exs: min(exs, key=lambda e: abs(separation(e) - 110))),
        ("hidden", "a point hidden by the wall", "pinhole + distortion", ("sees a hidden point",),
         lambda exs: exs[0]),
        ("ordinary", "an ordinary point", "pinhole + distortion", ("wrong place",),
         lambda exs: sorted([e for e in exs if n_true(e) == 1], key=pin_err)[len([e for e in exs if n_true(e) == 1]) // 2]),
        ("near the wall", "near the wall, seen once", "pinhole + distortion", ("wrong place",),
         lambda exs: sorted([e for e in exs if n_true(e) == 1], key=pin_err)[len([e for e in exs if n_true(e) == 1]) // 2]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, (set_name, title, m_key, o_keys, choose) in zip(axes, picks):
        key = next(((set_name, m_key, o) for o in o_keys if (set_name, m_key, o) in cases), None)
        if key is None:
            ax.set_axis_off()
            continue
        x, t, _ = choose(cases[key])
        x, t = np.array(x), np.array(t).reshape(-1, 2)
        p_ours, p_afrt, p_pin = ours(x), afrt_one(x), pin.project_points(x)[0].reshape(-1, 2)
        pts = np.vstack([q for q in (t, p_ours, p_afrt, p_pin) if len(q)])
        lo, hi = pts.min(0), pts.max(0)
        side = int(max(hi[0] - lo[0], hi[1] - lo[1], 40) + 50)
        c = np.round(0.5 * (lo + hi)).astype(int)
        region = (c[0] - side // 2, c[1] - side // 2, c[0] - side // 2 + side, c[1] - side // 2 + side)
        img = rc.render_dots(CAM, TUBE, [x], 0.001, supersample=4, region=region, outside=0.55)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                  extent=(region[0] - 0.5, region[2] - 0.5, region[3] - 0.5, region[1] - 0.5))
        for q in p_ours:
            ax.plot(*q, "o", mfc="none", mec="#d55e00", ms=18, mew=2, label="refractive (ours)")
        for q in p_afrt:
            ax.plot(*q, "x", color="#009e73", ms=13, mew=2.2, label="AFRT")
        for q in p_pin:
            ax.plot(*q, "s", mfc="none", mec="#0072b2", ms=14, mew=2, label="pinhole + distortion")
        h, lab = ax.get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="lower left", framealpha=0.85)
        off = np.min(np.linalg.norm(t - p_pin[0], axis=1)) if len(t) and len(p_pin) else None
        ax.set_title(f"{title}\ntrue images: {len(t)}; ours {len(p_ours)}, AFRT {len(p_afrt)}, pinhole {len(p_pin)}"
                     + (f" ({off:.0f} px off)" if off is not None else ""), fontsize=9)
        ax.set_xlim(region[0] - 0.5, region[2] - 0.5)
        ax.set_ylim(region[3] - 0.5, region[1] - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Where does a 1 mm dot in the water-filled tube appear?  Dark blobs: the dot, ray-traced; "
                 "markers: each model's answer", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "cases.png", dpi=105)
    print("wrote", HERE / "cases.png")


if __name__ == "__main__":
    if "--figure-only" in sys.argv:                          # redraw from cases.json
        z = json.loads((HERE / "cases.json").read_text())
        pz = np.load(HERE / "_cases_pinhole.npz")
        figure(rc.Camera(pz["K"], pz["R"], pz["t"], pz["dist"], CAM.size),
               {tuple(k.split(" | ")): v for k, v in z["cases"].items()})
    else:
        main()
