"""Demo A: locating a target through a water-filled tube.

A 1 mm dot moves on a helix inside the paper's reference port (an acrylic tube 200 mm across with 6 mm walls,
filled with water) while three cameras watch from outside.  Their frames are rendered by tracing rays through the
tube (refractive_camera.render_dots: vector Snell's law, none of the projection's closed forms), and the dot is
detected in them as dark blobs.  Then:

* Where does the dot appear?  refractive_camera.project() predicts every image, including the frames in which a
  camera sees the dot twice or not at all.  A published iterative method (AFRT, Belden 2013) is shown beside it.
* Where is the dot?  It is triangulated with the refractive model (back-projection through the tube, then
  reprojection refinement), and as a robotics user would without one: pinhole cameras with lens distortion,
  calibrated on a dot grid inside the tube (r <= 60 mm, |z| <= 40 mm), triangulating straight rays.

    python examples/demo_a_localization/demo_a.py [--frames 150] [--workers 8]

Needs numpy, scipy, matplotlib and Pillow.  Writes, beside this script: demo_a_out.txt (every number printed),
results.json, summary.png and animation.gif.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import refractive_camera as rc  # noqa: E402
from refractive_camera.baselines import afrt  # noqa: E402

# ------------------------------------------------------------------------------------------------------- the scene
TUBE = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.49, 1.33))     # metres
DOT_R = 0.001
BG, DOT = 0.85, 0.05                       # grey levels of the background and the dot
LENS = (-0.08, 0.02, 0.0, 0.0, 0.0)        # a mild barrel distortion, the cameras' own lens
CAL_R, CAL_Z = 0.060, 0.040                # the baseline's calibration volume
MATCH_PX = 5.0                             # a predicted image and a detected blob within this are the same image
REACHES = 1e-6                             # m: a ray passing this close to the dot's centre reaches it


def cameras():
    """Three cameras around the tube, 75 degrees apart, at slightly different heights, looking at its axis."""
    out = []
    for az, dist, h in ((0.0, 0.40, 0.0), (75.0, 0.42, 0.06), (150.0, 0.40, -0.05)):
        a = np.radians(az)
        out.append(rc.Camera.look_at((dist * np.cos(a), dist * np.sin(a), h), (0.0, 0.0, 0.0), f=2200.0,
                                     size=(1280, 1024), dist=LENS))
    return out


def trajectory(frames):
    """Out from near the axis to 87 mm from it in a spiral over a third of the frames, then once round the tube at
    87 mm (the dot's surface 6 mm from the inner wall), rising 100 mm in all.  On the way round, every camera's view
    crosses its shadow and its two-image bands."""
    k = int(round(frames / 3))
    t = np.arange(k) / k
    r1, th1 = 0.005 + 0.082 * t, 0.3 + 2 * np.pi * 1.5 * t
    th2 = th1[-1] + 2 * np.pi * np.arange(1, frames - k + 1) / (frames - k)
    r = np.r_[r1, np.full(frames - k, 0.087)]
    th = np.r_[th1, th2]
    return np.c_[r * np.cos(th), r * np.sin(th), np.linspace(-0.05, 0.05, frames)]


def half_camera(cam):
    """The same camera at half the resolution (pixel centres at integers: u_half = (u + 0.5)/2 - 0.5)."""
    K = cam.K.copy()
    K[0, :2] /= 2
    K[1, 1] /= 2
    K[0, 2], K[1, 2] = (K[0, 2] + 0.5) / 2 - 0.5, (K[1, 2] + 0.5) / 2 - 0.5
    return rc.Camera(K, cam.R, cam.t, cam.dist, (cam.size[0] // 2, cam.size[1] // 2))


# ------------------------------------------------------------------------------------- rendering and detection
def observe(job):
    """Render one camera's frame and detect the dot: blob centroids (full-resolution pixels) and areas, and the
    half-resolution frame as 8 bits for the figures.  Rendering uses tracing only."""
    from scipy import ndimage
    cam, X = job
    half = half_camera(cam)
    coarse = rc.render_dots(half, TUBE, [X], DOT_R, supersample=2, outside=BG)
    W, H = cam.size
    blobs = []
    lab, _ = ndimage.label(coarse < BG - 0.1 * (BG - DOT))
    for sl in ndimage.find_objects(lab):
        u0, u1 = max(2 * sl[1].start - 8, 0), min(2 * sl[1].stop + 8, W)
        v0, v1 = max(2 * sl[0].start - 8, 0), min(2 * sl[0].stop + 8, H)
        fine = rc.render_dots(cam, TUBE, [X], DOT_R, supersample=6, region=(u0, v0, u1, v1), outside=BG)
        lab2, _ = ndimage.label(fine < BG - 0.02 * (BG - DOT))
        for sl2 in ndimage.find_objects(lab2):
            mask = np.zeros_like(fine, bool)
            mask[sl2] = True
            w = np.where(mask, np.clip(BG - fine, 0, None), 0.0)
            vv, uu = np.mgrid[v0:v1, u0:u1]
            c = np.array([(w * uu).sum() / w.sum(), (w * vv).sum() / w.sum()])
            if not any(np.hypot(*(c - b[0])) < 1.0 for b in blobs):
                blobs.append((c, float(w.sum() / (BG - DOT))))
    img8 = np.clip(coarse * 255, 0, 255).astype(np.uint8)
    return [(b[0].tolist(), b[1]) for b in blobs], img8


# ------------------------------------------------------------------------------- the refraction-unaware baseline
def rotvec_matrix(w):
    th = np.linalg.norm(w)
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def calibrate_pinhole(cam, rng, noise_px=0.1):
    """Fit fx, fy, cx, cy, (k1, k2, p1, p2, k3), R and t of a plain pinhole camera to a dot grid inside the tube,
    observed through it (single-image points, 0.1 px noise), starting from the true camera."""
    from scipy.optimize import least_squares
    g = np.arange(-CAL_R, CAL_R + 1e-9, 0.010)
    P = np.array([(x, y, z) for x in g for y in g for z in np.arange(-CAL_Z, CAL_Z + 1e-9, 0.010)
                  if np.hypot(x, y) <= CAL_R])
    im = rc.project(P, cam, TUBE)
    one = np.flatnonzero(im.counts() == 1)
    rows = np.array([im.rows_of(i)[im.in_image[im.rows_of(i)]][0] for i in one])
    P, uv = P[one], im.uv[rows] + rng.normal(scale=noise_px, size=(len(one), 2))

    def build(p):
        K = [[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]]
        R = rotvec_matrix(p[9:12]) @ cam.R
        return rc.Camera(K, R, p[12:15], p[4:9], cam.size)

    p0 = np.r_[cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2], cam.dist, np.zeros(3), cam.t]
    fit = least_squares(lambda p: (build(p).project_points(P)[0] - uv).ravel(), p0, method="lm", x_scale="jac")
    model = build(fit.x)
    res = model.project_points(P)[0] - uv
    return model, dict(points=len(P), rms_px=float(np.sqrt(np.mean(np.sum(res ** 2, 1)))),
                       fx=float(fit.x[0]), fy=float(fit.x[1]), k1=float(fit.x[4]),
                       distance=float(np.hypot(*model.center[:2])), true_distance=float(np.hypot(*cam.center[:2])))


def triangulate_straight(cams, uv):
    """The least-squares point of the straight rays of pinhole cameras."""
    S, b = np.zeros((3, 3)), np.zeros(3)
    for cam, x in zip(cams, uv):
        o, d = cam.rays(np.asarray(x)[None])
        A = np.eye(3) - np.outer(d[0], d[0])
        S += A
        b += A @ o[0]
    return np.linalg.solve(S, b)


# ------------------------------------------------------------------------------------------------- the analysis
def match(pred, blobs):
    """Greedy one-to-one matching of predicted pixels to blob centroids within MATCH_PX: (matched, missed, extra)
    and the distances of the matched pairs."""
    pred, blobs = [np.asarray(p) for p in pred], [np.asarray(b) for b in blobs]
    pairs = sorted((np.hypot(*(p - b)), i, j) for i, p in enumerate(pred) for j, b in enumerate(blobs))
    used_p, used_b, dist = set(), set(), []
    for d, i, j in pairs:
        if d <= MATCH_PX and i not in used_p and j not in used_b:
            used_p.add(i)
            used_b.add(j)
            dist.append(d)
    return np.array([len(used_p), len(pred) - len(used_p), len(blobs) - len(used_b)]), dist


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--redraw", action="store_true", help="redraw summary.png from results.json, rendering nothing")
    args = ap.parse_args()
    if args.redraw:
        redraw()
        return
    lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    t_start = time.perf_counter()
    cams, X = cameras(), trajectory(args.frames)
    say(f"Demo A: {args.frames} frames, 3 cameras, backends {rc.available_backends()}, refractive_camera "
        f"{rc.__version__}, Python {sys.version.split()[0]}")
    say(f"tube {TUBE.radii} m, indices {TUBE.indices}; dot radius {DOT_R * 1000:g} mm; lens {LENS}")

    # 1. Render and detect (tracing only)
    t0 = time.perf_counter()
    jobs = [(cams[c], X[f]) for f in range(len(X)) for c in range(3)]
    with ProcessPoolExecutor(args.workers) as ex:
        obs = list(ex.map(observe, jobs, chunksize=4))
    blobs = [[obs[3 * f + c][0] for c in range(3)] for f in range(len(X))]
    frames8 = [[obs[3 * f + c][1] for c in range(3)] for f in range(len(X))]
    say(f"rendered and detected {len(jobs)} frames in {time.perf_counter() - t0:.0f} s ({args.workers} processes)")

    # 2. Where the dot appears: ours and AFRT against the detected blobs
    ours_uv, afrt_uv = [], []
    t_ours = t_afrt = 0.0
    n_proj = 0
    tally = {k: np.zeros(3, int) for k in ("ours", "afrt")}
    offsets = {k: [] for k in ("ours", "afrt")}
    afrt_status = {"converged-true": 0, "converged-false": 0, "cap": 0, "breakdown": 0}
    counts = np.zeros((3, 4), int)                       # per camera: frames with 0, 1, 2, 3+ images (ours)
    for f in range(len(X)):
        row_o, row_a = [], []
        for c, cam in enumerate(cams):
            t1 = time.perf_counter()
            im = rc.project(X[f], cam, TUBE)
            t_ours += time.perf_counter() - t1
            p_o = [im.uv[k] for k in np.flatnonzero(im.in_image)]
            counts[c, min(len(p_o), 3)] += 1
            t1 = time.perf_counter()
            d, status, _ = afrt(TUBE, cam.center, X[f])
            t_afrt += time.perf_counter() - t1
            n_proj += 1
            p_a = []
            if d is not None and status == "converged":
                miss = rc.trace(TUBE, cam.center, d).miss_distance(X[f][None])[0]
                good = miss < REACHES
                afrt_status["converged-true" if good else "converged-false"] += 1
                uv, front = cam.pixels_from_directions(cam.R @ d)
                if front and cam.in_image(uv):
                    p_a = [uv]
            else:
                afrt_status[status] += 1
            b = [x[0] for x in blobs[f][c]]
            for k, p in (("ours", p_o), ("afrt", p_a)):
                m, dist = match(p, b)
                tally[k] += m
                offsets[k] += dist
            row_o.append([p.tolist() for p in p_o])
            row_a.append([p.tolist() for p in p_a])
        ours_uv.append(row_o)
        afrt_uv.append(row_a)
    n_blobs = sum(len(b) for fr in blobs for b in fr)
    say(f"\nwhere the dot appears ({n_proj} camera-frames, {n_blobs} blobs detected):")
    for c in range(3):
        say(f"  camera {c}: frames with 0/1/2/3+ images predicted: {counts[c].tolist()}")
    for k, name in (("ours", "project()"), ("afrt", "AFRT, one start")):
        m, miss, extra = tally[k]
        o = np.array(offsets[k])
        say(f"  {name:16s}: {m} images matched to blobs, {miss} predicted but not seen, {extra} blobs not predicted;"
            f" {1e3 * (t_ours if k == 'ours' else t_afrt) / n_proj:.2f} ms per projection")
        if o.size:
            say(f"  {'':16s}  blob centroid to predicted image: median {np.median(o):.3f} px, p95 "
                f"{np.percentile(o, 95):.3f} px, max {o.max():.3f} px")
    say(f"  AFRT outcomes: {afrt_status}")

    # 3. Calibrate the baseline (pinhole + distortion) on the dot grid
    rng = np.random.default_rng(7)
    base, cal = [], []
    for cam in cams:
        model, info = calibrate_pinhole(cam, rng)
        base.append(model)
        cal.append(info)
    say("\nbaseline calibration (pinhole + distortion, refraction ignored), on the dot grid "
        f"r <= {CAL_R * 1000:g} mm, |z| <= {CAL_Z * 1000:g} mm:")
    for c, info in enumerate(cal):
        say(f"  camera {c}: {info['points']} points, rms {info['rms_px']:.3f} px; fx {info['fx']:.1f} "
            f"fy {info['fy']:.1f} (true 2200), k1 {info['k1']:.4f} (true {LENS[0]}); centre "
            f"{info['distance']:.3f} m from the axis (true {info['true_distance']:.3f} m)")

    # 4. Localize every frame
    est_o, est_b, info_o = [], [], []
    t_loc_o = t_loc_b = 0.0
    for f in range(len(X)):
        obs_c, obs_uv = [], []
        for c in range(3):
            for uv, _area in blobs[f][c]:
                obs_c.append(cams[c])
                obs_uv.append(uv)
        t1 = time.perf_counter()
        if len({id(c) for c in obs_c}) >= 2:
            P, inf = rc.triangulate(obs_c, obs_uv, TUBE)
        else:
            P, inf = np.full(3, np.nan), {"rms_px": np.nan}
        t_loc_o += time.perf_counter() - t1
        est_o.append(P)
        info_o.append(inf["rms_px"])
        t1 = time.perf_counter()
        pick = [(c, max(blobs[f][c], key=lambda b: b[1])[0]) for c in range(3) if blobs[f][c]]
        est_b.append(triangulate_straight([base[c] for c, _ in pick], [uv for _, uv in pick])
                     if len(pick) >= 2 else np.full(3, np.nan))
        t_loc_b += time.perf_counter() - t1
    est_o, est_b = np.array(est_o), np.array(est_b)
    err_o, err_b = (np.linalg.norm(e - X, axis=1) * 1000 for e in (est_o, est_b))
    inside = (np.hypot(X[:, 0], X[:, 1]) <= CAL_R) & (np.abs(X[:, 2]) <= CAL_Z)

    def stats(e):
        e = e[np.isfinite(e)]
        return dict(n=int(e.size), rms=float(np.sqrt(np.mean(e ** 2))), p95=float(np.percentile(e, 95)),
                    max=float(e.max())) if e.size else dict(n=0)

    summary = {}
    say("\nwhere the dot is: 3-D error in mm")
    for name, e in (("refractive (ours)", err_o), ("pinhole + distortion", err_b)):
        s_in, s_out = stats(e[inside]), stats(e[~inside])
        summary[name] = dict(inside=s_in, outside=s_out)
        say(f"  {name:22s} inside the calibration volume ({s_in['n']} frames): rms {s_in['rms']:.4f}, "
            f"p95 {s_in['p95']:.4f}, max {s_in['max']:.4f};  outside ({s_out['n']}): rms {s_out['rms']:.4f}, "
            f"p95 {s_out['p95']:.4f}, max {s_out['max']:.4f}")
    say(f"  ours: median reprojection rms {np.nanmedian(info_o):.3f} px; "
        f"{1e3 * t_loc_o / len(X):.1f} ms per frame (baseline {1e3 * t_loc_b / len(X):.2f} ms)")
    say(f"\ntotal {time.perf_counter() - t_start:.0f} s")

    results = dict(frames=len(X), truth=X.tolist(), ours=est_o.tolist(), baseline=est_b.tolist(),
                   error_mm=dict(ours=err_o.tolist(), baseline=err_b.tolist()), inside=inside.tolist(),
                   summary=summary, matching={k: v.tolist() for k, v in tally.items()}, afrt=afrt_status,
                   image_counts=counts.tolist(), calibration=cal,
                   timing_ms=dict(project=1e3 * t_ours / n_proj, afrt=1e3 * t_afrt / n_proj,
                                  localize_ours=1e3 * t_loc_o / len(X), localize_baseline=1e3 * t_loc_b / len(X)),
                   blobs=[[[b[0] for b in cb] for cb in fr] for fr in blobs], ours_uv=ours_uv, afrt_uv=afrt_uv,
                   offsets_px={k: [float(v) for v in o] for k, o in offsets.items()})
    (HERE / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (HERE / "demo_a_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from figures import animation, summary_figure
    blob_uv = [[[b[0] for b in cb] for cb in fr] for fr in blobs]
    # The cameras do not move: each one's view of the empty tube (rays that miss the water darker) is the backdrop
    backdrops = [rc.render_dots(half_camera(c), TUBE, np.empty((0, 3)), DOT_R, supersample=1, outside=0.55)
                 for c in cams]
    summary_figure(HERE / "summary.png", TUBE, cams, X, est_o, est_b, err_o, err_b, inside, frames8, blob_uv,
                   ours_uv, afrt_uv)
    animation(HERE / "animation.gif", cams, X, err_o, err_b, frames8, blob_uv, ours_uv, afrt_uv, backdrops)
    print("figures written")


def redraw():
    """summary.png again, from results.json (the animation needs the rendered frames, which are not saved)."""
    from figures import summary_figure
    r = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    cams, X = cameras(), np.array(r["truth"])
    arr = lambda k: np.array(r[k], float)                                                  # noqa: E731
    summary_figure(HERE / "summary.png", TUBE, cams, X, arr("ours"), arr("baseline"),
                   np.array(r["error_mm"]["ours"], float), np.array(r["error_mm"]["baseline"], float),
                   np.array(r["inside"]), None, r["blobs"], r["ours_uv"], r["afrt_uv"])
    print("summary.png redrawn")


if __name__ == "__main__":
    main()
