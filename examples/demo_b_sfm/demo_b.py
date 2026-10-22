"""Demo B: structure from motion around a water-filled tube.

A robot-held camera circles the paper's reference tube (acrylic, water) and photographs the stones inside from 60
poses.  Its poses are known roughly, as from the robot's kinematics.  From the images alone (SIFT features, matches,
tracks) the poses and the stones are reconstructed with three camera models in the same bundle adjustment:
  * refractive (ours): project() and its analytic Jacobians;
  * pinhole + AFRT: the same window, each point projected by one AFRT start;
  * pinhole + distortion: the window ignored, intrinsics refined;
and COLMAP's own pipeline (pycolmap, OPENCV camera), run as it comes, is the out-of-the-box reference.

    python examples/demo_b_sfm/demo_b.py            # about 30 minutes; caches renders and features in _cache/
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
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parent), str(HERE)]
import refractive_camera as rc  # noqa: E402
from refractive_camera.scene import Box, Cylinder, Disk, Sphere, cast, render_scene  # noqa: E402
import tube_garden as tg  # noqa: E402
import sfm  # noqa: E402

CACHE_ROOT = Path(os.environ["REFCAM_CACHE"]) / "demo_b_sfm" if "REFCAM_CACHE" in os.environ else HERE / "_cache"
CACHE = CACHE_ROOT
SIZE, FOCAL = (1280, 1024), 1500.0
K = np.array([[FOCAL, 0, (SIZE[0] - 1) / 2], [0, FOCAL, (SIZE[1] - 1) / 2], [0, 0, 1]])
DIST = np.zeros(5)
SIG_C, SIG_R = 0.0015, np.radians(0.2)          # pose priors: 1.5 mm and 0.2 degrees per axis
OUT_LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT_LINES.append(s)


# ---- rendering ---------------------------------------------------------------------------------------------------

def _render_one(args):
    k, R, t, path = args
    from PIL import Image
    cam = rc.Camera(K, R, t, DIST, SIZE)
    rgb, centre = render_scene(cam, tg.TUBE, tg.objects(), supersample=2)
    Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(path)
    water = centre.inner.reshape(SIZE[1], SIZE[0])
    Image.fromarray((water * 255).astype(np.uint8)).save(str(path).replace(".png", "_water.png"))
    return k


def render_views(cams, workers):
    CACHE.mkdir(exist_ok=True)
    jobs = [(k, c.R, c.t, CACHE / f"view_{k:02d}.png") for k, c in enumerate(cams)
            if not (CACHE / f"view_{k:02d}.png").exists()]
    if jobs:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_render_one, jobs))
        say(f"rendered {len(jobs)} views in {time.perf_counter() - t0:.0f} s")


# ---- features, matches, tracks ------------------------------------------------------------------------------------

def features(cams_true, n_feat):
    import cv2
    from PIL import Image
    path = CACHE / f"features_{n_feat}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return list(z["kp"]), list(z["desc"]), list(z["gt"]), list(z["gt_obj"])
    sift = cv2.SIFT_create(nfeatures=n_feat)
    kps, descs, gts, gobj = [], [], [], []
    objs = tg.objects()
    for k, cam in enumerate(cams_true):
        img = np.asarray(Image.open(CACHE / f"view_{k:02d}.png").convert("L"))
        water = np.asarray(Image.open(CACHE / f"view_{k:02d}_water.png")) > 0
        mask = cv2.erode(water.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        kp, desc = sift.detectAndCompute(img, mask)
        uv = np.array([p.pt for p in kp], float)
        h = cast(cam, tg.TUBE, uv, objs)                      # ground truth: what each keypoint's ray meets
        kps.append(uv)
        descs.append(desc)
        gts.append(h.point)
        gobj.append(h.obj)
    np.savez(path, kp=np.array(kps, dtype=object), desc=np.array(descs, dtype=object),
             gt=np.array(gts, dtype=object), gt_obj=np.array(gobj, dtype=object))
    return kps, descs, gts, gobj


def match_and_verify(kps, descs, cams_prior, max_angle=50.0, ratio=0.8, tol=0.008):
    """SIFT matches between views up to max_angle apart around the tube, kept when the two keypoints' rays (traced
    through the window with the prior poses) pass within `tol` of each other in the water."""
    import cv2
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rays = [rc.back_project(c, tg.TUBE, kp) for c, kp in zip(cams_prior, kps)]
    az = np.array([np.arctan2(c.center[1], c.center[0]) for c in cams_prior])
    pairs = []
    for i in range(len(kps)):
        for j in range(i + 1, len(kps)):
            d = np.degrees(abs(np.angle(np.exp(1j * (az[i] - az[j])))))
            if d <= max_angle:
                pairs.append((i, j))
    kept, total = {}, 0
    for i, j in pairs:
        m1 = bf.knnMatch(descs[i], descs[j], k=2)
        m2 = bf.knnMatch(descs[j], descs[i], k=1)
        back = {m[0].queryIdx: m[0].trainIdx for m in m2 if m}
        ab = np.array([(m[0].queryIdx, m[0].trainIdx) for m in m1
                       if len(m) == 2 and m[0].distance < ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx],
                      int).reshape(-1, 2)
        total += len(ab)
        if not len(ab):
            continue
        ri, rj = rays[i], rays[j]
        a, b = ab[:, 0], ab[:, 1]
        ok = ri.ok[a] & rj.ok[b]
        ea, da, La = ri.entry[a], ri.direction[a], ri.length[a]
        eb, db, Lb = rj.entry[b], rj.direction[b], rj.length[b]
        w0 = ea - eb
        bb = np.einsum("ij,ij->i", da, db)
        dd = np.einsum("ij,ij->i", da, w0)
        ee = np.einsum("ij,ij->i", db, w0)
        den = 1 - bb * bb
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.clip((bb * ee - dd) / den, 0, La)
            t = np.clip((ee - bb * dd) / den, 0, Lb)
        gap = np.linalg.norm((ea + s[:, None] * da) - (eb + t[:, None] * db), axis=1)
        good = ok & (den > 1e-3) & (gap < tol)
        kept[(i, j)] = ab[good]
    say(f"matching: {len(pairs)} view pairs, {total} mutual ratio-test matches, "
        f"{sum(len(v) for v in kept.values())} kept by the ray check")
    return kept


def build_tracks(matches, n_feats, min_len=3):
    offs = np.concatenate([[0], np.cumsum(n_feats)])
    parent = np.arange(offs[-1])

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), ab in matches.items():
        for a, b in ab:
            ra, rb = find(offs[i] + a), find(offs[j] + b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(x) for x in range(offs[-1])])
    view_of = np.searchsorted(offs, np.arange(offs[-1]), side="right") - 1
    order = np.argsort(roots, kind="stable")
    tracks, conflicts = [], 0
    start = 0
    rs = roots[order]
    for end in np.r_[np.flatnonzero(np.diff(rs)) + 1, len(rs)]:
        nodes = order[start:end]
        start = end
        if len(nodes) < 2:
            continue
        views = view_of[nodes]
        if len(np.unique(views)) < len(views):
            conflicts += 1
            continue
        if len(nodes) >= min_len:
            tracks.append([(int(v), int(n - offs[v])) for v, n in zip(views, nodes)])
    say(f"tracks: {len(tracks)} with at least {min_len} views ({conflicts} dropped: two features in one view)")
    return tracks


# ---- ground truth and errors --------------------------------------------------------------------------------------

def surface_distance(p, objs):
    """Distance from points (n, 3) to the nearest surface of the scene."""
    best = np.full(len(p), np.inf)
    for ob in objs:
        q = (p - ob.center) @ ob.rotation
        if isinstance(ob, Sphere):
            d = np.linalg.norm(q, axis=1) - ob.radius
        elif isinstance(ob, Box):
            e = np.abs(q) - ob.half
            d = np.linalg.norm(np.maximum(e, 0), axis=1) + np.minimum(e.max(1), 0)
        elif isinstance(ob, Cylinder):
            e = np.c_[np.hypot(q[:, 0], q[:, 1]) - ob.radius, np.abs(q[:, 2]) - ob.half_height]
            d = np.linalg.norm(np.maximum(e, 0), axis=1) + np.minimum(e.max(1), 0)
        elif isinstance(ob, Disk):
            d = np.hypot(np.maximum(np.hypot(q[:, 0], q[:, 1]) - ob.radius, 0), q[:, 2])
        best = np.minimum(best, np.abs(d))
    return best


def track_truth(tracks, gts, gobj, tol=5e-4):
    """Each track's true point (the median of what its keypoints' rays meet) and whether its keypoints agree."""
    X = np.full((len(tracks), 3), np.nan)
    ok = np.zeros(len(tracks), bool)
    for t, tr in enumerate(tracks):
        P = np.array([gts[v][f] for v, f in tr])
        o = np.array([gobj[v][f] for v, f in tr])
        if (o < 0).any():
            continue
        med = np.median(P, 0)
        X[t] = med
        ok[t] = np.all(np.linalg.norm(P - med, axis=1) < tol)
    return X, ok


def rot_err_deg(Ra, Rb):
    return np.degrees(np.linalg.norm(sfm.log_so3(np.einsum("nij,nkj->nik", Ra, Rb)), axis=1))


# ---- the methods ---------------------------------------------------------------------------------------------------

def init_points(model_name, cams_prior, tracks, kps, intr=None):
    """Least-squares point of each track's rays, traced through the window (refractive) or straight (pinhole)."""
    X = np.zeros((len(tracks), 3))
    A = np.zeros((len(tracks), 3, 3))
    b = np.zeros((len(tracks), 3))
    by_view = {}
    for t, tr in enumerate(tracks):
        for v, f in tr:
            by_view.setdefault(v, []).append((t, f))
    for v, items in by_view.items():
        ts = np.array([x[0] for x in items])
        fs = np.array([x[1] for x in items])
        cam = cams_prior[v]
        if model_name == "pinhole":
            o, d = cam.rays(kps[v][fs])
        else:
            tr = rc.back_project(cam, tg.TUBE, kps[v][fs])
            o, d = tr.entry, tr.direction
            bad = ~tr.ok
            o, d = np.where(bad[:, None], 0.0, o), np.where(bad[:, None], 0.0, d)
        M = np.eye(3) - d[:, :, None] * d[:, None, :]
        np.add.at(A, ts, M)
        np.add.at(b, ts, np.einsum("nij,nj->ni", M, o))
    good = np.linalg.matrix_rank(A) == 3
    X[good] = np.linalg.solve(A[good], b[good][..., None])[..., 0]
    X[~good] = np.nan
    return X


def align_axis(src, dst):
    """Rotation about the tube's axis and shift along it (the two freedoms the tube leaves) taking src to dst."""
    th = np.arctan2(np.sum(src[:, 0] * dst[:, 1] - src[:, 1] * dst[:, 0]),
                    np.sum(src[:, 0] * dst[:, 0] + src[:, 1] * dst[:, 1]))
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    return 1.0, R, np.array([0.0, 0.0, np.mean(dst[:, 2] - src[:, 2])])


def evaluate(X, have, Rs, cs, cams_true, gtX, gt_ok):
    """Errors of points (against each track's true point) and poses, in three frames:
    'absolute' as reconstructed; 'axis' after the rotation about the axis and shift along it that best align the
    camera centres (the two freedoms the tube cannot fix, held only by the priors); 'similarity' after the best
    similarity of the camera centres (shape only, the usual SfM metric)."""
    Rt = np.array([c.R for c in cams_true])
    ct = np.array([c.center for c in cams_true])
    m = have & gt_ok
    r_axis = np.hypot(gtX[m, 0], gtX[m, 1])
    res = {}
    for frame, fit in (("absolute", None), ("axis", align_axis), ("similarity", umeyama)):
        s, R, t = (1.0, np.eye(3), np.zeros(3)) if fit is None else fit(cs, ct)
        Xa = (s * (R @ X.T)).T + t
        ca = (s * (R @ cs.T)).T + t
        err = 1e3 * np.linalg.norm(Xa[m] - gtX[m], axis=1)
        e = {"scale": float(s), "point_err_mm": _stats(err),
             "centre_err_mm": _stats(1e3 * np.linalg.norm(ca - ct, axis=1)),
             "rot_err_deg": _stats(rot_err_deg(np.einsum("nij,jk->nik", Rs, R.T), Rt)),
             "surface_dist_mm": _stats(1e3 * surface_distance(Xa[have], tg.objects())), "point_err_by_r_mm": {}}
        for lo, hi in [(0, 50), (50, 70), (70, 85), (85, 94)]:
            sel = (r_axis >= lo / 1e3) & (r_axis < hi / 1e3)
            e["point_err_by_r_mm"][f"{lo}-{hi}"] = _stats(err[sel])
        res[frame] = e
    return res


def run_model(name, cams_true, cams_prior, tracks, kps, gtX, gt_ok, workers):
    Rp = np.array([c.R for c in cams_prior])
    cp = np.array([c.center for c in cams_prior])
    obs_v = np.array([v for tr in tracks for v, f in tr])
    obs_p = np.array([t for t, tr in enumerate(tracks) for _ in tr])
    obs_uv = np.array([kps[v][f] for tr in tracks for v, f in tr])
    if name == "refractive":
        model = sfm.RefractiveModel(tg.TUBE, K, DIST, SIZE, threads=workers)
        intr = np.zeros(0)
    elif name == "afrt":
        model = sfm.AFRTModel(tg.TUBE, K, DIST, SIZE, workers=workers)
        intr = np.zeros(0)
    elif name == "afrt_pure":
        model = sfm.AFRTPureModel(tg.TUBE, K, DIST, SIZE, workers=workers)
        intr = np.zeros(0)
    else:
        model = sfm.PinholeModel(SIZE)
        intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0, 0], float)
    X0 = init_points(name, cams_prior, tracks, kps)
    use = np.isfinite(X0).all(1)
    keep_obs = use[obs_p]
    obs = {"view": obs_v[keep_obs], "point": obs_p[keep_obs], "uv": obs_uv[keep_obs]}
    X0 = np.where(use[:, None], X0, 0.0)
    t0 = time.perf_counter()
    say(f"  {name}: {use.sum()} points, {keep_obs.sum()} observations; bundle adjustment")
    Rs, cs, intr, X, info = sfm.bundle_adjust(model, Rp, cp, intr, X0, obs, prior_R=Rp, prior_c=cp, sigma_rot=SIG_R,
                                              sigma_c=SIG_C, iters=30, log=say)
    # drop the observations the model cannot explain, then refine again
    res = info["residual"]
    med = np.nanmedian(res)
    thr = 3.0 if name != "pinhole" else max(3.0, 4 * med)
    good = np.isfinite(res) & (res < thr)
    obs2 = {k: v[good] for k, v in obs.items()}
    say(f"  {name}: kept {good.sum()} of {len(good)} observations (residual < {thr:.2f} px); refining")
    Rs, cs, intr, X, info2 = sfm.bundle_adjust(model, Rs, cs, intr, X, obs2, prior_R=Rp, prior_c=cp,
                                               sigma_rot=SIG_R, sigma_c=SIG_C, iters=30, log=say)
    elapsed = time.perf_counter() - t0
    if hasattr(model, "close"):
        model.close()
    nobs = np.bincount(obs2["point"], minlength=len(X))
    have = use & (nobs >= 2)
    out = {"points": int(have.sum()), "observations": int(len(obs2["view"])),
           "observations_before": int(len(good)), "time_s": elapsed,
           "residual_median_px": float(np.nanmedian(info2["residual"])),
           "residual_rms_px": float(np.sqrt(np.nanmean(info2["residual"] ** 2))), "intrinsics": intr.tolist()}
    out.update(evaluate(X, have, Rs, cs, cams_true, gtX, gt_ok))
    np.savez(CACHE / f"result_{name}.npz", X=X, have=have, Rs=Rs, cs=cs, intr=intr, obs_view=obs2["view"],
             obs_point=obs2["point"], obs_uv=obs2["uv"], residual=info2["residual"])
    return out


def _stats(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if not e.size:
        return {"n": 0}
    return {"n": int(e.size), "median": float(np.median(e)), "rms": float(np.sqrt(np.mean(e ** 2))),
            "p95": float(np.percentile(e, 95)), "max": float(e.max())}


def run_colmap(cams_true, images=None, tag="colmap"):
    """COLMAP's own incremental pipeline (pycolmap), OPENCV camera started at the true focal length."""
    import pycolmap
    images = CACHE if images is None else images
    work = images / tag
    work.mkdir(exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    names = [f"view_{k:02d}.png" for k in range(len(cams_true))]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    # COLMAP puts the first pixel's centre at (0.5, 0.5)
    reader.camera_params = f"{FOCAL},{FOCAL},{K[0, 2] + 0.5},{K[1, 2] + 0.5},0,0,0,0"
    t0 = time.perf_counter()
    pycolmap.extract_features(db, images, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    recs = pycolmap.incremental_mapping(db, images, work / "sparse")
    elapsed = time.perf_counter() - t0
    if not recs:
        return {"registered": 0, "time_s": elapsed, "verified_pairs": pycolmap_pair_stats(db)}
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    ids = {im.name: im for im in rec.images.values() if im.has_pose}
    src, dst, rot_pairs = [], [], []
    for k, name in enumerate(names):
        if name in ids:
            pose = ids[name].cam_from_world()
            Rm = pose.rotation.matrix()
            src.append(-Rm.T @ pose.translation)
            dst.append(cams_true[k].center)
            rot_pairs.append((Rm, cams_true[k].R))
    src, dst = np.array(src), np.array(dst)
    s, Ra, ta = umeyama(src, dst)
    centres = (s * (Ra @ src.T)).T + ta
    pts = np.array([p.xyz for p in rec.points3D.values()])
    pts = (s * (Ra @ pts.T)).T + ta
    inside = tg.TUBE.inside(pts)
    rerr = [np.degrees(np.linalg.norm(sfm.log_so3((Rm @ Ra.T) @ Rt.T))) for Rm, Rt in rot_pairs]
    cam = rec.cameras[next(iter(rec.cameras))]
    np.savez(images / f"result_{tag}.npz", points=pts, centres=centres)
    return {"registered": int(len(ids)), "points": int(len(pts)), "points_in_water": int(inside.sum()),
            "verified_pairs": pycolmap_pair_stats(db),
            "time_s": elapsed, "scale": float(s),
            "centre_err_mm": _stats(1e3 * np.linalg.norm(centres - dst, axis=1)),
            "rot_err_deg": _stats(rerr),
            "surface_dist_mm": _stats(1e3 * surface_distance(pts[inside], tg.objects())),
            "camera": [float(v) for v in cam.params]}


def pycolmap_pair_stats(db):
    """How many image pairs COLMAP's two-view geometry verified, and their median inlier count."""
    import sqlite3
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT rows FROM two_view_geometries WHERE rows > 0").fetchall()
    con.close()
    n = [r[0] for r in rows]
    return {"pairs": len(n), "median_inliers": float(np.median(n)) if n else 0.0}


AIR = rc.CylindricalWindow(radii=(0.100, 0.094), indices=(1.0, 1.0, 1.0))


def _render_air(args):
    k, R, t, path = args
    from PIL import Image
    cam = rc.Camera(K, R, t, DIST, SIZE)
    rgb, _ = render_scene(cam, AIR, tg.objects(), supersample=2)
    Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(path)
    return k


def run_colmap_air(cams_true, workers):
    """The control: the same scene and cameras with no refraction (every index 1), through COLMAP."""
    air = CACHE.parent / (CACHE.name + "_air")
    air.mkdir(exist_ok=True)
    jobs = [(k, c.R, c.t, air / f"view_{k:02d}.png") for k, c in enumerate(cams_true)
            if not (air / f"view_{k:02d}.png").exists()]
    if jobs:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_render_air, jobs))
    return run_colmap(cams_true, images=air, tag="colmap_air")


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


# ---- main ---------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--features", type=int, default=4000)
    ap.add_argument("--models", default="refractive,afrt,pinhole,colmap")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-ring", type=int, default=20)
    args = ap.parse_args()
    global CACHE
    CACHE = CACHE_ROOT / f"ring{args.per_ring}"
    CACHE.mkdir(parents=True, exist_ok=True)
    cams_true = tg.ring_cameras(n_per_ring=args.per_ring, f=FOCAL, size=SIZE)
    rng = np.random.default_rng(args.seed)
    cams_prior = []
    for c in cams_true:                            # the robot's estimate of each pose
        R = sfm.expm_so3(rng.normal(0, SIG_R, 3)) @ c.R
        cams_prior.append(sfm.camera(K, DIST, R, c.center + rng.normal(0, SIG_C, 3), SIZE))
    say(f"Demo B: {len(cams_true)} views of {SIZE[0]} x {SIZE[1]} px, f = {FOCAL:.0f} px; pose priors "
        f"{SIG_C * 1e3:.1f} mm and {np.degrees(SIG_R):.2f} deg per axis")
    render_views(cams_true, args.workers)
    models = args.models.split(",")
    rpath = HERE / "results.json"
    results = json.loads(rpath.read_text()) if rpath.exists() else {}
    if any(m in ("refractive", "afrt", "afrt_pure", "pinhole") for m in models):
        kps, descs, gts, gobj = features(cams_true, args.features)
        say(f"features: {np.mean([len(k) for k in kps]):.0f} per view")
        matches = match_and_verify(kps, descs, cams_prior)
        tracks = build_tracks(matches, [len(k) for k in kps])
        gtX, gt_ok = track_truth(tracks, gts, gobj)
        say(f"ground truth: {gt_ok.sum()} of {len(tracks)} tracks have keypoints on one surface point "
            f"(within 0.5 mm)")
        prior_c = np.linalg.norm(np.array([c.center for c in cams_prior]) - np.array([c.center for c in cams_true]),
                                 axis=1)
        prior_r = rot_err_deg(np.array([c.R for c in cams_prior]), np.array([c.R for c in cams_true]))
        results["setup"] = {"views": len(cams_true), "size": SIZE, "focal": FOCAL, "tracks": len(tracks),
                            "tracks_true": int(gt_ok.sum()), "prior_centre_err_mm": _stats(1e3 * prior_c),
                            "prior_rot_err_deg": _stats(prior_r)}
    for name in models:
        say(f"\n== {name}")
        if name == "colmap":
            results[name] = run_colmap(cams_true)
        elif name == "colmap_air":
            results[name] = run_colmap_air(cams_true, args.workers)
        else:
            results[name] = run_model(name, cams_true, cams_prior, tracks, kps, gtX, gt_ok, args.workers)
        say(json.dumps(results[name], indent=1))
    rpath.write_text(json.dumps(results, indent=1))
    with open(HERE / "demo_b_out.txt", "a") as fh:
        fh.write("\n".join(OUT_LINES) + "\n")


if __name__ == "__main__":
    main()
