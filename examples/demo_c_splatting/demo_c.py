"""Demo C: Gaussian splatting of the stones inside the water-filled tube.

48 views around the tube train a 3D Gaussian splatting; 12 views between them test it.  The ground-truth images are
ray-traced through the tube (refractive_camera.scene), with no Gaussians in them.  Three cameras, same rasterizer,
same initialisation and schedule:
  * refractive (ours): every image of every Gaussian, by project(), with the analytic footprint Jacobian;
  * pinhole: the true cameras, the window ignored;
  * pinhole + distortion: each camera's own OpenCV model fitted through the tube, the best a pinhole can do.
Two more models train ours from the poses a robot would report, off by about 2 mm and 0.3 degrees (as Demo B's prior):
  * refractive_noisy: the poses as reported;
  * refractive_refined: the poses refined with the Gaussians, through the analytic derivative of the projection in
    the camera's pose (splat.PoseSplatCamera).
Each camera that models the tube draws in training the images it is scored on, and lights only the pixels whose ray
reaches the water: --repair and --water-mask, both on by default.  --no-repair and --no-water-mask train as the demo
first did; their results are named with _norepair and _nomask.

    python examples/demo_c_splatting/demo_c.py              # about an hour on a small GPU
    python examples/demo_c_splatting/demo_c.py --iters 1500 # a quick look
    python examples/demo_c_splatting/demo_c.py --models refractive_noisy,refractive_refined
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parent), str(HERE), str(HERE.parent / "demo_b_sfm")]
import refractive_camera as rc  # noqa: E402
from refractive_camera.scene import render_scene  # noqa: E402
import tube_garden as tg  # noqa: E402
import splat  # noqa: E402

CACHE = Path(os.environ["REFCAM_CACHE"]) / "demo_c_splatting" if "REFCAM_CACHE" in os.environ else HERE / "_cache"
POSE_NOISE = (1.2e-3, math.radians(0.17))       # per axis: centre (m), rotation (rad); 2.1 mm and 0.29 deg rms in 3-D
POSE_FROM = 500                                 # step from which the poses are refined
SIZE, FOCAL = (320, 256), 375.0
BG = (0.08, 0.08, 0.1)
EXTENT = 0.2
OUT_LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT_LINES.append(s)


def views():
    """Training views on three rings; test views between them (interpolation); and 'far' test views from poses
    outside the training rings, closer and higher or farther and lower (extrapolation, as when a robot moves the
    camera somewhere new)."""
    train = tg.ring_cameras(n_per_ring=16, f=FOCAL, size=SIZE, seed=1)
    test = tg.ring_cameras(n_per_ring=4, f=FOCAL, size=SIZE, seed=2, phase=2 * math.pi / 32 + 0.21)
    far = tg.ring_cameras(n_per_ring=6, rings=((0.09, 0.30), (-0.07, 0.46)), f=FOCAL, size=SIZE, seed=3, phase=0.37)
    return train, test, far


def ground_truth(cams, tag):
    path = CACHE / f"gt_{tag}.npz"
    if path.exists():
        z = np.load(path)
        return z["rgb"], z["water"], z["wall"]
    rgbs, water, wall = [], [], []
    for c in cams:
        rgb, centre = render_scene(c, tg.TUBE, tg.objects(), supersample=3, background=BG)
        rgbs.append(rgb.astype(np.float32))
        water.append(centre.inner.reshape(SIZE[1], SIZE[0]))
        r = np.hypot(centre.point[:, 0], centre.point[:, 1])
        wall.append((centre.obj >= 0).reshape(SIZE[1], SIZE[0]) & (r > 0.079).reshape(SIZE[1], SIZE[0]))
    np.savez(path, rgb=np.array(rgbs), water=np.array(water), wall=np.array(wall))
    return np.array(rgbs), np.array(water), np.array(wall)


def calibrate_through_tube(cam, rng):
    """The camera's own OpenCV model (fx, fy, cx, cy, k1, k2, p1, p2, k3, pose) fitted to points of the scene volume
    seen through the tube (one image each), as a pinhole user would calibrate it with a target in the water."""
    from scipy.optimize import least_squares
    n = 4000
    r = 0.09 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(-np.pi, np.pi, n)
    X = np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-0.075, 0.045, n)]
    im = rc.project(X, cam, tg.TUBE)
    one = np.flatnonzero((im.counts() == 1))
    rows = np.array([im.rows_of(i)[0] for i in one])
    ok = im.in_image[rows]
    X, uv = X[one[ok]], im.uv[rows[ok]]
    from sfm import expm_so3, log_so3

    def unpack(p):
        Kp = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        R = expm_so3(p[9:12]) @ cam.R
        return rc.Camera(Kp, R, p[12:15], p[4:9], SIZE)

    def res(p):
        c = unpack(p)
        u, _ = c.project_points(X)
        return np.nan_to_num((u - uv).ravel(), nan=1e3)

    p0 = np.r_[cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2], np.zeros(5), np.zeros(3), cam.t]
    sol = least_squares(res, p0, x_scale="jac", method="lm", max_nfev=4000)
    rms = np.sqrt(np.mean(sol.fun ** 2))
    return unpack(sol.x), rms, len(X)


def psnr(a, b, mask=None):
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    return float(10 * np.log10(1.0 / max(d.mean(), 1e-12)))


def noisy(cams, seed=7):
    """The cameras at the poses a robot would report: centre and rotation off by POSE_NOISE per axis."""
    from sfm import expm_so3
    rng = np.random.default_rng(seed)
    out = []
    for c in cams:
        R = expm_so3(rng.normal(0, POSE_NOISE[1], 3)) @ c.R
        centre = c.center + rng.normal(0, POSE_NOISE[0], 3)
        out.append(rc.Camera(c.K, R, -R @ centre, c.dist, c.size))
    return out


def pose_errors(cams, truth):
    """rms centre error (mm) and rotation error (deg) of cameras against the true ones."""
    from sfm import log_so3
    dc = [np.linalg.norm(a.center - b.center) for a, b in zip(cams, truth)]
    dr = [np.linalg.norm(log_so3(a.R @ b.R.T)) for a, b in zip(cams, truth)]
    return {"centre_mm_rms": 1e3 * float(np.sqrt(np.mean(np.square(dc)))),
            "rotation_deg_rms": float(np.degrees(np.sqrt(np.mean(np.square(dr)))))}


REFRACTIVE = ("refractive", "afrt", "afrt_pure")      # the cameras that model the tube


def splat_camera(kind, cam, dev, args, refine_poses=False):
    """The splat camera of a model: ours, AFRT's images with our Jacobian, AFRT on its own, or a pinhole.  With
    --water-mask a camera that models the tube lights only the pixels whose ray reaches the water.  With --repair ours
    and the AFRT model search a Gaussian again at once when Newton's method loses one of its images: ours by project(),
    the AFRT model by one AFRT start."""
    if refine_poses:
        sc = splat.PoseSplatCamera(cam, tg.TUBE, dev, threads=args.threads)
    elif kind in REFRACTIVE:
        cls = {"refractive": splat.RefractiveSplatCamera, "afrt": splat.AFRTSplatCamera,
               "afrt_pure": splat.AFRTPureSplatCamera}[kind]
        sc = cls(cam, tg.TUBE, dev, threads=args.threads)
    else:
        return splat.PinholeSplatCamera(cam, dev)
    if getattr(args, "water_mask", False):
        sc.mask = torch.tensor(splat.water_mask(cam, tg.TUBE), device=dev)
    if getattr(args, "repair", False) and kind in ("refractive", "afrt"):
        sc.repair = True
    return sc


def result_key(kind, args):
    """The name of a model's results and files: its kind, then _hybrid, _nomask and _norepair where they apply."""
    key = kind + ("_hybrid" if args.footprint == "hybrid" and kind in ("refractive", "afrt") else "")
    key += "_nomask" if not args.water_mask and kind not in ("pinhole", "pinhole_dist") else ""
    key += "_norepair" if not args.repair and kind not in ("pinhole", "pinhole_dist", "afrt_pure") else ""
    return key


def draw(gauss, camera, bg, kind, args):
    """Render with the footprint chosen by --footprint: linear (J Sigma J^T) everywhere, or, for the refractive
    cameras, the unscented footprint where the linear one is wider than 6 px (next to the caustics)."""
    if args.footprint == "hybrid" and kind in ("refractive", "afrt"):
        return splat.render_ut(gauss, camera, bg, above=6.0, min_points=2)
    return splat.render(gauss, camera, bg)


def train(kind, cams, gt, args, dev, log_every=500, refine_poses=False):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n0 = args.init
    r = 0.092 * np.sqrt(rng.uniform(0, 1, n0))
    a = rng.uniform(-np.pi, np.pi, n0)
    init = np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(-0.075, 0.05, n0)]
    gauss = splat.Gaussians(torch.tensor(init, dtype=torch.float32, device=dev), 0.003,
                            torch.full((n0, 3), 0.5, device=dev))
    gt_t = torch.tensor(gt, device=dev)
    bg = torch.tensor(BG, device=dev, dtype=torch.float32)
    scams = [splat_camera(kind, c, dev, args, refine_poses) for c in cams]
    last = np.full(len(scams), -10 ** 9)
    pose_opt, gauge = None, None
    if refine_poses:
        pose_opt = torch.optim.Adam([{"params": [c.omega for c in scams], "lr": args.pose_lr_rot},
                                     {"params": [c.dc for c in scams], "lr": args.pose_lr_pos}], eps=1e-15)
        # the gauge: turning every camera (and the scene) about the tube's axis, or shifting them along it, changes no
        # image; that motion of the cameras is projected out after every step
        zh = np.array([0.0, 0.0, 1.0])
        g_rot = np.concatenate([np.r_[-c.R @ zh, np.cross(zh, c.center)] for c in cams])
        g_tz = np.concatenate([np.r_[0.0, 0.0, 0.0, zh] for _ in cams])
        u1 = g_rot / np.linalg.norm(g_rot)
        u2 = g_tz - (g_tz @ u1) * u1
        u2 /= np.linalg.norm(u2)
        gauge = torch.tensor(np.stack([u1, u2]), dtype=torch.float64, device=dev)

    def make_opt():
        return torch.optim.Adam([{"params": [gauss.means], "lr": args.pos_lr * EXTENT, "name": "means"},
                                 {"params": [gauss.log_scales], "lr": 0.005},
                                 {"params": [gauss.quats], "lr": 0.001},
                                 {"params": [gauss.opacity_logits], "lr": 0.05},
                                 {"params": [gauss.color_logits], "lr": 0.01}], eps=1e-15)

    opt = make_opt()
    acc = torch.zeros(len(gauss.means), device=dev)
    cnt = torch.zeros(len(gauss.means), device=dev)
    t0 = time.perf_counter()
    lost = 0
    for it in range(args.iters):
        v = int(rng.integers(len(scams)))
        if kind in REFRACTIVE and it - last[v] >= args.refresh:
            scams[v].refresh(gauss.means)
            last[v] = it
        img, uv2d, idx, nbig = draw(gauss, scams[v], bg, kind, args)
        lost += nbig
        l1 = (img - gt_t[v]).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - splat.ssim(img, gt_t[v]))
        opt.zero_grad(set_to_none=True)
        if pose_opt is not None:
            pose_opt.zero_grad(set_to_none=True)
        loss.backward()
        for prm in gauss.parameters():                     # a safety net: no step may carry a NaN or an infinity
            if prm.grad is not None:
                torch.nan_to_num_(prm.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if uv2d.grad is not None:
            gn = uv2d.grad.norm(dim=1).float()
            acc.index_add_(0, idx, gn)
            cnt.index_add_(0, idx, torch.ones_like(gn))
        opt.step()
        if pose_opt is not None and it >= POSE_FROM:
            pose_opt.step()
            with torch.no_grad():
                pvec = torch.cat([torch.cat([c.omega, c.dc]) for c in scams])
                pvec = pvec - (gauge @ pvec) @ gauge
                for i, c in enumerate(scams):
                    c.omega.copy_(pvec[6 * i:6 * i + 3])
                    c.dc.copy_(pvec[6 * i + 3:6 * i + 6])
            for gp, base in zip(pose_opt.param_groups, (args.pose_lr_rot, args.pose_lr_pos)):
                gp["lr"] = base * (0.1 ** ((it - POSE_FROM) / max(args.iters - POSE_FROM, 1)))
        frac = it / max(args.iters - 1, 1)
        for gp in opt.param_groups:
            if gp.get("name") == "means":
                gp["lr"] = args.pos_lr * EXTENT * (0.01 ** frac)
        if args.densify_from <= it < args.densify_until and it % 100 == 0 and it > 0:
            n_before = len(gauss.means)
            parents, keep, acc, cnt = densify(gauss, acc, cnt, args,
                                              r_max=tg.TUBE.radii[-1] if kind in REFRACTIVE else None)
            for c in scams:
                if kind in REFRACTIVE:
                    c.inherit(parents, n_before)
                    c.remap(keep)
            opt = make_opt()
        if it % log_every == 0 or it == args.iters - 1:
            say(f"    {kind:9s} it {it:5d}  loss {loss.item():.4f}  gaussians {len(gauss.means)}  "
                f"{time.perf_counter() - t0:.0f} s")
    fails = sum(getattr(c, "failures", 0) for c in scams)
    info = {"train_s": time.perf_counter() - t0, "gaussians": int(len(gauss.means)), "footprint": args.footprint,
            "newton_failures": int(fails), "splats_cut_near_caustics": int(lost)}
    if refine_poses:
        info["refined_cameras"] = [c.camera() for c in scams]
    return gauss, scams, info


@torch.no_grad()
def densify(gauss, acc, cnt, args, r_max=None):
    """3DGS's adaptive density control, simplified: clone small Gaussians with large screen gradients, split large
    ones, prune transparent ones.  Returns the parents of the new Gaussians (in order) and the keep mask."""
    dev = gauss.means.device
    N = len(gauss.means)
    avg = acc / cnt.clamp(min=1)
    grow = avg > args.grad_thr
    big = torch.exp(gauss.log_scales).max(1).values > 0.01 * EXTENT
    if N > args.max_gaussians:
        grow &= False
    clone = torch.nonzero(grow & ~big).flatten()
    split = torch.nonzero(grow & big).flatten()
    new = {k: [] for k in ("means", "log_scales", "quats", "opacity_logits", "color_logits")}
    for k in new:
        new[k].append(getattr(gauss, k).data[clone])
    R = splat.quat_to_rot(gauss.quats.data[split])
    for _ in range(2):
        z = torch.randn(len(split), 3, device=dev) * torch.exp(gauss.log_scales.data[split])
        new["means"].append(gauss.means.data[split] + (R @ z[..., None])[..., 0])
        new["log_scales"].append(gauss.log_scales.data[split] - math.log(1.6))
        for k in ("quats", "opacity_logits", "color_logits"):
            new[k].append(getattr(gauss, k).data[split])
    parents = torch.cat([clone, split, split])
    for k in new:
        p = getattr(gauss, k)
        setattr(gauss, k, torch.nn.Parameter(torch.cat([p.data, *new[k]])))
    M = len(gauss.means)
    keep = torch.ones(M, dtype=torch.bool, device=dev)
    keep[split] = False
    keep &= torch.sigmoid(gauss.opacity_logits.data) > 0.005
    keep &= torch.exp(gauss.log_scales.data).max(1).values < 0.1 * EXTENT
    keep &= torch.isfinite(gauss.means.data).all(1)
    if r_max is not None:                                  # through the window, a Gaussian outside the water is never drawn
        keep &= torch.hypot(gauss.means.data[:, 0], gauss.means.data[:, 1]) < r_max
    for k in new:
        setattr(gauss, k, torch.nn.Parameter(getattr(gauss, k).data[keep]))
    return parents, keep, torch.zeros(int(keep.sum()), device=dev), torch.zeros(int(keep.sum()), device=dev)


@torch.no_grad()
def evaluate(kind, gauss, cams, gt, water, wall, dev, args, tag):
    bg = torch.tensor(BG, device=dev, dtype=torch.float32)
    imgs = []
    for c in cams:
        sc = splat_camera(kind, c, dev, args)
        sc.refresh(gauss.means)
        img, *_ = draw(gauss, sc, bg, kind, args)
        imgs.append(img.clamp(0, 1).cpu().numpy())
    imgs = np.array(imgs)
    np.save(CACHE / f"render_test_{tag}.npy", imgs.astype(np.float16))
    ps = [psnr(i, g) for i, g in zip(imgs, gt)]
    pw = [psnr(i, g, w) for i, g, w in zip(imgs, gt, water)]
    pn = [psnr(i, g, w) for i, g, w in zip(imgs, gt, wall) if w.any()]
    ss = [float(splat.ssim(torch.tensor(i), torch.tensor(g))) for i, g in zip(imgs, gt)]
    from demo_b import surface_distance
    op = torch.sigmoid(gauss.opacity_logits).cpu().numpy()
    m = gauss.means.detach().cpu().numpy()[op > 0.3]
    m = m[np.isfinite(m).all(1)]
    sd = 1e3 * surface_distance(m.astype(float), tg.objects())
    return {"psnr": float(np.mean(ps)), "psnr_water": float(np.mean(pw)), "psnr_near_wall": float(np.mean(pn)),
            "ssim": float(np.mean(ss)), "psnr_per_view": ps,
            "centres_opaque": int(len(m)), "centre_to_surface_mm": {
                "median": float(np.median(sd)), "p90": float(np.percentile(sd, 90)),
                "within_1mm": float(np.mean(sd < 1.0))}}


def parser():
    """The demo's options; parser().parse_args([]) gives its defaults, for the scripts that draw its models."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--init", type=int, default=10000)
    ap.add_argument("--refresh", type=int, default=500, help="steps between project() refreshes of a view")
    ap.add_argument("--pos-lr", type=float, default=1.6e-4)
    ap.add_argument("--grad-thr", type=float, default=2e-6)
    ap.add_argument("--densify-from", type=int, default=300)
    ap.add_argument("--densify-until", type=int, default=2500)
    ap.add_argument("--max-gaussians", type=int, default=40000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="refractive,pinhole,pinhole_dist")
    ap.add_argument("--pose-lr-rot", type=float, default=1e-4, help="rad per step, refined poses")
    ap.add_argument("--pose-lr-pos", type=float, default=1e-4, help="m per step, refined poses")
    ap.add_argument("--footprint", default="linear", choices=["linear", "hybrid"],
                    help="hybrid: the unscented footprint where the linear one is wider than 6 px")
    ap.add_argument("--water-mask", action=argparse.BooleanOptionalAction, default=True,
                    help="the tube's cameras light only the pixels whose ray reaches the water")
    ap.add_argument("--repair", action=argparse.BooleanOptionalAction, default=True,
                    help="ours and the AFRT model search a Gaussian again at once when Newton's method loses one of "
                         "its images")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the trained models in _cache/ again")
    return ap


def main():
    args = parser().parse_args()
    CACHE.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_cams, test_cams, far_cams = views()
    say(f"Demo C: {len(train_cams)} training and {len(test_cams)} test views of {SIZE[0]} x {SIZE[1]} px; "
        f"device {dev}")
    t0 = time.perf_counter()
    gt_train, _, _ = ground_truth(train_cams, "train")
    gt_test, water_test, wall_test = ground_truth(test_cams, "test")
    gt_far, water_far, wall_far = ground_truth(far_cams, "far")
    say(f"ground truth rendered in {time.perf_counter() - t0:.0f} s")
    rpath = HERE / "results.json"
    results = json.loads(rpath.read_text()) if rpath.exists() else {}
    for kind in args.models.split(","):
        key = result_key(kind, args)
        say(f"\n== {key}")
        tr, te, tf = train_cams, test_cams, far_cams
        model_kind = kind if kind in REFRACTIVE else "pinhole"
        refine = kind == "refractive_refined"
        if kind in ("refractive_noisy", "refractive_refined"):
            model_kind = "refractive"
            tr = noisy(train_cams)
            say(f"  training poses as a robot reports them: {pose_errors(tr, train_cams)}")
        if kind == "pinhole_dist":
            rng = np.random.default_rng(5)
            fits = [calibrate_through_tube(c, rng) for c in train_cams + test_cams + far_cams]
            rms = [f[1] for f in fits]
            say(f"  per-camera OpenCV fits through the tube: rms {np.median(rms):.2f} px median, "
                f"{np.max(rms):.2f} px max, on {int(np.median([f[2] for f in fits]))} points each")
            n1, n2 = len(train_cams), len(train_cams) + len(test_cams)
            tr, te, tf = [f[0] for f in fits[:n1]], [f[0] for f in fits[n1:n2]], [f[0] for f in fits[n2:]]
        if args.eval_only:
            sd = torch.load(CACHE / f"gaussians_{key}.pt")
            gauss = splat.Gaussians(sd["means"].to(dev), 0.001, torch.full((len(sd["means"]), 3), 0.5, device=dev))
            gauss.load_state_dict({k: v.to(dev) for k, v in sd.items()})
            info = {k: v for k, v in results.get(key, {}).items() if k in ("train_s", "gaussians", "newton_failures",
                                                                            "splats_cut_near_caustics")}
        else:
            gauss, _, info = train(model_kind, tr, gt_train, args, dev, refine_poses=refine)
            if kind in ("refractive_noisy", "refractive_refined"):
                info["poses_reported"] = pose_errors(tr, train_cams)
            if refine:
                info["poses_refined"] = pose_errors(info.pop("refined_cameras"), train_cams)
                say(f"  refined training poses: {info['poses_refined']}")
        ev = evaluate(model_kind, gauss, te, gt_test, water_test, wall_test, dev, args, key)
        info.update(ev)
        ef = evaluate(model_kind, gauss, tf, gt_far, water_far, wall_far, dev, args, key + "_far")
        info["far"] = {k: ef[k] for k in ("psnr", "psnr_water", "psnr_near_wall", "ssim", "psnr_per_view")}
        if kind == "pinhole_dist":
            info["calibration_rms_px"] = float(np.median(rms))
        results[key] = info
        if not args.eval_only:
            torch.save({k: v.detach().cpu() for k, v in gauss.state_dict().items()}, CACHE / f"gaussians_{key}.pt")
        say(f"  test PSNR {ev['psnr']:.2f} dB (water {ev['psnr_water']:.2f}, near the wall {ev['psnr_near_wall']:.2f}),"
            f" SSIM {ev['ssim']:.3f}; opaque centres to the true surface: median "
            f"{ev['centre_to_surface_mm']['median']:.2f} mm")
        say(f"  far views: PSNR {ef['psnr']:.2f} dB (water {ef['psnr_water']:.2f}, near the wall "
            f"{ef['psnr_near_wall']:.2f}), SSIM {ef['ssim']:.3f}")
    rpath.write_text(json.dumps(results, indent=1))
    with open(HERE / "demo_c_out.txt", "a") as fh:
        fh.write("\n".join(OUT_LINES) + "\n")


if __name__ == "__main__":
    main()
