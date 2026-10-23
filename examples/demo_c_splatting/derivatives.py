"""Demo C, part 2: the derivative a rasterizer needs, and the images AFRT misses.

A rasterizer needs, for every Gaussian and view, the Jacobian J = d(uv)/dX of the projection at the Gaussian's centre:
its footprint is J Sigma J^T, and the gradient that moves the centre goes through J.
  * Ours: the implicit-function derivative of the closed form, a 2 x 2 solve on terms the projection has already
    evaluated (splat.RefractiveSplatCamera).
  * AFRT returns a ray and no derivative.  It can be given one in two ways, both measured here:
    - central differences: six more AFRT runs per Gaussian, per view, per step, and a step size to choose;
    - autograd through its iterations, which keeps every iteration in memory.
On the trained Gaussians of the 12 test views, on the same GPU, this script measures:
  1. how close each J is to ours (ours is checked against central differences of project() to 1e-9 elsewhere);
  2. the time per view of each;
  3. the pixels that show an image AFRT does not find, and how the two trained models render them.

    python examples/demo_c_splatting/derivatives.py        # after demo_c.py; about 10 minutes
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parent), str(HERE)]
import refractive_camera as rc  # noqa: E402
from refractive_camera.scene import render_scene  # noqa: E402
import tube_garden as tg  # noqa: E402
import splat  # noqa: E402
import demo_c as D  # noqa: E402
from afrt_torch import AFRTCamera  # noqa: E402

KMAX = 300                          # AFRT iterations at most (the library's scalar version allows 2,000)
STEPS = (1e-4, 1e-5, 1e-6)          # finite-difference steps, metres
CHUNK = 4000                        # points per autograd chunk (memory)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(f, *a, repeat=3):
    best = np.inf
    out = None
    for _ in range(repeat):
        sync()
        t0 = time.perf_counter()
        out = f(*a)
        sync()
        best = min(best, time.perf_counter() - t0)
    return out, best


def ours(cam, means, dev):
    sc = splat.RefractiveSplatCamera(cam, tg.TUBE, dev)
    _, t_refresh = timed(sc.refresh, means, repeat=1)
    (g, uv, J, _, valid), t_step = timed(sc.project, means)
    return {"g": g, "uv": uv.detach(), "J": J.detach(), "valid": valid, "t_refresh": t_refresh, "t_step": t_step}


def afrt_fd(ac, means, h):
    uv0, done = ac.pixels(means, kmax=KMAX)
    cols = []
    for e in torch.eye(3, dtype=torch.float64, device=means.device):
        up, _ = ac.pixels(means + h * e, kmax=KMAX)
        um, _ = ac.pixels(means - h * e, kmax=KMAX)
        cols.append((up - um) / (2 * h))
    return uv0, done, torch.stack(cols, -1)


def afrt_autograd(ac, means):
    Js, uvs, dones = [], [], []
    for s in range(0, len(means), CHUNK):
        X = means[s:s + CHUNK].clone().requires_grad_(True)
        uv, done = ac.pixels(X, kmax=KMAX)
        rows = []
        for i in range(2):
            u = torch.nan_to_num(uv[:, i], nan=0.0)
            (gx,) = torch.autograd.grad(u.sum(), X, retain_graph=(i == 0))
            rows.append(gx)
        Js.append(torch.stack(rows, 1).detach())
        uvs.append(uv.detach())
        dones.append(done)
    return torch.cat(uvs), torch.cat(dones), torch.cat(Js)


def match(o, uv_a, done):
    """For each Gaussian AFRT converged on: the row of ours with the same image (nearest pixel, within 1e-3 px)."""
    g = o["g"].cpu().numpy()
    uv_o = o["uv"].cpu().numpy()
    ua = uv_a.cpu().numpy()
    ok = done.cpu().numpy() & np.isfinite(ua).all(1)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    lo, hi = np.searchsorted(gs, np.arange(len(ua))), np.searchsorted(gs, np.arange(len(ua)), side="right")
    pairs = []
    for i in np.flatnonzero(ok):
        rows = order[lo[i]:hi[i]]
        if not len(rows):
            continue
        d = np.linalg.norm(uv_o[rows] - ua[i], axis=1)
        k = int(np.argmin(d))
        if d[k] < 1e-3:
            pairs.append((i, rows[k]))
    return np.array(pairs, int).reshape(-1, 2)


def rel_err(Jm, Jr):
    return np.linalg.norm(Jm - Jr, axis=(1, 2)) / np.linalg.norm(Jr, axis=(1, 2))


def afrt_blind_pixels(cam, dev):
    """Pixels whose visible stone point AFRT reaches by another ray, or not at all: they show an image AFRT does not
    know.  Returns the mask (H, W) and the mask of pixels that see a stone in the water."""
    W, H = cam.size
    _, hits = render_scene(cam, tg.TUBE, tg.objects(), supersample=1, background=D.BG)
    see = (hits.obj >= 0) & hits.inner
    V, U = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    uv = np.c_[U.ravel(), V.ravel()]
    blind = np.zeros(W * H, bool)
    idx = np.flatnonzero(see)
    if len(idx):
        _, d_pix = cam.rays(uv[idx])
        ac = AFRTCamera(cam, tg.TUBE, dev)
        d_a, done = ac.directions(torch.tensor(hits.point[idx], device=dev), kmax=KMAX)
        d_a, done = d_a.cpu().numpy(), done.cpu().numpy()
        cosang = np.einsum("ij,ij->i", d_a, d_pix / np.linalg.norm(d_pix, axis=1, keepdims=True))
        blind[idx] = ~done | ~(cosang > np.cos(1e-5))
    return blind.reshape(H, W), see.reshape(H, W)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    _, test_cams, _ = D.views()
    sd = torch.load(D.CACHE / "gaussians_refractive.pt")
    means = sd["means"].to(dev).to(torch.float64)
    say(f"Demo C, part 2: derivatives on the {len(means)} trained Gaussians of ours, {len(test_cams)} test views, "
        f"device {dev}; AFRT capped at {KMAX} iterations")
    err = {f"fd_{h:g}": [] for h in STEPS}
    err["autograd"] = []
    big = []
    times = {k: [] for k in ("ours_refresh", "ours_step", "afrt", "afrt_fd", "afrt_autograd")}
    nan_fd, n_pairs, n_conv, n_imgs, iters = 0, 0, 0, 0, []
    n_outside, outside = 0, set()       # AFRT's images of points project() reports outside its model (not in the water)
    mem = 0
    for v, cam in enumerate(test_cams):
        o = ours(cam, means, dev)
        times["ours_refresh"].append(o["t_refresh"])
        times["ours_step"].append(o["t_step"])
        n_imgs += int(o["valid"].sum())
        ac = AFRTCamera(cam, tg.TUBE, dev)
        (uv_a, done), t_a = timed(ac.pixels, means, 1e-9, KMAX)
        iters.append(ac.iterations)
        times["afrt"].append(t_a)
        n_conv += int(done.sum())
        pairs = match(o, uv_a, done)
        n_pairs += len(pairs)
        un = np.setdiff1d(np.flatnonzero(done.cpu().numpy()), pairs[:, 0])
        if len(un):
            out = un[rc.project(means[un].cpu().numpy(), cam, tg.TUBE).status == "outside_model"]
            n_outside += len(out)
            outside.update(out.tolist())
        Jr = o["J"].cpu().numpy()[pairs[:, 1]]
        norms = np.linalg.norm(Jr, axis=(1, 2))
        big.append(norms > np.percentile(norms, 95))
        for h in STEPS:
            (u0, d0, Jf), t_fd = timed(afrt_fd, ac, means, h, repeat=1)
            if h == 1e-5:
                times["afrt_fd"].append(t_fd)
            Jm = Jf.cpu().numpy()[pairs[:, 0]]
            e = rel_err(Jm, Jr)
            if h == 1e-5:
                nan_fd += int((~np.isfinite(e)).sum())
            err[f"fd_{h:g}"].append(e)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        (u1, d1, Jg), t_ag = timed(afrt_autograd, ac, means, repeat=1)
        if torch.cuda.is_available():
            mem = max(mem, torch.cuda.max_memory_allocated() / 2 ** 20)
        times["afrt_autograd"].append(t_ag)
        err["autograd"].append(rel_err(Jg.cpu().numpy()[pairs[:, 0]], Jr))
        print(f"  view {v}: {len(pairs)} Gaussians compared", flush=True)
    big = np.concatenate(big)
    say(f"  ours: {n_imgs} images of Gaussians over the views; AFRT converged for {n_conv} Gaussian-views: {n_pairs} are "
        f"an image of ours (matched within 1e-3 px), {n_outside} are of {len(outside)} Gaussians that project() reports "
        f"as outside its model (not in the water), {n_conv - n_pairs - n_outside} are neither; slowest AFRT point per view "
        f"took {min(iters)}-{max(iters)} iterations")
    say("\n1. The Jacobian d(uv)/dX against ours (relative error, Frobenius norm):")
    summary = {"gaussians": int(len(means)), "views": len(test_cams), "pairs": int(n_pairs), "afrt_kmax": KMAX,
               "afrt_converged": int(n_conv), "afrt_outside_model": int(n_outside),
               "gaussians_outside_model": len(outside)}
    for k, v in err.items():
        e = np.concatenate(v)
        fin = np.isfinite(e)
        name = "autograd through AFRT's iterations" if k == "autograd" else f"AFRT, central differences, h = {k[3:]} m"
        say(f"  {name:42s}: median {np.nanmedian(e):.1e}, 99th pct {np.nanpercentile(e, 99):.1e}, max {np.nanmax(e):.1e}; "
            f"off by more than 1 %: {100 * np.mean(e[fin] > 0.01):.2f} %; next to a caustic (top 5 % of |J|): median "
            f"{np.nanmedian(e[big]):.1e}, 99th pct {np.nanpercentile(e[big], 99):.1e}; not finite {int((~fin).sum())}")
        summary[k] = {"median": float(np.nanmedian(e)), "p99": float(np.nanpercentile(e, 99)),
                      "max": float(np.nanmax(e)), "over_1pct": float(np.mean(e[fin] > 0.01)),
                      "caustic_median": float(np.nanmedian(e[big])),
                      "caustic_p99": float(np.nanpercentile(e[big], 99)), "not_finite": int((~fin).sum())}
    say("\n2. Time per view (all Gaussians, same GPU):")
    for k, label in (("ours_step", "ours, a training step: Newton from the last momenta + the analytic J"),
                     ("ours_refresh", "ours, a refresh: project() finds every image (C++, CPU, 8 threads)"),
                     ("afrt", "AFRT, pixels only"), ("afrt_fd", "AFRT + central differences (h = 1e-5 m)"),
                     ("afrt_autograd", "AFRT + autograd through its iterations")):
        say(f"  {label:72s}: {1e3 * np.median(times[k]):8.1f} ms")
        summary[f"time_{k}_ms"] = 1e3 * float(np.median(times[k]))
    say(f"  autograd's peak GPU memory, {CHUNK} Gaussians at a time: {mem:.0f} MB")
    summary["autograd_peak_mb"] = float(mem)

    # 3. pixels that show an image AFRT does not find
    gt = np.load(D.CACHE / "gt_test.npz")
    rend = {k: np.load(D.CACHE / f"render_test_{k}.npy").astype(np.float32) for k in ("refractive", "afrt")}
    masks = []
    for cam in test_cams:
        masks.append(afrt_blind_pixels(cam, dev))
    blind = np.array([m[0] for m in masks])
    see = np.array([m[1] for m in masks])
    np.save(D.CACHE / "afrt_blind_test.npy", blind)
    say(f"\n3. Pixels that show an image AFRT does not find: {int(blind.sum())} of the {int(see.sum())} pixels that see a "
        f"stone in the water ({100 * blind.sum() / see.sum():.2f} %), in {int(blind.any((1, 2)).sum())} of "
        f"{len(test_cams)} test views")
    for k, label in (("refractive", "ours"), ("afrt", "pinhole + AFRT (its own trained model)")):
        p_blind = D.psnr(rend[k], gt["rgb"], np.broadcast_to(blind[..., None], rend[k].shape))
        p_rest = D.psnr(rend[k], gt["rgb"], np.broadcast_to((see & ~blind)[..., None], rend[k].shape))
        say(f"  {label:40s}: PSNR on those pixels {p_blind:.1f} dB; on the other stone pixels {p_rest:.1f} dB")
        summary[f"blind_psnr_{k}"] = p_blind
        summary[f"rest_psnr_{k}"] = p_rest
    summary["blind_pixels"] = int(blind.sum())
    summary["stone_pixels"] = int(see.sum())
    rpath = HERE / "derivatives.json"
    rpath.write_text(json.dumps(summary, indent=1))
    (HERE / "derivatives_out.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
