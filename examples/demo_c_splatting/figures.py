"""Figures of Demo C from the trained Gaussians in _cache/:
  * test_views.png: test views side by side;
  * metrics.png: PSNR and geometry per camera model;
  * geometry.png: where the opaque Gaussians sit, seen from above;
  * derivatives.png: the Jacobian AFRT would need, its error and cost, and the pixels AFRT cannot draw
    (from derivatives.py);
  * poses.png: training from a robot's poses, with and without pose refinement;
  * orbit.gif: a camera circling the tube, ground truth beside ours and the pinhole model.

    python examples/demo_c_splatting/figures.py [--no-orbit]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parent), str(HERE), str(HERE.parent / "demo_b_sfm")]
import refractive_camera as rc  # noqa: E402
from refractive_camera.scene import render_scene  # noqa: E402
import tube_garden as tg  # noqa: E402
import splat  # noqa: E402
import demo_c as D  # noqa: E402
import figstyle as fs  # noqa: E402

LABELS = {"refractive": "refraction-aware (ours)", "afrt": "AFRT's images + our Jacobian", "pinhole": "pinhole",
          "pinhole_dist": "pinhole + distortion"}
COLOURS = {k: fs.COLOR[k] for k in LABELS}
KEY = {k: k for k in LABELS}            # the results shown for each camera: the hybrid footprint's, once trained


def hybrid(key):
    return key.endswith("_hybrid")


def load(kind, dev):
    sd = torch.load(D.CACHE / f"gaussians_{kind}.pt")
    g = splat.Gaussians(sd["means"].to(dev), 0.001, torch.full((len(sd["means"]), 3), 0.5, device=dev))
    g.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    return g


@torch.no_grad()
def render_with(kind, g, cam, dev, fit=None, footprint="linear"):
    bg = torch.tensor(D.BG, device=dev, dtype=torch.float32)
    if kind in ("refractive", "afrt"):
        sc = (splat.RefractiveSplatCamera if kind == "refractive" else splat.AFRTSplatCamera)(cam, tg.TUBE, dev)
    else:
        sc = splat.PinholeSplatCamera(fit if fit is not None else cam, dev)
    sc.refresh(g.means)
    if footprint == "hybrid" and kind in ("refractive", "afrt"):
        img, *_ = splat.render_ut(g, sc, bg, above=6.0, min_points=2)
    else:
        img, *_ = splat.render(g, sc, bg)
    return img.clamp(0, 1).cpu().numpy()


def test_views(plt, res, kinds):
    gt = np.load(D.CACHE / "gt_test.npz")
    renders = {k: np.load(D.CACHE / f"render_test_{KEY[k]}.npy").astype(np.float32) for k in kinds}
    views = [0, 5, 9]
    fig, axes = plt.subplots(len(views), 1 + len(kinds), figsize=(2.9 * (1 + len(kinds)), 2.55 * len(views)))
    for r, v in enumerate(views):
        cols = [("ground truth (ray-traced)", gt["rgb"][v])] + [(LABELS[k], renders[k][v]) for k in kinds]
        for c, (title, img) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c:
                p = D.psnr(img, gt["rgb"][v])
                pw = D.psnr(img, gt["rgb"][v], gt["wall"][v]) if gt["wall"][v].any() else float("nan")
                ax.text(0.02, 0.03, f"{p:.1f} dB  (near the wall {pw:.1f})", transform=ax.transAxes, fontsize=7.5,
                        color="white", bbox=dict(facecolor="black", alpha=0.45, lw=0, pad=2))
            if r == 0:
                ax.set_title(title, fontsize=9, loc="center")
    fig.suptitle("Gaussian splatting through the water-filled tube: held-out views (PSNR against the ground truth)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(HERE / "test_views.png", dpi=110)
    plt.close(fig)


def metrics(plt, res, kinds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
    groups = [("between the\ntraining views", lambda k: res[KEY[k]]["psnr"]),
              ("between, next\nto the wall", lambda k: res[KEY[k]]["psnr_near_wall"]),
              ("from new\nposes", lambda k: res[KEY[k]]["far"]["psnr"]),
              ("new poses, next\nto the wall", lambda k: res[KEY[k]]["far"]["psnr_near_wall"])]
    w = 0.8 / (len(kinds) + 0.6)
    gx = np.arange(len(groups))
    for j, k in enumerate(kinds):
        bars = axes[0].bar(gx + (j - (len(kinds) - 1) / 2) * w * 1.1, [f(k) for _, f in groups], w,
                           color=COLOURS[k], label=LABELS[k])
        if k == "refractive":
            fs.bar_labels(axes[0], bars, lambda v: f"{v:.1f}", fontsize=7)
    axes[0].set_xticks(gx)
    axes[0].set_xticklabels([g for g, _ in groups])
    axes[0].set_ylabel("PSNR on held-out views (dB)")
    axes[0].set_ylim(0, 36)
    axes[0].grid(axis="x", visible=False)
    axes[0].legend(ncol=len(kinds), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    axes[0].set_title("Rendering", pad=24)
    x = np.arange(len(kinds))
    bars = axes[1].barh(x, [res[KEY[k]]["centre_to_surface_mm"]["median"] for k in kinds], 0.55,
                        color=[COLOURS[k] for k in kinds])
    fs.hbar_labels(axes[1], bars, lambda v: f"{v:.2f} mm")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([LABELS[k] for k in kinds])
    axes[1].invert_yaxis()
    axes[1].grid(axis="y", visible=False)
    axes[1].set_xlabel("median distance of opaque Gaussians to the true surface (mm)")
    axes[1].set_xlim(0, 1.25 * max(res[KEY[k]]["centre_to_surface_mm"]["median"] for k in kinds))
    axes[1].set_title("Geometry", pad=24)
    fig.tight_layout()
    fig.savefig(HERE / "metrics.png")
    plt.close(fig)


def geometry(plt, kinds):
    from demo_b import surface_distance
    objs = tg.objects()
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.9))
    t = np.linspace(0, 2 * np.pi, 300)
    sc = None
    for ax, k in zip(np.atleast_1d(axes), kinds):
        sd = torch.load(D.CACHE / f"gaussians_{KEY[k]}.pt")
        op = torch.sigmoid(sd["opacity_logits"]).numpy()
        m = sd["means"].numpy()[op > 0.3]
        m = m[np.isfinite(m).all(1)]
        dd = 1e3 * surface_distance(m.astype(float), objs)
        for R, a in zip(tg.TUBE.radii, (1.0, 0.5)):
            ax.plot(1e3 * R * np.cos(t), 1e3 * R * np.sin(t), color=fs.MUTED, lw=1.0, alpha=a)
        sc = ax.scatter(1e3 * m[:, 0], 1e3 * m[:, 1], c=np.clip(dd, 0, 10), s=0.6, cmap="viridis_r", vmin=0, vmax=10)
        ax.set_aspect("equal")
        ax.set_xlim(-110, 110)
        ax.set_ylim(-110, 110)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{LABELS[k]}\nmedian {np.median(dd):.2f} mm; {100 * np.mean(dd < 1):.0f} % within 1 mm",
                     fontsize=8.5, loc="center")
    cb = fig.colorbar(sc, ax=np.atleast_1d(axes).tolist(), shrink=0.75, pad=0.02)
    cb.set_label("distance to the true surface (mm)")
    cb.outline.set_visible(False)
    fig.suptitle("Opaque Gaussians seen from above, over the tube's wall", x=0.01, ha="left", fontsize=10,
                 fontweight="semibold")
    fig.savefig(HERE / "geometry.png")
    plt.close(fig)


def derivatives(plt):
    s = json.loads((HERE / "derivatives.json").read_text())
    blind = np.load(D.CACHE / "afrt_blind_test.npy")
    gt = np.load(D.CACHE / "gt_test.npz")
    rend = {k: np.load(D.CACHE / f"render_test_{k}.npy").astype(np.float32) for k in ("refractive", "afrt")}
    fig = plt.figure(figsize=(13, 6.6))
    gsp = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.5, wspace=0.75)
    ax = fig.add_subplot(gsp[0, :2])
    rows = [("autograd", "AFRT, autograd through\nits iterations"), ("fd_1e-06", "AFRT, differences,\nh = 1 um"),
            ("fd_1e-05", "AFRT, differences,\nh = 10 um"), ("fd_0.0001", "AFRT, differences,\nh = 100 um")]
    y = np.arange(len(rows))
    for i, (k, label) in enumerate(rows):
        ax.plot([s[k]["median"], s[k]["p99"]], [i, i], color=fs.COLOR["afrt"], lw=2, solid_capstyle="round")
        ax.scatter([s[k]["median"]], [i], s=46, color=fs.COLOR["afrt"], edgecolor=fs.SURFACE, linewidth=1.5,
                   zorder=3)
        ax.scatter([s[k]["p99"]], [i], s=46, facecolor=fs.SURFACE, edgecolor=fs.COLOR["afrt"], linewidth=1.6,
                   zorder=3)
        ax.annotate(f"99 %: {s[k]['p99']:.0e}\nnext to a caustic: {s[k]['caustic_p99']:.0e}",
                    (s[k]["p99"], i), xytext=(9, 0), textcoords="offset points", va="center", fontsize=7.5,
                    color=fs.INK2)
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 1e0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("relative error against ours (dot: median; ring: 99th percentile)")
    ax.set_title(f"1. AFRT's Jacobian d(uv)/dX, estimated two ways ({s['pairs']:,} Gaussian-views)")

    ax = fig.add_subplot(gsp[0, 2:])
    items = [("time_ours_step_ms", "ours: a training step\n(Newton + analytic J)", fs.COLOR["refractive"]),
             ("time_afrt_ms", "AFRT: pixels only", fs.COLOR["afrt"]),
             ("time_afrt_fd_ms", "AFRT + differences", fs.COLOR["afrt"]),
             ("time_afrt_autograd_ms", "AFRT + autograd", fs.COLOR["afrt"])]
    vals = [s[k] for k, _, _ in items]
    bars = ax.barh(np.arange(len(items)), vals, 0.55, color=[c for _, _, c in items])
    fs.hbar_labels(ax, bars, lambda v: f"{v:,.0f} ms")
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels([lab for _, lab, _ in items])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, 30 * max(vals))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"ms per view, {s['gaussians']:,} Gaussians, same GPU (log scale)")
    ax.set_title("2. Time per view")

    # 3. crops where AFRT is blind
    per_view = blind.reshape(len(blind), -1).sum(1)
    order = np.argsort(-per_view)[:2]
    for j, v in enumerate(order):
        ys, xs = np.nonzero(blind[v])
        cy, cx = int(np.median(ys)), int(np.median(xs))
        h, w = 64, 80
        y0 = int(np.clip(cy - h // 2, 0, blind.shape[1] - h))
        x0 = int(np.clip(cx - w // 2, 0, blind.shape[2] - w))
        sub = gsp[1, 2 * j:2 * j + 2].subgridspec(1, 3, wspace=0.04)
        for i, (img, title) in enumerate(((gt["rgb"][v], "ground truth"), (rend["refractive"][v], "ours"),
                                          (rend["afrt"][v], "pinhole + AFRT"))):
            a = fig.add_subplot(sub[0, i])
            crop = np.clip(img[y0:y0 + h, x0:x0 + w], 0, 1)
            a.imshow(crop, interpolation="nearest")
            m = blind[v, y0:y0 + h, x0:x0 + w]
            a.contour(m.astype(float), levels=[0.5], colors=["#e87ba4"], linewidths=1.0)
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(False)
            a.set_title(title, fontsize=8, loc="center")
    fig.text(0.01, 0.47, f"3. Pixels that show an image AFRT does not find (pink outline): "
             f"{s['blind_pixels']:,} of {s['stone_pixels']:,} stone pixels; PSNR there: ours "
             f"{s['blind_psnr_refractive']:.1f} dB, pinhole + AFRT {s['blind_psnr_afrt']:.1f} dB",
             fontsize=10, fontweight="semibold")
    fig.savefig(HERE / "derivatives.png")
    plt.close(fig)


def poses(plt, res):
    have = [k for k in ("refractive", "refractive_noisy", "refractive_refined") if k in res]
    if len(have) < 3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    rep = res["refractive_refined"]["poses_reported"]
    names = {"refractive": "true poses",
             "refractive_noisy": f"a robot's poses ({rep['centre_mm_rms']:.1f} mm, {rep['rotation_deg_rms']:.2f} deg off)",
             "refractive_refined": "a robot's poses, refined"}
    cols = {"refractive": fs.MUTED, "refractive_noisy": fs.COLOR["refractive"],
            "refractive_refined": fs.COLOR["refractive_plus"]}
    groups = [("between the\ntraining views", lambda k: res[k]["psnr"]),
              ("from new\nposes", lambda k: res[k]["far"]["psnr"]),
              ("next to\nthe wall", lambda k: res[k]["psnr_near_wall"])]
    w = 0.24
    for j, k in enumerate(have):
        bars = axes[0].bar(np.arange(len(groups)) + (j - 1) * (w + 0.03), [f(k) for _, f in groups], w,
                           color=cols[k], label=names[k].replace("\n", " "))
        fs.bar_labels(axes[0], bars, lambda v: f"{v:.1f}", fontsize=7)
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels([g for g, _ in groups])
    axes[0].set_ylim(0, 36)
    axes[0].set_ylabel("PSNR on held-out views (dB)")
    axes[0].grid(axis="x", visible=False)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    axes[0].set_title("Training from a robot's poses", pad=26)
    ax = axes[1]
    before, after = res["refractive_refined"]["poses_reported"], res["refractive_refined"]["poses_refined"]
    labels = ["camera centres (mm rms)", "rotations (0.1 deg rms)"]
    b = [before["centre_mm_rms"], 10 * before["rotation_deg_rms"]]
    a = [after["centre_mm_rms"], 10 * after["rotation_deg_rms"]]
    y = np.arange(2)
    b1 = ax.barh(y - 0.17, b, 0.3, color=fs.COLOR["refractive"], label="as reported")
    b2 = ax.barh(y + 0.17, a, 0.3, color=fs.COLOR["refractive_plus"], label="refined with the Gaussians")
    fs.hbar_labels(ax, b1, lambda v: f"{v:.2f}")
    fs.hbar_labels(ax, b2, lambda v: f"{v:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.3 * max(b + a))
    ax.legend(loc="lower right")
    ax.set_title("Pose error of the 48 training cameras", pad=26)
    fig.tight_layout()
    fig.savefig(HERE / "poses.png")
    plt.close(fig)


def without_tube(plt, res, kinds):
    """Each model's Gaussians, trained on views through the tube, drawn by a plain camera with the tube taken away,
    beside the stones ray-traced with no tube (every index 1).  Returns the mean PSNR per model."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    air = rc.CylindricalWindow(radii=tg.TUBE.radii, indices=(1.0, 1.0, 1.0))
    _, test_cams, _ = D.views()
    show = [k for k in ("refractive", "pinhole_dist", "pinhole") if k in kinds]
    gs = {k: load(KEY[k], dev) for k in show}
    gt = np.load(D.CACHE / "gt_test.npz")["rgb"]
    views = [0, 5, 9]
    scores = {k: [] for k in show}
    fig, axes = plt.subplots(len(views), 2 + len(show), figsize=(2.75 * (2 + len(show)), 2.3 * len(views)))
    for r, v in enumerate(views):
        cam = test_cams[v]
        truth = render_scene(cam, air, tg.objects(), supersample=3, background=D.BG)[0].astype(np.float32)
        cols = [("seen through the tube (truth)", gt[v]), ("the stones without the tube (truth)", truth)]
        for k in show:
            with torch.no_grad():
                img, *_ = splat.render(gs[k], splat.PinholeSplatCamera(cam, dev),
                                       torch.tensor(D.BG, device=dev, dtype=torch.float32))
            img = img.clamp(0, 1).cpu().numpy()
            scores[k].append(D.psnr(img, truth))
            cols.append((f"{LABELS[k]}, tube removed", img))
        for c, (title, im) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(np.clip(im, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=8.5, loc="center")
            if c >= 2:
                ax.text(0.02, 0.03, f"{scores[show[c - 2]][-1]:.1f} dB", transform=ax.transAxes, fontsize=7.5,
                        color="white", bbox=dict(facecolor="black", alpha=0.45, lw=0, pad=2))
    fig.suptitle("Trained on views through the tube, drawn by a plain camera with the tube taken away "
                 "(PSNR against the stones without the tube)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(HERE / "without_tube.png", dpi=110)
    plt.close(fig)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def orbit(kinds):
    from PIL import Image, ImageDraw, ImageFont
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orbit_cams = tg.ring_cameras(n_per_ring=60, rings=((0.01, 0.38),), f=D.FOCAL, size=D.SIZE, seed=9, jitter=0.0)
    show = [k for k in ("refractive", "pinhole_dist") if k in kinds]
    gs = {k: load(KEY[k], dev) for k in show}
    rng = np.random.default_rng(5)
    try:
        font = ImageFont.truetype("segoeuib.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    W, H = D.SIZE
    S = 2                                                     # upscale for the GIF
    frames = []
    for i, cam in enumerate(orbit_cams):
        row = [render_scene(cam, tg.TUBE, tg.objects(), supersample=2, background=D.BG)[0]]
        names = [("ground truth (ray-traced)", None)]
        for k in show:
            fit = D.calibrate_through_tube(cam, rng)[0] if k == "pinhole_dist" else None
            row.append(render_with("refractive" if k == "refractive" else "pinhole", gs[k], cam, dev, fit,
                                   "hybrid" if hybrid(KEY[k]) else "linear"))
            names.append((LABELS[k], fs.COLOR[k]))
        canvas = Image.new("RGB", (len(row) * W * S + (len(row) - 1) * 6, H * S + 34), (26, 26, 25))
        dr = ImageDraw.Draw(canvas)
        for j, (img, (name, col)) in enumerate(zip(row, names)):
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W * S, H * S), Image.LANCZOS)
            x0 = j * (W * S + 6)
            canvas.paste(im, (x0, 34))
            if col is not None:
                dr.rectangle([x0 + 8, 11, x0 + 20, 23], fill=col)
            dr.text((x0 + (28 if col else 8), 7), name, font=font, fill=(240, 239, 236))
        frames.append(canvas.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        print(f"orbit frame {i + 1}/{len(orbit_cams)}", end="\r", flush=True)
    frames[0].save(HERE / "orbit.gif", save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    print("\nwrote orbit.gif")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.use()
    res = json.loads((HERE / "results.json").read_text())
    for k in ("refractive", "afrt"):
        if k + "_hybrid" in res and "--linear" not in sys.argv:
            KEY[k] = k + "_hybrid"
    kinds = [k for k in ("refractive", "afrt", "pinhole", "pinhole_dist") if KEY[k] in res]
    print("showing", {k: KEY[k] for k in kinds})
    test_views(plt, res, kinds)
    metrics(plt, res, kinds)
    geometry(plt, kinds)
    print("wrote test_views.png, metrics.png, geometry.png")
    if (HERE / "derivatives.json").exists():
        derivatives(plt)
        print("wrote derivatives.png")
    poses(plt, res)
    wt = without_tube(plt, res, kinds)
    (HERE / "without_tube.json").write_text(json.dumps({"psnr_db": wt, "views": [0, 5, 9]}, indent=1))
    print("wrote without_tube.png:", {k: round(v, 2) for k, v in wt.items()})
    if "--no-orbit" not in sys.argv:
        orbit(kinds)


if __name__ == "__main__":
    main()
