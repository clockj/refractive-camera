"""The summary figure of Demo B, from results.json and the cached results of demo_b.py."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parent), str(HERE)]
import tube_garden as tg  # noqa: E402
import figstyle as fs  # noqa: E402
from refractive_camera.scene import Box, Cylinder, Disk, Sphere  # noqa: E402

OURS, BASE, AFRT = fs.COLOR["refractive"], fs.COLOR["pinhole_dist"], fs.COLOR["afrt"]
CACHE = (Path(os.environ["REFCAM_CACHE"]) / "demo_b_sfm" if "REFCAM_CACHE" in os.environ else HERE / "_cache") / "ring20"


def surface_samples(objs, n=4000, seed=0):
    """Points spread over the scene's surfaces, for a backdrop."""
    rng = np.random.default_rng(seed)
    out = []
    for ob in objs:
        if isinstance(ob, Sphere):
            v = rng.normal(size=(n // 4, 3))
            p = ob.radius * v / np.linalg.norm(v, axis=1, keepdims=True)
        elif isinstance(ob, Box):
            p = rng.uniform(-1, 1, (n // 4, 3)) * ob.half
            ax = rng.integers(0, 3, len(p))
            p[np.arange(len(p)), ax] = np.sign(rng.uniform(-1, 1, len(p))) * ob.half[ax]
        elif isinstance(ob, Cylinder):
            a = rng.uniform(0, 2 * np.pi, n // 4)
            p = np.c_[ob.radius * np.cos(a), ob.radius * np.sin(a), rng.uniform(-1, 1, len(a)) * ob.half_height]
        elif isinstance(ob, Disk):
            r = ob.radius * np.sqrt(rng.uniform(0, 1, n))
            a = rng.uniform(0, 2 * np.pi, n)
            p = np.c_[r * np.cos(a), r * np.sin(a), np.zeros(n)]
        out.append(p @ ob.rotation.T + ob.center)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fs.use()
    res = json.loads((HERE / "results.json").read_text())
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.32, wspace=0.45)

    # (a) one view, with the observations of its tracked points
    ax = fig.add_subplot(gs[0, 0])
    img = np.asarray(Image.open(CACHE / "view_00.png"))
    ax.imshow(img)
    z = np.load(CACHE / "result_refractive.npz")
    uv = z["obs_uv"][z["obs_view"] == 0]
    ax.plot(uv[:, 0], uv[:, 1], ".", color="white", ms=2.2, alpha=0.9)
    ax.set_title("One of 60 views through the tube, with its tracked features", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # (b), (c): the reconstructions from above, over the true scene
    objs = tg.objects()
    back = np.vstack([s for s, ob in zip(surface_samples(objs), objs) if not isinstance(ob, Disk)])
    t = np.linspace(0, 2 * np.pi, 300)
    for k, (name, col, label) in enumerate([("refractive", OURS, "refractive (ours)"),
                                            ("pinhole", BASE, "pinhole + distortion")]):
        ax = fig.add_subplot(gs[0, 1 + k])
        z = np.load(CACHE / f"result_{name}.npz")
        X = z["X"][z["have"]]
        ax.plot(back[:, 0] * 1e3, back[:, 1] * 1e3, ".", color=fs.GRID, ms=1)
        for R, a in zip(tg.TUBE.radii, (1.0, 0.5)):
            ax.plot(1e3 * R * np.cos(t), 1e3 * R * np.sin(t), color=fs.MUTED, lw=1.0, alpha=a)
        ax.plot(X[:, 0] * 1e3, X[:, 1] * 1e3, ".", color=col, ms=1.6)
        e = res[name]["similarity"]["point_err_mm"]
        ax.grid(False)
        ax.set_title(f"{label}: {len(X)} points\nmedian error {e['median']:.2f} mm, "
                     f"95th percentile {e['p95']:.1f} mm", fontsize=9)
        ax.set_aspect("equal")
        lim = 145 if name == "pinhole" else 110
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("x (mm)")
        if k == 0:
            ax.set_ylabel("y (mm), seen from above; grey: the true stones")

    # (d) error against distance from the axis
    ax = fig.add_subplot(gs[1, 0])
    bins = ["0-50", "50-70", "70-85", "85-94"]
    centres = [25, 60, 77.5, 89.5]
    for name, col, label, mk in [("refractive", OURS, "refractive (ours)", "o"), ("afrt", AFRT, "pinhole + AFRT", "x"),
                                 ("pinhole", BASE, "pinhole + distortion", "s")]:
        b = res[name]["similarity"]["point_err_by_r_mm"]
        med = [b[k]["median"] for k in bins]
        p95 = [b[k]["p95"] for k in bins]
        ax.plot(centres, med, "o-", color=col, label=label, ms=6, mec=fs.SURFACE, mew=1.2)
        ax.plot(centres, p95, "o-", color=col, alpha=0.35, ms=4, lw=1.2, mec=fs.SURFACE, mew=1.0)
    air = res.get("colmap_air", {}).get("surface_dist_mm", {}).get("median")
    if air:
        ax.axhline(air, color=fs.BASE, lw=1.0)
        ours_sd = res["refractive"]["similarity"]["surface_dist_mm"]["median"]
        ax.text(3, air * 1.15, f"COLMAP, no tube: points {air:.2f} mm from the surface (ours, through it: "
                f"{ours_sd:.2f} mm)", fontsize=6.5, color=fs.INK2)
    ax.axvspan(94, 100, color="#f0efec", lw=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0, 100)
    ax.set_xlabel("distance of the point from the tube's axis (mm); wall at 94 mm")
    ax.set_ylabel("3-D error after similarity alignment (mm)")
    ax.set_title("Error near the wall (strong: median; faint: 95th percentile)", fontsize=9)
    ax.legend(loc="center left")

    # pose errors: centres and rotations, each on its own axis
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.9)
    labels = ["robot prior", "refractive (ours)", "pinhole + AFRT", "pinhole + distortion"]
    cen = [res["setup"]["prior_centre_err_mm"]["median"], res["refractive"]["absolute"]["centre_err_mm"]["median"],
           res["afrt"]["absolute"]["centre_err_mm"]["median"], res["pinhole"]["absolute"]["centre_err_mm"]["median"]]
    rot = [res["setup"]["prior_rot_err_deg"]["median"], res["refractive"]["absolute"]["rot_err_deg"]["median"],
           res["afrt"]["absolute"]["rot_err_deg"]["median"], res["pinhole"]["absolute"]["rot_err_deg"]["median"]]
    cols = [fs.MUTED, OURS, AFRT, BASE]
    y = np.arange(4)
    for i, (vals, unit, title) in enumerate(((cen, "mm", "Camera centres: median error"),
                                            (rot, "deg", "Camera rotations: median error"))):
        ax = fig.add_subplot(sub[i, 0])
        bars = ax.barh(y, vals, 0.6, color=cols)
        fs.hbar_labels(ax, bars, (lambda v: f"{v:.2f} mm") if unit == "mm" else (lambda v: f"{v:.3f} deg"),
                       fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.35 * max(vals))
        ax.set_title(title, fontsize=9)

    # (f) COLMAP
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    c, a = res.get("colmap", {}), res.get("colmap_air", {})
    lines = ["COLMAP's own pipeline (pycolmap, OPENCV camera)", "",
             "same 60 images, through the tube:",
             f"   {c.get('registered', '?')} of 60 views registered",
             f"   ({c.get('verified_pairs', {}).get('pairs', '?')} image pairs verified)", "",
             "same scene and cameras, no tube (control):",
             f"   {a.get('registered', '?')} of 60 views registered",
             f"   camera centres {a.get('centre_err_mm', {}).get('median', float('nan')):.2f} mm,"
             f" points {a.get('surface_dist_mm', {}).get('median', float('nan')):.2f} mm",
             f"   from the truth (medians, after a similarity)"]
    ax.text(0.0, 0.95, "\n".join(lines), fontsize=8.5, va="top", color=fs.INK2)
    fig.savefig(HERE / "summary.png")
    print("wrote", HERE / "summary.png")


if __name__ == "__main__":
    main()
