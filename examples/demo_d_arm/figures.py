"""The figures of Demo D: summary.png (the catches) and fold_zone.png (tracking where the wall hides or doubles the
ball; the whole-ball fit; one camera alone), from results.json and _cache/fold_zone_runs.json."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE), str(HERE.parent)]
import arm  # noqa: E402
import figstyle as fs  # noqa: E402

KINDS = ("refractive", "afrt", "pinhole")
NAME = {"refractive": "refraction-aware (ours)", "afrt": "pinhole + AFRT", "pinhole": "pinhole + distortion"}
COL = {"refractive": fs.COLOR["refractive"], "afrt": fs.COLOR["afrt"], "pinhole": fs.COLOR["pinhole_dist"]}
TOL = np.array([arm.FINGER[0] + 0.002, arm.OPEN - arm.BALL_R, arm.FINGER[2] - arm.BALL_R]) * 1e3   # mm: across,
# along the closing axis, along the tool


def summary(plt):
    import matplotlib
    res = json.loads((HERE / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    groups = (("open water\n40-120 mm from the axis", False), ("near the wall\n136-160 mm", True))
    w = 0.2
    for j, k in enumerate(KINDS):
        tr = res[k]["trials"]
        vals = [sum(t["caught"] for t in tr if t["near_wall"] == near) for _, near in groups]
        n = [sum(1 for t in tr if t["near_wall"] == near) for _, near in groups]
        bars = ax.bar(np.arange(2) + (j - 1) * (w + 0.04), vals, w, color=COL[k], label=NAME[k])
        for b, v, nn in zip(bars, vals, n):
            ax.annotate(f"{v}/{nn}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=fs.INK2)
    ax.set_xticks(range(2))
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylim(0, 23)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("balls caught, of 20 paths")
    ax.grid(axis="x", visible=False)
    ax.set_title("Catches with the slim gripper")

    ax = axes[1]
    ax.axvspan(1e3 * arm.R_OUT / 1.33, 1e3 * arm.R_IN, color="#f0efec", lw=0)
    ax.text(1e3 * arm.R_OUT / 1.33 + 2, 7.2, "the wall shows a point\ntwice or hides it", fontsize=7.5,
            color=fs.INK2, va="top")
    ax.axhline(1.0, color=fs.BASE, lw=1.0)
    ax.text(3, 1.08, "the fingers' reach: the ball is held below this line", fontsize=7.5, color=fs.INK2)
    for k in KINDS:
        tr = [t for t in res[k]["trials"] if t["offset_mm"] is not None]
        r = np.array([t["r_mm"] for t in tr])
        frac = np.array([np.max(np.abs(t["offset_mm"]) / TOL) for t in tr])
        ok = np.array([t["caught"] for t in tr])
        ax.scatter(r[ok], frac[ok], s=34, color=COL[k], edgecolor=fs.SURFACE, linewidth=1.2, zorder=3,
                   label=f"{NAME[k]}: {ok.sum()} of {len(ok)} caught")
        ax.scatter(r[~ok], frac[~ok], s=42, facecolor="none", edgecolor=COL[k], linewidth=1.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 8)
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0, 1e3 * arm.R_IN)
    ax.set_xlabel("the ball's distance from the tank's axis as the fingers close (mm); the wall is at 172 mm")
    ax.set_ylabel("the ball's offset from the grasp point,\nrelative to the fingers' reach")
    ax.legend(loc="upper left")
    ax.set_title("Where each ball was as the fingers closed (filled: caught; hollow: missed)")
    fig.tight_layout()
    fig.savefig(HERE / "summary.png")
    plt.close(fig)


def fold_zone(plt):
    runs = json.loads((arm.CACHE / "fold_zone_runs.json").read_text())
    summ = json.loads((HERE / "fold_zone.json").read_text())
    kinds = ("refractive", "refractive_ball", "afrt", "pinhole")
    name = {"refractive": "ours, blob centroids", "refractive_ball": "ours, whole-ball fit",
            "afrt": "pinhole + AFRT", "pinhole": "pinhole + distortion"}
    col = {"refractive": fs.COLOR["refractive"], "refractive_ball": fs.COLOR["refractive_plus"],
           "afrt": fs.COLOR["afrt"], "pinhole": fs.COLOR["pinhole_dist"]}
    one_col = "#e87ba4"
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 2.4], hspace=0.12, wspace=0.22)
    seed = 0
    ref = next(r for r in runs if r["kind"] == "refractive" and r["seed"] == seed)
    t = np.array([x["t"] for x in ref["rows"]])
    ti = np.array([x["true_images"] for x in ref["rows"]])
    ax0 = fig.add_subplot(gs[0, 0])
    for c in range(ti.shape[1]):
        for nimg, key in ((0, "none"), (1, "one"), (2, "two")):
            m = ti[:, c] == nimg
            ax0.scatter(t[m], np.full(m.sum(), c), marker="s", s=9, color=fs.SEEN[key], linewidth=0)
    ax0.set_yticks(range(ti.shape[1]))
    ax0.set_yticklabels([f"camera {c}" for c in range(ti.shape[1])])
    ax0.set_ylim(-0.6, ti.shape[1] - 0.4)
    ax0.grid(False)
    ax0.tick_params(labelbottom=False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.set_title("One path: what each camera shows (dark: hidden; grey: once; pink: twice)", fontsize=9)
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    for k in kinds:
        r = next(x for x in runs if x["kind"] == k and x["seed"] == seed)
        e = np.array([x["meas_err_mm"] if x["meas_err_mm"] is not None else np.nan for x in r["rows"]])
        ax1.plot(t, e, color=col[k], lw=1.4, label=name[k])
    ax1.set_yscale("log")
    ax1.set_ylabel("error of the 3-D measurement (mm)")
    ax1.set_xlabel("time (s)")
    ax1.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    ax1.set_ylim(0.005, 200)

    ax2 = fig.add_subplot(gs[:, 1])
    for k in kinds:
        e = np.array([x["meas_err_mm"] for r in runs if r["kind"] == k for x in r["rows"]
                      if x["meas_err_mm"] is not None and x["t"] >= 1.0])
        e = np.sort(e)
        ax2.plot(e, np.arange(1, len(e) + 1) / len(e), color=col[k], lw=2,
                 label=f"{name[k]}: median {np.median(e):.2f} mm")
    one = np.array([o[2] for r in runs if r["kind"] == "refractive_ball" for x in r["rows"]
                    for o in x["one_camera"] if o[2] is not None])
    if len(one):
        one = np.sort(one)
        ax2.plot(one, np.arange(1, len(one) + 1) / len(one), color=one_col, lw=2,
                 label=f"ours, ONE camera where it sees the ball twice: median {np.median(one):.2f} mm")
    ax2.set_xscale("log")
    ax2.set_xlim(0.003, 300)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("error of the 3-D measurement (mm)")
    ax2.set_ylabel("fraction of frames with a smaller error")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), fontsize=7.5, ncol=1)
    ax2.set_title("All 20 paths (16 s each, 140-162 mm from the axis)", fontsize=9)
    fig.savefig(HERE / "fold_zone.png")
    plt.close(fig)


def visibility(plt):
    """What each camera sees of a horizontal slice through the tank at the cameras' height: once, twice, or not at
    all, and how many cameras see each point."""
    import refractive_camera as rc
    from matplotlib.colors import ListedColormap
    import demo_d as D
    cams = D.cameras()
    n = 401
    g = np.linspace(-arm.R_IN, arm.R_IN, n)
    X, Y = np.meshgrid(g, g)
    inside = np.hypot(X, Y) < arm.R_IN - 0.0005
    P = np.c_[X[inside] + arm.TANK_XY[0], Y[inside] + arm.TANK_XY[1], np.full(inside.sum(), D.CAM_Z)]
    counts, seen_by = [], np.zeros(inside.sum(), int)
    for c in cams:
        im = rc.project(P, c, D.TANK, threads=8)
        k = np.bincount(im.point[im.in_image], minlength=len(P))
        counts.append(k)
        seen_by += k > 0
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    t = np.linspace(0, 2 * np.pi, 400)
    cmap = ListedColormap([fs.SEEN["none"], "#e9e8e3", fs.SEEN["two"]])
    for i, ax in enumerate(axes):
        img = np.full(X.shape, np.nan)
        if i < 3:
            img[inside] = np.minimum(counts[i], 2)
            ax.imshow(img, origin="lower", extent=[-1e3 * arm.R_IN, 1e3 * arm.R_IN] * 2, cmap=cmap, vmin=-0.5,
                      vmax=2.5, interpolation="nearest")
            c = cams[i].center - np.r_[arm.TANK_XY, 0]
            d = c[:2] / np.linalg.norm(c[:2])
            ax.annotate("", xy=1e3 * 0.186 * d, xytext=1e3 * 0.226 * d,
                        arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=1.4))
            ax.set_title(f"camera {i} (arrow: where it looks from)", fontsize=9, loc="center")
            share = [np.mean(np.minimum(counts[i], 2) == v) for v in (0, 1, 2)]
            ax.text(0.5, -0.09, f"hidden {100 * share[0]:.0f} %   once {100 * share[1]:.0f} %   twice "
                    f"{100 * share[2]:.0f} %", transform=ax.transAxes, ha="center", fontsize=7.5, color=fs.INK2)
        else:
            img[inside] = seen_by
            cm4 = ListedColormap([fs.SEEN["none"], "#f2c6b4", "#eb9670", fs.COLOR["refractive"]])
            ax.imshow(img, origin="lower", extent=[-1e3 * arm.R_IN, 1e3 * arm.R_IN] * 2, cmap=cm4, vmin=-0.5,
                      vmax=3.5, interpolation="nearest")
            ax.set_title("how many of the three cameras see the point", fontsize=9, loc="center")
            share = [np.mean(seen_by == v) for v in range(4)]
            ax.text(0.5, -0.09, "   ".join(f"{v}: {100 * share[v]:.1f} %" for v in range(4)), transform=ax.transAxes,
                    ha="center", fontsize=7.5, color=fs.INK2)
        for R, a in ((arm.R_OUT, 1.0), (arm.R_IN, 0.6), (arm.R_OUT / 1.33, 0.9)):
            ax.plot(1e3 * R * np.cos(t), 1e3 * R * np.sin(t), color=fs.MUTED if R != arm.R_OUT / 1.33 else fs.INK2,
                    lw=0.9 if R != arm.R_OUT / 1.33 else 0.8, ls="-", alpha=a)
        ax.set_xlim(-232, 232)
        ax.set_ylim(-232, 232)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("A slice through the tank at the cameras' height (wall 172-180 mm; inner circle: 135 mm, where "
                 "double images and the shadow begin) - dark: hidden, light: seen once, pink: seen twice",
                 x=0.01, ha="left", fontsize=9.5, fontweight="semibold")
    fig.savefig(HERE / "visibility.png")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.use()
    summary(plt)
    visibility(plt)
    if (arm.CACHE / "fold_zone_runs.json").exists():
        fold_zone(plt)
    print("wrote summary.png" + (" and fold_zone.png" if (arm.CACHE / "fold_zone_runs.json").exists() else ""))


if __name__ == "__main__":
    main()
