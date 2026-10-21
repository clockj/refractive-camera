"""Figures of Demo A: a summary figure and an animation."""
import numpy as np

OURS, BASE, AFRT, TRUE = "#d55e00", "#0072b2", "#009e73", "#000000"


def _half(uv):
    """Full-resolution pixel -> half-resolution pixel (pixel centres at integers)."""
    return (np.asarray(uv, float) + 0.5) / 2 - 0.5


def _top_view(ax, window, cams, X, est_o, est_b):
    """The tube from above: walls, cameras with their fields of view, the dot's path and both estimates."""
    t = np.linspace(0, 2 * np.pi, 200)
    for R, a in zip(window.radii, (1.0, 0.6)):
        ax.plot(R * np.cos(t), R * np.sin(t), color="0.45", lw=1.0, alpha=a)
    for i, cam in enumerate(cams):
        c = cam.center
        W = cam.size[0]
        for u in (0.0, W - 1.0):                                # the edges of the field of view
            d = cam.rays(np.array([[u, (cam.size[1] - 1) / 2]]))[1][0]
            ax.plot([c[0], c[0] + 0.33 * d[0]], [c[1], c[1] + 0.33 * d[1]], color="0.7", lw=0.8)
        ax.plot(c[0], c[1], "s", color="0.2", ms=6)
        ax.text(c[0], c[1] + 0.03, f"cam {i}", fontsize=8, ha="center")
    ax.plot(X[:, 0], X[:, 1], color=TRUE, lw=1.0, label="true path")
    ax.plot(est_b[:, 0], est_b[:, 1], ".", color=BASE, ms=3, label="pinhole + distortion")
    ax.plot(est_o[:, 0], est_o[:, 1], ".", color=OURS, ms=3, label="refractive (ours)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.45, 0.47)
    ax.set_ylim(-0.16, 0.47)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7, frameon=False)


def _camera_panel(ax, cam, window, X, blobs, ours, afrt, title, margin=22, smallest=48):
    """A full-resolution rendering of the part of one camera's frame that holds every image of the dot (or, if it has
    none, where the dot would be without the window), with markers."""
    import refractive_camera as rc
    pts = np.array(list(ours) + list(blobs), float).reshape(-1, 2)
    if not len(pts):
        pts = cam.project_points(X)[0].reshape(1, 2)
    lo, hi = pts.min(0), pts.max(0)
    side = int(max(hi[0] - lo[0], hi[1] - lo[1], smallest) + 2 * margin)
    c = np.round(0.5 * (lo + hi)).astype(int)
    region = (c[0] - side // 2, c[1] - side // 2, c[0] - side // 2 + side, c[1] - side // 2 + side)
    img = rc.render_dots(cam, window, [X], 0.001, supersample=4, region=region, outside=0.55)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(region[0] - 0.5, region[2] - 0.5, region[3] - 0.5,
                                                        region[1] - 0.5))
    for q in blobs:
        ax.plot(q[0], q[1], "+", color="white", ms=8, mew=0.8)
    for q in ours:
        ax.plot(q[0], q[1], "o", mfc="none", mec=OURS, ms=15, mew=1.3)
    for q in afrt:
        ax.plot(q[0], q[1], "x", color=AFRT, ms=9, mew=1.3)
    ax.set_xlim(region[0] - 0.5, region[2] - 0.5)
    ax.set_ylim(region[3] - 0.5, region[1] - 0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _pick_frame(ours_uv, afrt_uv, blobs, gate=5.0):
    """A frame where a camera sees two images, each detected as its own blob (the widest apart first), else one where
    a camera sees none."""
    best = None
    for f, (row_o, row_a, row_b) in enumerate(zip(ours_uv, afrt_uv, blobs)):
        for c in range(3):
            P, B = np.array(row_o[c]).reshape(-1, 2), np.array(row_b[c]).reshape(-1, 2)
            clean = len(P) == 2 and len(B) == 2 and all(np.min(np.hypot(*(B - p).T)) < gate for p in P)
            sep = float(np.hypot(*(P[0] - P[1]))) if len(P) == 2 else 0.0
            score = (2 if clean else (1 if len(P) == 0 else 0), sep, -abs(f - len(ours_uv) // 2))
            if best is None or score > best[0]:
                best = (score, f)
    return best[1]


def summary_figure(path, window, cams, X, est_o, est_b, err_o, err_b, inside, frames8, blobs, ours_uv, afrt_uv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.5, 8.0), dpi=110)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.25, wspace=0.2)

    _top_view(fig.add_subplot(gs[0, 0]), window, cams, X, est_o, est_b)
    fig.axes[-1].set_title("(a) from above: tube, cameras, the dot's path", fontsize=10)

    ax = fig.add_subplot(gs[0, 1:])
    f = np.arange(len(X))
    out = ~np.asarray(inside)
    for k in np.flatnonzero(out):
        ax.axvspan(k - 0.5, k + 0.5, color="0.92", lw=0)
    ax.semilogy(f, err_b, color=BASE, lw=1.4, label="pinhole + distortion, calibrated inside the tube")
    ax.semilogy(f, np.maximum(err_o, 1e-6), color=OURS, lw=1.4, label="refractive model (ours)")
    lost = np.flatnonzero(~np.isfinite(err_b))
    if lost.size:
        ax.plot(lost, np.full(lost.size, ax.get_ylim()[1] * 0.7), "v", color=BASE, ms=4,
                label="pinhole: no estimate (its distortion model cannot be inverted there)")
    ax.set_xlabel("frame (the dot spirals out from the axis towards the wall, rising 100 mm)")
    ax.set_ylabel("3-D error (mm)")
    ax.set_xlim(-0.5, len(X) - 0.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(b) localisation error; grey: frames outside the baseline's calibration volume", fontsize=10)
    ax.grid(alpha=0.3, which="both", lw=0.4)

    fr = _pick_frame(ours_uv, afrt_uv, blobs)
    for c in range(3):
        n = len(ours_uv[fr][c])
        tag = {0: "no image (shadow)", 1: "one image", 2: "two images"}.get(n, f"{n} images")
        _camera_panel(fig.add_subplot(gs[1, c]), cams[c], window, X[fr], blobs[fr][c], ours_uv[fr][c],
                      afrt_uv[fr][c], f"({'cde'[c]}) camera {c}, frame {fr}: {tag}")
    fig.text(0.5, 0.03, "white +: blob detected in the rendered frame    orange o: project(), every image    "
                        "green x: AFRT, one start    (darker grey: rays that never reach the water)",
             ha="center", fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def animation(path, cams, X, err_o, err_b, frames8, blobs, ours_uv, afrt_uv, backdrops, max_frames=150):
    """Each camera's frame over its view of the empty tube, the images project() and AFRT predict, and the error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    step = max(1, int(np.ceil(len(X) / max_frames)))
    frames = list(range(0, len(X), step))
    fig = plt.figure(figsize=(12.0, 5.4), dpi=80)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.42], hspace=0.32, wspace=0.04)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_e = fig.add_subplot(gs[1, :])

    def draw(fr):
        for c, ax in enumerate(axes):
            ax.clear()
            img = np.minimum(backdrops[c], frames8[fr][c] / 255.0)
            H, W = img.shape
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
            for p in ours_uv[fr][c]:
                q = _half(p)
                ax.plot(q[0], q[1], "o", mfc="none", mec=OURS, ms=13, mew=1.4)
            for p in afrt_uv[fr][c]:
                q = _half(p)
                ax.plot(q[0], q[1], "s", mfc="none", mec=AFRT, ms=19, mew=1.2)
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            n = len(ours_uv[fr][c])
            ax.set_title(f"camera {c}: " + {0: "hidden by the wall", 1: "one image"}.get(n, f"{n} images"),
                         fontsize=10)
        ax_e.clear()
        ax_e.semilogy(err_b, color=BASE, lw=1.3, label="pinhole + distortion, calibrated in the tube")
        ax_e.semilogy(np.maximum(err_o, 1e-6), color=OURS, lw=1.3, label="refractive model (ours)")
        ax_e.axvline(fr, color="0.3", lw=0.8)
        ax_e.set_ylabel("3-D error (mm)", fontsize=8)
        ax_e.set_xlabel("frame", fontsize=8)
        ax_e.tick_params(labelsize=7)
        ax_e.legend(fontsize=7, loc="lower right")
        fig.suptitle(f"A dot spirals out inside a water-filled tube (frame {fr}).  Orange circles: project(), every "
                     "image.  Green squares: AFRT, one start.", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(path, writer=PillowWriter(fps=6))
    plt.close(fig)
