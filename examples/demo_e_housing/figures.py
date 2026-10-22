"""Demo E's figures, from results.json: summary.png, views.png and docking.gif.

    python examples/demo_e_housing/figures.py                (about 15 minutes; renders are cached in _cache/)
    python examples/demo_e_housing/figures.py --only summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE), str(HERE.parent)]
import refractive_camera as rc  # noqa: E402
import calibrate as cb  # noqa: E402
import demo_e  # noqa: E402
import dock_scene as ds  # noqa: E402
import figstyle  # noqa: E402
import housing as hz  # noqa: E402
from figstyle import COLOR  # noqa: E402

ORDER = ("flat", "dome", "tube")
TITLE = {"flat": "Flat port", "dome": "Dome port", "tube": "Tube housing"}
NAME = {"pinhole": "pinhole + distortion, calibrated in water", "refractive": "refractive model (ours)"}
PIN, OURS = COLOR["pinhole_dist"], COLOR["refractive"]


def results():
    return json.loads((HERE / "results.json").read_text())


# ---- summary.png ----------------------------------------------------------------------------------------------------
def summary(res):
    import matplotlib.pyplot as plt
    figstyle.use()
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), sharey="row")
    for j, name in enumerate(ORDER):
        r = res[name]
        ax = axes[0, j]
        ranges = np.array([float(k) for k in r["where"]["noisy"]])
        for model, col in (("pinhole", PIN), ("refractive", OURS)):
            w = r["where"]["noisy"]
            med = np.array([w[k][model]["median"] for k in w])
            top = np.array([w[k][model]["max"] for k in w])
            ax.fill_between(ranges, med, top, color=col, alpha=0.13, lw=0)
            ax.plot(ranges, med, "-", color=col, marker="o", ms=3.5)
        w = r["where"]["exact"]
        ax.plot(ranges, [max(w[k]["pinhole"]["median"], 1e-3) for k in w], "--", color=PIN, lw=1.6)
        ours_exact = max(w[k]["refractive"]["max"] for k in w)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 300)
        ax.set_xticks([0.15, 0.25, 0.4, 0.6, 1, 1.5, 2.5, 4])
        ax.set_xticklabels(["0.15", "0.25", "0.4", "0.6", "1", "1.5", "2.5", "4"])
        ax.minorticks_off()
        ax.set_title(f"{TITLE[name]}: where points appear")
        ax.set_xlabel("range from the camera (m)")
        bound = 10.0 ** np.ceil(np.log10(max(ours_exact, 1e-12)))
        ax.text(0.97, 0.96, f"ours, exact calibration:\nunder {bound:.0e} px".replace("e-0", "e-"),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=figstyle.INK2)
        if j == 0:
            ax.set_ylabel("pixel error: median, band to the max")
        ax = axes[1, j]
        rows = [d for d in r["docking"] if "pinhole" in d]
        rng_m = np.array([d["range"] for d in rows])
        for model, col in (("pinhole", PIN), ("refractive", OURS)):
            ax.plot(rng_m, [d[model]["position_mm"] for d in rows], "-", color=col, lw=1.6, marker="o", ms=2.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(3.1, 0.33)
        ax.set_ylim(0.03, 60)
        ax.set_xticks([3, 2, 1, 0.7, 0.5, 0.35])
        ax.set_xticklabels(["3", "2", "1", "0.7", "0.5", "0.35"])
        ax.minorticks_off()
        ax.set_title(f"{TITLE[name]}: docking")
        ax.set_xlabel("range to the dock (m), as the robot approaches")
        if j == 0:
            ax.set_ylabel("dock position error (mm)")
    handles = [plt.Line2D([], [], color=OURS, marker="o", ms=4, label=NAME["refractive"]),
               plt.Line2D([], [], color=PIN, marker="o", ms=4, label=NAME["pinhole"]),
               plt.Line2D([], [], color=PIN, ls="--", label="pinhole + distortion, calibrated on exact pixels "
                                                           "(its model error alone)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("One camera in three underwater housings: calibrated in water with 0.2 px of noise",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(HERE / "summary.png")
    plt.close(fig)
    print("wrote summary.png")


# ---- renders --------------------------------------------------------------------------------------------------------
def render_frame(name, k, scale, supersample=2):
    """The view through housing `name` at frame k of the approach, cached as a PNG."""
    from PIL import Image
    path = hz.CACHE / f"render_{name}_{k:02d}_{scale:g}.png"
    if path.exists():
        return np.asarray(Image.open(path)).astype(float) / 255
    h = hz.housings()[name]
    R_wh, t_wh, _ = demo_e.approach()[k]
    objs = ds.objects_in_housing(ds.objects_world(demo_e.DOCK_DOTS), R_wh, t_wh)
    img = ds.render(h["true"], h["window"], objs, R_wh, scale=scale, supersample=supersample)
    hz.CACHE.mkdir(parents=True, exist_ok=True)
    Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
    return img


def scaled(cam, s):
    K = cam.K.copy()
    K[0, 0], K[1, 1] = K[0, 0] * s, K[1, 1] * s
    K[0, 2], K[1, 2] = (K[0, 2] + 0.5) * s - 0.5, (K[1, 2] + 0.5) * s - 0.5
    return rc.Camera(K, cam.R, cam.t, None, (int(round(hz.W * s)), int(round(hz.H * s))))


# ---- views.png ------------------------------------------------------------------------------------------------------
def pinhole_error_map(res, name, rng_m=0.25, nx=64, ny=48):
    """The pinhole's pixel error (exact calibration) over the image, for points at `rng_m` from the lens."""
    h = hz.housings()[name]
    cam, win = h["true"], h["window"]
    cal = res[name]["calibration"]["exact"]
    pin = dict(K=np.array(cal["pinhole"]["K"]), dist=np.array(cal["pinhole"]["dist"]))
    M = (np.array(cal["M_pinhole"][0]), np.array(cal["M_pinhole"][1]))
    u, v = np.meshgrid(np.linspace(0, hz.W - 1, nx), np.linspace(0, hz.H - 1, ny))
    rays = np.c_[(u.ravel() - cam.K[0, 2]) / cam.K[0, 0], (v.ravel() - cam.K[1, 2]) / cam.K[1, 1], np.ones(u.size)]
    X = (rays / np.linalg.norm(rays, axis=1, keepdims=True) * rng_m - cam.t) @ cam.R
    truth, ok = hz.shoot(cam, win, X)
    err = np.linalg.norm(cb.project_pinhole(pin, X @ M[0].T + M[1]) - truth, axis=1)
    err[~(ok & demo_e.in_view(truth, 0))] = np.nan
    return truth, err


def views(res):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.interpolate import griddata
    figstyle.use()
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), gridspec_kw=dict(hspace=0.16, wspace=0.04))
    fig.subplots_adjust(left=0.01, right=0.93, top=0.95, bottom=0.07)
    k = 40
    rng_m = demo_e.approach()[k][2]
    gu, gv = np.meshgrid(np.arange(0, hz.W, 8.0), np.arange(0, hz.H, 8.0))
    for j, name in enumerate(ORDER):
        ax = axes[0, j]
        ax.imshow(render_frame(name, k, 0.5, supersample=3))
        ax.set_axis_off()
        ax.set_title(f"{TITLE[name]}: the dock {rng_m:.1f} m away")
        ax = axes[1, j]
        truth, err = pinhole_error_map(res, name)
        ok = np.isfinite(err)
        field = 10 ** griddata(truth[ok], np.log10(err[ok]), (gu, gv), method="linear")
        im = ax.imshow(field, cmap="magma", norm=LogNorm(vmin=0.03, vmax=100), extent=(-0.5, hz.W - 0.5, hz.H - 0.5,
                                                                                      -0.5), interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(figstyle.DARK)
        ax.set_title(f"pinhole + distortion, 0.25 m away: up to {np.nanmax(err):.1f} px", fontsize=9)
    cb_ = fig.colorbar(im, ax=axes[1, :], fraction=0.025, pad=0.01)
    cb_.set_label("pixel error (px)")
    fig.text(0.01, 0.015, "Bottom: where a pinhole + distortion calibrated in the water (on exact pixels) puts points "
             "0.25 m from the lens, against the truth. The refractive model's error is under 0.0001 px everywhere.",
             fontsize=8.5, color=figstyle.INK2)
    fig.savefig(HERE / "views.png")
    plt.close(fig)
    print("wrote views.png")


# ---- docking.gif ----------------------------------------------------------------------------------------------------
def outline_pixels(cam, win, R, t, n=40):
    """The board's frame and dots, posed at (R, t) in the housing frame, as the true camera sees them."""
    hx, hy = 0.235, 0.155
    s = np.linspace(-1, 1, n)
    edge = np.concatenate([np.c_[s * hx, -hy + 0 * s], np.c_[hx + 0 * s, s * hy], np.c_[-s * hx, hy + 0 * s],
                           np.c_[-hx + 0 * s, -s * hy]])
    P = np.c_[edge, np.zeros(len(edge))] @ np.asarray(R).T + np.asarray(t)
    D = demo_e.DOCK_DOTS @ np.asarray(R).T + np.asarray(t)
    out = []
    for X in (P, D):
        im = rc.project(X, cam, win)
        uv = np.full((len(X), 2), np.nan)
        uv[im.point] = im.uv
        out.append(uv)
    return out


def rgb(hexcol):
    return tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))


def font(size, bold=False):
    from PIL import ImageFont
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf"),
                 "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def chart(rows, k, size):
    """The running chart of the dock's position error, frames up to k: an RGB array of `size` (W, H)."""
    import matplotlib.pyplot as plt
    figstyle.use()
    W, H = size
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(figstyle.DARK)
    ax.set_facecolor(figstyle.DARK)
    for sp in ax.spines.values():
        sp.set_color("#5d5c58")
    ax.tick_params(colors="#c3c2b7", labelcolor="#c3c2b7")
    ax.grid(color="#3a3a38")
    done = [d for d in rows[:k + 1] if "pinhole" in d]
    for model, col in (("pinhole", PIN), ("refractive", OURS)):
        ax.plot([d["range"] for d in done], [d[model]["position_mm"] for d in done], "-", color=col, lw=2)
        if done:
            ax.plot(done[-1]["range"], done[-1][model]["position_mm"], "o", color=col, ms=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.1, 0.33)
    ax.set_ylim(0.1, 60)
    ax.set_xticks([3, 2, 1, 0.7, 0.5, 0.35])
    ax.set_xticklabels(["3", "2", "1", "0.7", "0.5", "0.35"])
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels(["0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_xlabel("range to the dock (m)", color="#c3c2b7")
    ax.set_ylabel("the dock's position error (mm)", color="#c3c2b7")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def video(res, name="tube", scale=0.6):
    """docking.gif: the approach through one housing, each model's estimate of the dock drawn where the camera would
    truly see it, with a magnified corner and the running error chart."""
    from PIL import Image, ImageDraw
    h = hz.housings()[name]
    cam, win = scaled(h["true"], scale), h["window"]
    rows = res[name]["docking"]
    frames_out = []
    W_OUT, H_OUT = 1280, 720
    VW, VH = cam.size
    x0, y0 = 24, 80
    for k, row in enumerate(rows):
        view = (render_frame(name, k, scale) * 255).round().astype(np.uint8)
        img = Image.fromarray(view)
        dr = ImageDraw.Draw(img)
        lines = {}
        if "pinhole" in row:
            for model, col in (("pinhole", PIN), ("refractive", OURS)):
                edge, dots = outline_pixels(cam, win, row[model]["R"], row[model]["t"])
                lines[model] = (edge, dots, rgb(col))
                pts = [tuple(p) for p in edge if np.isfinite(p).all()]
                if len(pts) > 1:
                    dr.line(pts + [pts[0]], fill=rgb(col), width=2)
                for u, v in dots[np.isfinite(dots).all(1)]:
                    dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=rgb(col), width=2)
        # the magnified corner: the board's top-left, as the truth puts it
        edge_t, _ = outline_pixels(cam, win, row["truth"]["R"], row["truth"]["t"])
        cu, cv = edge_t[0]
        Z, half_w, half_h = 4, 40, 30
        inset = None
        if np.isfinite([cu, cv]).all() and half_w <= cu <= VW - half_w and half_h <= cv <= VH - half_h:
            box = (int(cu) - half_w, int(cv) - half_h, int(cu) + half_w, int(cv) + half_h)
            inset = Image.fromarray(view).crop(box).resize((2 * half_w * Z, 2 * half_h * Z), Image.NEAREST)
            di = ImageDraw.Draw(inset)
            for model, (edge, dots, col) in lines.items():
                e = (edge - [box[0], box[1]]) * Z
                pts = [tuple(p) for p in e if np.isfinite(p).all()]
                if len(pts) > 1:
                    di.line(pts, fill=col, width=3)
                for u, v in (dots - [box[0], box[1]]) * Z:
                    if np.isfinite([u, v]).all():
                        di.ellipse([u - 14, v - 14, u + 14, v + 14], outline=col, width=3)
        canvas = Image.new("RGB", (W_OUT, H_OUT), rgb(figstyle.DARK))
        canvas.paste(img, (x0, y0))
        if inset is not None:
            canvas.paste(inset, (x0 + VW - inset.size[0] - 8, y0 + 8))
            ImageDraw.Draw(canvas).rectangle([x0 + VW - inset.size[0] - 8, y0 + 8, x0 + VW - 8, y0 + 8 + inset.size[1]],
                                             outline=(230, 230, 225), width=2)
        ch = chart(rows, k, (W_OUT - x0 - VW - 40, VH))
        canvas.paste(Image.fromarray(ch), (x0 + VW + 16, y0))
        dc = ImageDraw.Draw(canvas)
        dc.text((x0, 12), f"A robot docks with its camera in a {TITLE[name].lower()}", font=font(30, True),
                fill=(240, 240, 235))
        dc.text((x0, 48), h["about"].capitalize() + "; the dock marked by a 6 x 4 dot board.", font=font(17),
                fill=(195, 194, 183))
        y = y0 + VH + 10
        if "pinhole" in row:
            for model, col in (("refractive", OURS), ("pinhole", PIN)):
                d = row[model]
                dc.rectangle([x0, y + 6, x0 + 22, y + 10], fill=rgb(col))
                dc.text((x0 + 32, y - 4), f"{NAME[model]}: the dock {d['position_mm']:.1f} mm and "
                        f"{d['rotation_deg']:.2f} deg off", font=font(17), fill=(235, 235, 230))
                y += 25
        dc.text((x0 + VW + 24, y0 + VH + 10), f"range {row['range']:.2f} m", font=font(20, True), fill=(235, 235, 230))
        dc.text((x0 + VW + 24, y0 + VH + 38), "outlines: where each estimate puts the dock; inset: a corner x4",
                font=font(14), fill=(195, 194, 183))
        frames_out.append(canvas)
        print(f"frame {k + 1} / {len(rows)}", end="\r", flush=True)
    frames_out += [frames_out[-1]] * 15
    small = [f.resize((W_OUT * 5 // 8, H_OUT * 5 // 8), Image.LANCZOS) for f in frames_out]
    small = [s.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for s in small]
    small[0].save(HERE / "docking.gif", save_all=True, append_images=small[1:], duration=140, loop=0, optimize=True)
    print("\nwrote docking.gif")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=("summary", "views", "video"))
    args = p.parse_args()
    res = results()
    if args.only in (None, "summary"):
        summary(res)
    if args.only in (None, "views"):
        views(res)
    if args.only in (None, "video"):
        video(res)


if __name__ == "__main__":
    main()
