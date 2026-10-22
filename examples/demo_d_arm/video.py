"""The video of Demo D: one ball path near the wall, two perceptions side by side.

Each half shows the arm from outside (MuJoCo; the translucent ball is where that perception puts the ball) above the
frame of one tank camera, ray-traced through the wall:
  * white crosses: the blobs the camera detects;
  * rings: where the perception's estimate of the ball appears by its model; ours draws every image, so two rings
    where the wall shows the ball twice, and none where it hides it.
Writes catch.mp4 (1280 x 720) and a smaller catch.gif.

    python examples/demo_d_arm/video.py --seed 9
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE), str(HERE.parent)]
import refractive_camera as rc  # noqa: E402
import arm  # noqa: E402
import demo_d as D  # noqa: E402
from figstyle import COLOR  # noqa: E402

W_OUT, H_OUT = 1280, 720
HALF = W_OUT // 2
TOP_H = 430
CROP = (80, 96, 560, 416)                               # the part of a camera frame shown (u0, v0, u1, v1)
STATE = {"track": "watching the ball", "approach": "reaching over it", "descend": "going down to it",
         "final": "closing in on the prediction", "close": "closing", "lift": "lifting", "done": ""}


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


def render_crop(cam, objs):
    """The camera's colours over CROP only (pixel centres), one ray per pixel."""
    u0, v0, u1, v1 = CROP
    V, U = np.meshgrid(np.arange(v0, v1, dtype=float), np.arange(u0, u1, dtype=float), indexing="ij")
    rgb = D.render_pixels(cam, np.c_[U.ravel(), V.ravel()], objs)
    return rgb.reshape(v1 - v0, u1 - u0, 3)


def ring(dr, u, v, r, col, width=3):
    dr.ellipse([u - r, v - r, u + r, v + r], outline=col, width=width)


def main():
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--cams", default="1,2", help="the two tank cameras shown")
    ap.add_argument("--every", type=int, default=1, help="control steps per video frame")
    ap.add_argument("--preview", type=int, default=None, help="render this step only, to _cache/video_preview.png")
    args = ap.parse_args()
    runs = {}
    for kind in ("refractive", "pinhole"):
        f = D.CACHE / f"frames_{kind}_{args.seed}.npy"
        if not f.exists():
            print(kind, D.run_trial((args.seed, kind, True)))
        runs[kind] = list(np.load(f, allow_pickle=True))
    res = {k: next(t for t in json.loads((HERE / "results.json").read_text())[k]["trials"] if t["seed"] == args.seed)
           for k in runs}
    fits = json.loads((D.CACHE / "pinhole_fits.json").read_text())
    pin = [rc.Camera(f["K"], f["R"], f["t"], f["dist"], D.SIZE) for f in fits]
    cams = D.cameras()
    shown = [int(c) for c in args.cams.split(",")]
    m = arm.build_model(ghost=True)
    m.cam("overview").pos[:] = [0.98, -0.62, 0.58]
    m.cam("overview").quat[:] = 0
    look = arm.look_at_axes([0.98, -0.62, 0.58], [0.46, 0.04, 0.33])
    xa, ya = np.array(look[:3]), np.array(look[3:])
    Rcam = np.column_stack([xa, ya, np.cross(xa, ya)])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rcam.ravel())
    m.cam("overview").quat[:] = q
    d = mujoco.MjData(m)
    n = max(len(r) for r in runs.values())
    f_title, f_small, f_tag = font(22, True), font(15), font(14, True)
    frames = []
    with mujoco.Renderer(m, TOP_H, HALF) as ren:
        for k in (range(0, n, args.every) if args.preview is None else [args.preview]):
            canvas = Image.new("RGB", (W_OUT, H_OUT), (26, 26, 25))
            for col_i, (kind, label) in enumerate((("refractive", "refraction-aware (ours)"),
                                                   ("pinhole", "pinhole + distortion"))):
                colour = rgb(COLOR["refractive" if kind == "refractive" else "pinhole_dist"])
                fr = runs[kind][min(k, len(runs[kind]) - 1)]
                d.qpos[:] = fr["qpos"]
                d.mocap_pos[0] = fr["ball"]
                d.mocap_pos[1] = fr["est"] if fr["est"] is not None else [0, 0, -1]
                m.geom_rgba[m.body_geomadr[m.body("estimate").id]] = list(np.array(colour) / 255) + [0.35]
                mujoco.mj_forward(m, d)
                ren.update_scene(d, camera="overview")
                top = Image.fromarray(ren.render())
                x0 = col_i * HALF
                canvas.paste(top, (x0, 0))
                dc = ImageDraw.Draw(canvas)
                objs = D.inside_objects(fr["ball"], arm.tool_parts(m, d))
                cw = (HALF - 48) // 2
                ch = int(cw * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0]))
                for j, ci in enumerate(shown):
                    cam = cams[ci]
                    view = render_crop(cam, objs)
                    im = Image.fromarray((np.clip(view, 0, 1) * 255).astype(np.uint8))
                    dr = ImageDraw.Draw(im)
                    ox, oy = CROP[0], CROP[1]
                    for u, v, _ in fr["blobs"][ci]:
                        u, v = u - ox, v - oy
                        dr.line([(u - 10, v), (u + 10, v)], fill=(255, 255, 255), width=3)
                        dr.line([(u, v - 10), (u, v + 10)], fill=(255, 255, 255), width=3)
                    if fr["est"] is not None:
                        if kind == "refractive":
                            im_e = rc.project(fr["est"], cam, D.TANK)
                            uvs = im_e.uv[im_e.in_image]
                        else:
                            uvs = pin[ci].project_points(fr["est"])[0].reshape(1, 2)
                        for u, v in uvs[np.isfinite(uvs).all(1)]:
                            ring(dr, u - ox, v - oy, 18, colour, 4)
                    im = im.resize((cw, ch), Image.LANCZOS)
                    xc = x0 + 16 + j * (cw + 16)
                    canvas.paste(im, (xc, TOP_H + 30))
                    nimg = int(rc.project(fr["ball"], cam, D.TANK).in_image.sum())
                    seen = {0: "the wall hides the ball", 1: "sees the ball once", 2: "sees the ball twice"}.get(nimg, "")
                    dc.text((xc, TOP_H + 7), f"camera {ci}: {seen}", font=f_small,
                            fill=(240, 239, 236) if nimg != 1 else (195, 194, 183))
                # header on the arm's view
                dc.rectangle([x0, 0, x0 + HALF, 44], fill=(26, 26, 25))
                dc.rectangle([x0 + 16, 14, x0 + 30, 28], fill=colour)
                dc.text((x0 + 40, 9), label, font=f_title, fill=(240, 239, 236))
                msg = STATE.get(fr["state"], fr["state"])
                if fr["state"] in ("lift", "done") or k >= len(runs[kind]) - 1:
                    ok = res[kind]["caught"]
                    msg = "caught it" if ok else "missed it"
                    tag = (12, 163, 12) if ok else (208, 59, 59)
                    dc.rounded_rectangle([x0 + HALF - 104, 10, x0 + HALF - 16, 36], 6, fill=tag)
                    dc.text((x0 + HALF - 92, 13), ("caught" if ok else "missed"), font=f_tag, fill=(255, 255, 255))
                dc.text((x0 + 16, TOP_H - 26), msg, font=f_small, fill=(40, 40, 38))
            dc = ImageDraw.Draw(canvas)
            dc.line([(HALF, 0), (HALF, H_OUT)], fill=(26, 26, 25), width=4)
            fr = runs["refractive"][min(k, len(runs["refractive"]) - 1)]
            r_mm = 1e3 * np.hypot(*(fr["ball"][:2] - arm.TANK_XY))
            dc.text((20, H_OUT - 26), f"t = {fr['t']:4.1f} s     the ball is {r_mm:5.1f} mm from the tank's axis "
                    f"(wall at 172 mm)     white +: blobs the camera detects     rings: where each perception's "
                    f"estimate appears, by its model", font=f_small, fill=(195, 194, 183))
            if args.preview is not None:
                canvas.save(D.CACHE / "video_preview.png")
                print("wrote", D.CACHE / "video_preview.png")
                return
            frames.append(np.asarray(canvas))
            print(f"frame {len(frames)} / {(n + args.every - 1) // args.every}", end="\r", flush=True)
    for _ in range(20):                                        # hold the last frame
        frames.append(frames[-1])
    imageio.mimsave(HERE / "catch.mp4", frames, fps=20 // args.every, macro_block_size=8, quality=8)
    small = [Image.fromarray(f).resize((W_OUT * 5 // 8, H_OUT * 5 // 8), Image.LANCZOS) for f in frames[::2]]
    small = [s.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for s in small]
    small[0].save(HERE / "catch.gif", save_all=True, append_images=small[1:], duration=100 * args.every, loop=0,
                  optimize=True)
    print(f"\nwrote catch.mp4 and catch.gif ")


if __name__ == "__main__":
    main()
