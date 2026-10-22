"""Demo E: a robot's camera in an underwater housing, through a flat port, a dome port and a tube.

    python examples/demo_e_housing/demo_e.py            (about 3 minutes)

For each housing (housing.py), one camera:
1. **Calibration in the water.** 30 views of a dot board 0.25-1.5 m away, the dots' pixels found by tracing alone
   (`housing.shoot`).  Two models are fitted to the same views: a pinhole with OpenCV's five distortion
   coefficients, and the refractive model, whose unknowns are the lens's intrinsics and the camera's pose in the
   housing (calibrate.py).  Each is fitted twice: to the exact pixels, which leaves only the model's own error, and
   with 0.2 px of noise, as a real calibration has.
2. **Where points appear**, from 0.15 m to 4 m: each model's pixel against the traced truth, over the whole image.
3. **Docking.** The robot approaches a dock marked by a 6 x 4 dot board, from 3 m to 0.35 m, swaying and turning.
   Each frame, each model (from the noisy calibration) estimates the dock's pose from its dots, also with 0.2 px of
   noise, in the robot's frame.

Both models are placed in the robot's frame by the same alignment, from the calibration boards (a perfect hand-eye
calibration).  The numbers go to results.json and demo_e_out.txt.
"""
import json
import re
import sys
import time

import numpy as np

import calibrate as cb
import housing as hz

RNG_SEED = 2026
NOISE = 0.2                              # px, on every detected dot
N_VIEWS = 30
CAL_DOTS = hz.board(9, 7, 0.05)          # 0.40 x 0.30 m
DOCK_DOTS = hz.board(6, 4, 0.08)         # 0.40 x 0.24 m
RANGES = (0.15, 0.25, 0.4, 0.6, 1.0, 1.5, 2.5, 4.0)
N_FRAMES = 60
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def compact_json(obj):
    """JSON with numbers to 10 significant digits and each list of numbers on one line."""
    def rounded(o):
        if isinstance(o, float):
            return float(f"{o:.10g}")
        if isinstance(o, dict):
            return {k: rounded(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rounded(v) for v in o]
        return o
    s = json.dumps(rounded(obj), indent=1)
    return re.sub(r"\[\s*([-+0-9.eE,\s]+?)\s*\]", lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(","))
                  + "]", s)


def in_view(uv, margin):
    return (uv[:, 0] >= margin) & (uv[:, 0] <= hz.W - 1 - margin) & (uv[:, 1] >= margin) & (uv[:, 1] <= hz.H - 1 -
                                                                                              margin)


def calibration_views(h, rng):
    """N_VIEWS views of the calibration board, every dot 40 px inside the image: (exact, noisy) lists of
    dict(dots, uv, R, t), with the board's true pose (R, t) in the housing frame."""
    cam, win = h["true"], h["window"]
    exact, noisy = [], []
    while len(exact) < N_VIEWS:
        d = rng.uniform(0.25, 1.5)
        turn = hz.expm(rng.normal(size=3) / np.sqrt(3) * np.radians(rng.uniform(0, 35)))
        centre = np.array([rng.uniform(-0.3, 0.3) * d, rng.uniform(-0.22, 0.22) * d, d])       # camera frame
        R_hb, t_hb = cam.R.T @ turn, cam.R.T @ (centre - cam.t)
        uv, ok = hz.shoot(cam, win, CAL_DOTS @ R_hb.T + t_hb)
        if ok.all() and in_view(uv, 40).all():
            exact.append(dict(dots=CAL_DOTS, uv=uv, R=R_hb, t=t_hb))
            noisy.append(dict(exact[-1], uv=uv + rng.normal(0.0, NOISE, uv.shape)))
    return exact, noisy


def calibrate_both(h, views):
    """Both models fitted to the views, and each one's alignment to the housing frame."""
    true_boards = [(v["R"], v["t"]) for v in views]
    pin = cb.calibrate_pinhole(views)
    ref = cb.calibrate_refractive(views, h["window"], h["nominal"])
    return pin, ref, cb.frame_map(pin["boards"], true_boards), cb.frame_map(ref["boards"], true_boards)


def where_points_appear(h, pin, ref, M_pin, M_ref):
    """Each model's pixel error (px) against the traced truth, per range, on a 25 x 19 grid over the image."""
    cam, win = h["true"], h["window"]
    u, v = np.meshgrid(np.linspace(20, hz.W - 21, 25), np.linspace(20, hz.H - 21, 19))
    rays = np.c_[(u.ravel() - cam.K[0, 2]) / cam.K[0, 0], (v.ravel() - cam.K[1, 2]) / cam.K[1, 1], np.ones(u.size)]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    out = {}
    for rng_m in RANGES:
        X = (rays * rng_m - cam.t) @ cam.R                         # housing frame, straight out from the lens
        truth, ok = hz.shoot(cam, win, X)
        ok &= in_view(truth, 0)
        Xp = X @ M_pin[0].T + M_pin[1]
        Xr = X @ M_ref[0].T + M_ref[1]
        e_pin = np.linalg.norm(cb.project_pinhole(pin, Xp) - truth, axis=1)[ok]
        r = cb._residuals(ref["camera"], win, Xr, truth, jac=False)
        e_ref = np.linalg.norm(r, axis=1)[ok]
        out[rng_m] = dict(n=int(ok.sum()), pinhole=summary(e_pin), refractive=summary(e_ref))
    return out


def summary(e):
    return dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)), max=float(np.max(e)))


def approach():
    """The robot's housing frame in the dock's (world) frame, per frame: (R_wh, t_wh) and the range."""
    s = np.linspace(0.0, 1.0, N_FRAMES)
    rng_m = 3.0 * (0.35 / 3.0) ** s                                   # 3 m to 0.35 m, slowing down
    ph = 2 * np.pi * 1.3 * s
    frames = []
    for k in range(N_FRAMES):
        pos = np.array([-rng_m[k], 0.22 * np.sin(ph[k]) * min(1.0, rng_m[k]), 0.10 * np.sin(2 * ph[k] + 0.4)
                        * min(1.0, rng_m[k])])
        yaw, pitch = np.radians(8.0) * np.sin(ph[k] + 0.5), np.radians(4.0) * np.sin(1.3 * ph[k])
        R_wh = hz.expm([0.0, 0.0, yaw]) @ hz.expm([0.0, pitch, 0.0])
        frames.append((R_wh, pos, float(rng_m[k])))
    return frames


def dock_in_world():
    """The dock board's pose in the world: at the origin, across the path of the robot, which comes along +x.  Its
    x axis runs to the robot's right (-y) and its y axis down (-z), as the robot's camera sees them."""
    return np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]), np.zeros(3)


def docking(h, pin, ref, M_pin, M_ref, rng):
    """Per frame: the dock's true pose in the housing frame and each model's estimate."""
    cam, win = h["true"], h["window"]
    R_wd, t_wd = dock_in_world()
    rows = []
    for R_wh, t_wh, rng_m in approach():
        R_hd, t_hd = R_wh.T @ R_wd, R_wh.T @ (t_wd - t_wh)            # dock -> housing
        X = DOCK_DOTS @ R_hd.T + t_hd
        uv, ok = hz.shoot(cam, win, X)
        seen = ok & in_view(uv, 5)
        uv = uv + rng.normal(0.0, NOISE, uv.shape)
        if seen.sum() < 8:
            rows.append(dict(range=rng_m, seen=int(seen.sum())))
            continue
        dots, obs = DOCK_DOTS[seen], uv[seen]
        # pinhole + distortion: OpenCV's PnP, then the alignment back to the housing frame
        Rp, tp = cb.pnp_pinhole(pin, dots, obs)
        est_pin = cb.compose(cb.inverse(M_pin), (Rp, tp))
        # ours: start from a plain PnP with our intrinsics, then the refractive fit, in our housing frame
        c = ref["camera"]
        R0, t0 = cb.pnp_pinhole(dict(K=c.K, dist=np.zeros(5)), dots, obs)
        start = (c.R.T @ R0, c.R.T @ (t0 - c.t))
        est_ref = cb.compose(cb.inverse(M_ref), cb.pnp_refractive(c, win, dots, obs, start))
        row = dict(range=rng_m, seen=int(seen.sum()))
        for name, (Re, te) in (("pinhole", est_pin), ("refractive", est_ref)):
            dR = Re @ R_hd.T
            row[name] = dict(position_mm=1e3 * float(np.linalg.norm(te - t_hd)),
                             rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))),
                             R=Re.tolist(), t=te.tolist())
        row["truth"] = dict(R=R_hd.tolist(), t=t_hd.tolist())
        rows.append(row)
    return rows


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)
    results = {}
    for name, h in hz.housings().items():
        say(f"\n== {h['label']}: {h['about']}")
        exact, noisy = calibration_views(h, rng)
        where, calib = {}, {}
        for kind, views in (("exact", exact), ("noisy", noisy)):
            pin, ref, M_pin, M_ref = calibrate_both(h, views)
            c = ref["camera"]
            c_true = cb.inverse(M_ref)[0] @ c.center + cb.inverse(M_ref)[1]
            say(f"  calibration on {N_VIEWS} views ({N_VIEWS * len(CAL_DOTS)} dots), "
                + ("exact pixels:" if kind == "exact" else f"noise {NOISE} px:"))
            say(f"    pinhole + distortion: rms {pin['rms']:.3f} px; f {pin['K'][0, 0]:.1f}, distortion "
                f"{np.round(pin['dist'], 4).tolist()}")
            say(f"    refractive: rms {ref['rms']:.3f} px in {ref['iters']} iterations; f {c.K[0, 0]:.2f} / "
                f"{c.K[1, 1]:.2f} (true 812), principal point ({c.K[0, 2]:.2f}, {c.K[1, 2]:.2f}) (true 644, 476.5); "
                f"the camera {1e3 * np.linalg.norm(c_true - h['true'].center):.2f} mm from its true place")
            where[kind] = where_points_appear(h, pin, ref, M_pin, M_ref)
            say("    where points appear: pixel error against the traced truth, median / 95th percentile / max")
            for r_m, w in where[kind].items():
                p, q = w["pinhole"], w["refractive"]
                say(f"      {r_m:4.2f} m ({w['n']:3d} points): pinhole + distortion {p['median']:6.2f} / "
                    f"{p['p95']:6.2f} / {p['max']:6.2f} px;  refractive {q['median']:5.2f} / {q['p95']:5.2f} / "
                    f"{q['max']:5.2f} px")
            calib[kind] = dict(pinhole=dict(rms=pin["rms"], K=pin["K"].tolist(), dist=pin["dist"].tolist()),
                               refractive=dict(rms=ref["rms"], K=c.K.tolist(), R=c.R.tolist(),
                                               centre=c.center.tolist(), iters=ref["iters"],
                                               centre_error_mm=1e3 * float(np.linalg.norm(c_true - h["true"].center))),
                               M_pinhole=[M_pin[0].tolist(), M_pin[1].tolist()],
                               M_refractive=[M_ref[0].tolist(), M_ref[1].tolist()])
        dock = docking(h, pin, ref, M_pin, M_ref, rng)                       # with the noisy calibration
        say("  docking: the dock's position (mm) and orientation (deg) error in the robot's frame, median / max")
        for lo, hi in ((1.0, 3.01), (0.35, 1.0)):
            sel = [d for d in dock if lo <= d["range"] < hi and "pinhole" in d]
            for m in ("pinhole", "refractive"):
                pe = [d[m]["position_mm"] for d in sel]
                re = [d[m]["rotation_deg"] for d in sel]
                say(f"    {lo:.2f}-{min(hi, 3.0):.2f} m ({len(sel)} frames), {m:10s}: {np.median(pe):7.2f} / "
                    f"{np.max(pe):7.2f} mm, {np.median(re):6.3f} / {np.max(re):6.3f} deg")
        results[name] = dict(label=h["label"], about=h["about"], calibration=calib,
                             where={kind: {str(k): v for k, v in w.items()} for kind, w in where.items()},
                             docking=dock)
    (hz.HERE / "results.json").write_text(compact_json(results))
    say(f"\nDemo E done in {time.perf_counter() - t_start:.0f} s")
    (hz.HERE / "demo_e_out.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.exit(main())
