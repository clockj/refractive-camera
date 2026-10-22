"""Demo D, part 4: putting a needle at the ball's centre.

    python examples/demo_d_arm/precision.py --workers 5       # about 20 minutes

The ball rests at a point of the tank (tethered: it does not move, so perception alone decides), and the arm, holding
a needle on the slim tool's shaft, must put the needle's tip at the ball's centre.  It knows the ball only from the
three tank cameras, whose frames are ray-traced through the curved wall with the needle parked above the water.  It
moves the tip onto the perception's estimate and servos it there with its own joint angles to 0.01 mm, so the arm adds
no error of its own.  A trial succeeds when the tip ends within TOL of the true centre and the robot never touched the
tank.

Four perceptions estimate the centre from the same frames:
  * ours, whole ball: the centre fitted to the ball's blobs, predicted from every image of its surface by project()
    and weighted with the analytic Jacobian (ball_image.py);
  * AFRT, whole ball: the same fit with AFRT alone: each surface point's image from one AFRT start and every
    derivative by finite differences of AFRT (afrt_ball.py), no closed form of the window;
  * AFRT, blob centroids: parts 1-2's AFRT perception (each blob's centroid taken as the image of the centre);
  * pinhole + distortion: each camera's OpenCV model calibrated through the tank (parts 1-2).
40 placements: 20 in the open water (40-120 mm from the axis) and 20 next to the wall (136-160 mm).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE)]
import refractive_camera as rc  # noqa: E402
import afrt_ball  # noqa: E402
import arm  # noqa: E402
import ball_image  # noqa: E402
import demo_d as D  # noqa: E402

TOL = 0.5e-3                                 # m: the tip must end this close to the true centre
N_EACH = 20
KINDS = ("ours_ball", "afrt_ball", "afrt", "pinhole")
LABELS = {"ours_ball": "ours, whole ball", "afrt_ball": "AFRT alone, whole ball", "afrt": "AFRT, blob centroids",
          "pinhole": "pinhole + distortion"}
NEEDLE_PARK = np.r_[arm.TANK_XY, 0.43]       # the tip above the water, on the axis, while the cameras look


def placements(seed=4):
    """N_EACH ball centres in the open water and N_EACH next to the wall, each seen by at least two cameras."""
    rng = np.random.default_rng(seed)
    cams = D.cameras()
    out = []
    for near in (False, True):
        k = 0
        while k < N_EACH:
            r = rng.uniform(0.136, 0.160) if near else rng.uniform(0.04, 0.12)
            a = rng.uniform(-np.pi, np.pi)
            X = np.r_[arm.TANK_XY + r * np.array([np.cos(a), np.sin(a)]), rng.uniform(0.17, 0.27)]
            if sum(int(rc.project(X, c, D.TANK).counts()[0] > 0) for c in cams) < 2:
                continue
            out.append((X, near))
            k += 1
    return out


def pinhole_fits():
    """Each camera's OpenCV model calibrated through the tank, as in parts 1-2 (cached)."""
    path = D.CACHE / "pinhole_fits.json"
    if not path.exists():
        rng = np.random.default_rng(1)
        fits = [D.calibrate_pinhole(c, rng)[0] for c in D.cameras()]
        D.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"K": f.K.tolist(), "R": f.R.tolist(), "t": f.t.tolist(),
                                     "dist": list(map(float, f.dist))} for f in fits]))
    return [rc.Camera(f["K"], f["R"], f["t"], f["dist"], D.SIZE) for f in json.loads(path.read_text())]


def estimate(kind, blobs, cams, fits):
    """The perception's estimate of the centre (3,), or None."""
    if kind in ("ours_ball", "afrt_ball", "afrt"):
        per = D.Refractive(cams) if kind == "ours_ball" else D.AFRTPerception(cams)
        X0 = per.locate(blobs, None)
        if X0 is None:
            return None
        X1 = per.locate(blobs, X0)                    # every predicted image of the first estimate takes its blob
        X1 = X0 if X1 is None else X1
        if kind == "afrt":
            return X1
        lib = ball_image if kind == "ours_ball" else afrt_ball
        obs = [(c, lib.blobs_near(X1, c, D.TANK, b)) for c, b in zip(cams, blobs)]
        obs = [(c, b) for c, b in obs if len(b)]
        X, rms = lib.fit(obs, X1, D.TANK, D.BALL_RGB)
        return X if np.all(np.isfinite(X)) and rms < 2.0 else X1
    per = D.Pinhole(fits)
    X0 = per.locate(blobs, None)
    if X0 is None:
        return None
    X1 = per.locate(blobs, X0)
    return X0 if X1 is None else X1


def place(m, target, ball):
    """Move the needle's tip from its park onto `target` (over it first, then down), then servo it there with the
    arm's own joint angles.  Returns the tip's final position and the number of contacts with the tank."""
    import mujoco
    d = mujoco.MjData(m)
    ik = arm.IK(m)
    R_down = arm.fingers_down(np.pi / 2)
    q, _ = ik.solve(np.array([0.0, 0.2, 0.0, -1.9, 0.0, 2.1, 0.785]), NEEDLE_PARK, R_down, iters=300)
    d.qpos[:7], d.ctrl[:7] = q, q
    d.mocap_pos[0] = ball
    mujoco.mj_forward(m, d)
    tank_ids = arm.tank_geoms(m)
    steps = int(round(D.DT_CTRL / m.opt.timestep))
    contacts = 0
    goal = NEEDLE_PARK.copy()
    above = np.r_[target[:2], target[2] + 0.05]
    for waypoint, speed in ((above, 0.20), (np.asarray(target, float), 0.05)):
        while True:
            step = waypoint - goal
            n = np.linalg.norm(step)
            goal = waypoint.copy() if n <= speed * D.DT_CTRL else goal + step / n * speed * D.DT_CTRL
            q, _ = ik.solve(d.qpos, goal, R_down)
            d.ctrl[:7] = q
            for _ in range(steps):
                mujoco.mj_step(m, d)
                contacts += arm.tank_contacts(m, d, tank_ids)
            if n <= speed * D.DT_CTRL:
                break
    aim = np.asarray(target, float).copy()
    for _ in range(8):                                  # servo the tip on the arm's own joint angles
        for _ in range(20 * steps):
            mujoco.mj_step(m, d)
            contacts += arm.tank_contacts(m, d, tank_ids)
        tip = d.site("tcp").xpos.copy()
        miss = target - tip
        if np.linalg.norm(miss) < 1e-5:
            break
        aim = aim + miss
        q, _ = ik.solve(d.qpos, aim, R_down)
        d.ctrl[:7] = q
    return d.site("tcp").xpos.copy(), contacts


def trial(args):
    """One placement: the frames, each perception's estimate, and the needle put there."""
    i, X, near = args
    import mujoco
    m = arm.build_model(tool="needle")
    d = mujoco.MjData(m)
    ik = arm.IK(m)
    q, _ = ik.solve(np.array([0.0, 0.2, 0.0, -1.9, 0.0, 2.1, 0.785]), NEEDLE_PARK, arm.fingers_down(np.pi / 2),
                    iters=300)
    d.qpos[:7] = q
    d.mocap_pos[0] = X
    mujoco.mj_forward(m, d)
    cams = D.cameras()
    objs = D.inside_objects(X, arm.tool_parts(m, d, arm.NEEDLE_GEOMS))
    # a finer coarse pass than parts 1-3 (one sample per 2 x 2 pixels): at 4 x 4 the detector missed small second
    # copies of the ball (17 px of 430 in one frame, 53 px from the main blob), the very part one AFRT start lacks
    blobs = [D.detect(c, objs, factor=2) for c in cams]
    fits = pinhole_fits()
    seen = [int(rc.project(X, c, D.TANK).counts()[0]) for c in cams]
    row = {"i": i, "X": X.tolist(), "near_wall": bool(near), "r_mm": 1e3 * float(np.hypot(*(X[:2] - arm.TANK_XY))),
           "images_per_camera": seen, "blobs_per_camera": [int(len(b)) for b in blobs]}
    for kind in KINDS:
        t0 = time.perf_counter()
        est = estimate(kind, blobs, cams, fits)
        dt = time.perf_counter() - t0
        if est is None:
            row[kind] = {"estimate": None, "error_mm": None, "tip_error_mm": None, "contacts": 0, "time_s": dt}
            continue
        tip, contacts = place(m, est, X)
        row[kind] = {"estimate": est.tolist(), "error_mm": 1e3 * float(np.linalg.norm(est - X)),
                     "tip_error_mm": 1e3 * float(np.linalg.norm(tip - X)),
                     "tip_to_estimate_mm": 1e3 * float(np.linalg.norm(tip - est)), "contacts": int(contacts),
                     "time_s": dt}
    return row


def success(r, tol):
    return r["tip_error_mm"] is not None and r["tip_error_mm"] <= 1e3 * tol and r["contacts"] == 0


def summarise(rows, say):
    say(f"Placing a needle at the ball's centre: a trial succeeds when the tip ends within {1e3 * TOL:g} mm of the "
        f"true centre.")
    for near, label in ((False, "open water (40-120 mm)"), (True, "next to the wall (136-160 mm)")):
        sel = [r for r in rows if r["near_wall"] == near]
        say(f"\n  {label}, {len(sel)} placements")
        for k in KINDS:
            errs = [r[k]["error_mm"] for r in sel if r[k]["error_mm"] is not None]
            ok = sum(success(r[k], TOL) for r in sel)
            tips = [r[k]["tip_to_estimate_mm"] for r in sel if r[k].get("tip_to_estimate_mm") is not None]
            say(f"    {LABELS[k]:24s}: {ok:2d} of {len(sel)} succeed; estimate's error median "
                f"{np.median(errs):6.3f} mm, max {np.max(errs):6.2f} mm; tip to estimate {np.max(tips):.4f} mm max; "
                f"{np.median([r[k]['time_s'] for r in sel]):.2f} s per estimate")


def figure():
    """precision.png: the share of trials whose tip ends within each tolerance, open water and next to the wall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent))
    import figstyle as fs
    fs.use()
    rows = json.loads((HERE / "precision.json").read_text())["trials"]
    style = {"ours_ball": (fs.COLOR["refractive"], "-"), "afrt_ball": (fs.COLOR["afrt"], "-"),
             "afrt": (fs.COLOR["afrt"], "--"), "pinhole": (fs.COLOR["pinhole_dist"], "-")}
    tols = np.logspace(-2, 1.5, 200)                             # mm
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for ax, near, title in ((axes[0], False, "Open water (40-120 mm from the axis)"),
                            (axes[1], True, "Next to the wall (136-160 mm)")):
        sel = [r for r in rows if r["near_wall"] == near]
        for k in KINDS:
            e = np.array([r[k]["tip_error_mm"] if r[k]["tip_error_mm"] is not None else np.inf for r in sel])
            col, ls = style[k]
            ax.step(tols, [100 * np.mean(e <= t) for t in tols], where="post", color=col, ls=ls, lw=2,
                    label=f"{LABELS[k]} ({np.median([r[k]['time_s'] for r in sel]):.2g} s)")
        ax.axvline(1e3 * TOL, color=fs.MUTED, lw=1, ls=":")
        ax.text(1e3 * TOL * 1.08, 4, f"{1e3 * TOL:g} mm", color=fs.INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(tols[0], tols[-1])
        ax.set_ylim(0, 104)
        ax.set_title(title)
        ax.set_xlabel("tolerance: the tip within this of the true centre (mm)")
    axes[0].set_ylabel("trials that succeed (%)")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, title="perception (median time per estimate)",
               title_fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Part 4: putting a needle at the ball's centre (20 placements each)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(HERE / "precision.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--figure-only", action="store_true", help="draw precision.png from precision.json")
    args = ap.parse_args()
    if args.figure_only:
        figure()
        return
    lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    arm.fetch_panda()
    pinhole_fits()
    jobs = [(i, X, near) for i, (X, near) in enumerate(placements())]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = []
        for row in ex.map(trial, jobs):
            rows.append(row)
            print(f"  trial {row['i'] + 1:2d}/{len(jobs)} ({'wall' if row['near_wall'] else 'open'}): " + ", ".join(
                f"{k} {row[k]['error_mm']:.3f}" if row[k]['error_mm'] is not None else f"{k} -" for k in KINDS),
                flush=True)
    summarise(rows, say)
    say(f"\n{len(jobs)} trials in {time.perf_counter() - t0:.0f} s")
    (HERE / "precision.json").write_text(json.dumps({"tol_mm": 1e3 * TOL, "trials": rows}, indent=1))
    (HERE / "precision_out.txt").write_text("\n".join(lines) + "\n")
    figure()


if __name__ == "__main__":
    main()
