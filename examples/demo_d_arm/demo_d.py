"""Demo D: a robot arm catches a ball swimming in a water-filled acrylic tank, seen by three cameras through the wall.

The arm (Franka Panda, MuJoCo) holds a slim long-reach gripper in the water.  It knows where the ball is only from
three cameras outside the tank, whose frames are ray-traced through the curved wall by refractive_camera.  Three
perceptions drive the same arm:
  * refraction-aware (ours): project() predicts every image of the ball, so its blobs are associated even when a
    camera sees it twice or loses it in the wall's shadow; the ball is triangulated through the wall;
  * pinhole + AFRT: the same, but each camera's predicted image comes from one AFRT start (Belden 2013);
  * pinhole + distortion: each camera's own OpenCV model, calibrated through the tank on a target in its middle, as a
    user unaware of refraction would do; straight-ray triangulation.
Each frame feeds a constant-velocity Kalman filter, whose prediction guides the tool.  Half of the ball's paths stay
in the open water (40-120 mm from the axis), half swim where the wall shows a point twice or hides it (136-160 mm).
A catch counts when the ball is between the fingers as they close, and the robot never touched the tank.

    python examples/demo_d_arm/demo_d.py --kinds refractive,afrt,pinhole     # 40 trials each (about 1 hour)
    python examples/demo_d_arm/demo_d.py --trials 6
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
from refractive_camera.scene import Box, Cylinder, Disk, NoiseTexture, Sphere, Uniform, shade  # noqa: E402
import arm  # noqa: E402

CACHE = arm.CACHE
TANK = rc.CylindricalWindow(radii=(arm.R_OUT, arm.R_IN), indices=(1.0, 1.49, 1.33),
                            axis_point=(arm.TANK_XY[0], arm.TANK_XY[1], 0.0))
SIZE, FOCAL, CAM_Z, CAM_DIST = (640, 512), 800.0, 0.22, 0.70
CAM_AZ = np.radians([-75.0, 5.0, 85.0])
DT_CTRL, FPS = 0.05, 20
ABOVE = (0.78, 0.82, 0.86)                       # what a camera sees above the water line, inside the tank
BG = (0.93, 0.93, 0.92)
BALL_RGB = (1.0, 0.45, 0.05)
SAND = NoiseTexture(0.01, [(0.55, 0.50, 0.40), (0.82, 0.76, 0.62)], spot=(0.35, 0.3, 0.25), seed=3)


def cameras():
    c = np.r_[arm.TANK_XY, CAM_Z]
    return [rc.Camera.look_at(c + [CAM_DIST * np.cos(a), CAM_DIST * np.sin(a), 0.0], c, f=FOCAL, size=SIZE)
            for a in CAM_AZ]


# ---- rendering the camera frames ---------------------------------------------------------------------------------

TOOL_RGB = {"shaft": (0.70, 0.72, 0.76), "wrist": (0.18, 0.18, 0.2), "left_pad": (0.93, 0.93, 0.95),
            "right_pad": (0.93, 0.93, 0.95)}


def inside_objects(ball, tool):
    """What stands in the water: the ball, the tool's parts (arm.tool_parts), the sandy floor and the stones."""
    objs = [Sphere(ball, arm.BALL_R, Uniform(BALL_RGB)),
            Disk((arm.TANK_XY[0], arm.TANK_XY[1], arm.Z_FLOOR + 0.004), arm.R_IN - 0.001, SAND)]
    for c, (_, r, col) in zip(arm.stone_centres(), arm.STONES):
        objs.append(Sphere(c, r, NoiseTexture(0.012, [tuple(0.6 * np.array(col)), col, tuple(np.minimum(1.3 * np.array(col), 1))],
                                              spot=tuple(0.4 * np.array(col)), seed=int(100 * r))))
    for name, pos, R, size in tool:
        if name == "shaft":
            objs.append(Cylinder(pos, size[0], size[1], Uniform(TOOL_RGB[name]), rotation=R))
        else:
            objs.append(Box(pos, tuple(size), Uniform(TOOL_RGB[name]), rotation=R))
    return objs


def render_pixels(cam, uv, objs):
    """Colours seen at pixels (n, 2): the scene in the water through the wall; above the water line, a plain tint."""
    from refractive_camera.scene import cast
    h = cast(cam, TANK, uv, objs)
    o, d = cam.rays(uv)
    tr = rc.trace(TANK, o, d)
    rgb = shade(objs, h, background=BG)
    wet = tr.ok & (tr.entry[:, 2] < arm.Z_WATER) & ~((h.obj >= 0) & (h.point[:, 2] > arm.Z_WATER))
    water_bg = tr.ok & (h.obj < 0)
    rgb[water_bg] = (0.70, 0.78, 0.84)
    rgb[tr.ok & ~wet] = ABOVE
    return rgb


def render_full(cam, objs, ss=2):
    W, H = cam.size
    off = (np.arange(ss) + 0.5) / ss - 0.5
    V, U = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    c = np.c_[U.ravel(), V.ravel()]
    sub = (c[:, None, :] + np.stack(np.meshgrid(off, off, indexing="xy"), -1).reshape(1, -1, 2)).reshape(-1, 2)
    return render_pixels(cam, sub, objs).reshape(-1, ss * ss, 3).mean(1).reshape(H, W, 3)


def orange(rgb):
    return np.clip((rgb[..., 0] - rgb[..., 2] - 0.35) / 0.3, 0, 1) * (rgb[..., 1] < 0.75 * rgb[..., 0])


def detect(cam, objs, factor=4, ss=2):
    """Ball blobs: a coarse frame finds them, a supersampled crop around each measures its centroid."""
    from scipy import ndimage
    W, H = cam.size
    Wc, Hc = W // factor, H // factor
    V, U = np.meshgrid(np.arange(Hc, dtype=float), np.arange(Wc, dtype=float), indexing="ij")
    uv = np.c_[U.ravel(), V.ravel()] * factor + (factor - 1) / 2
    coarse = orange(render_pixels(cam, uv, objs)).reshape(Hc, Wc) > 0.2
    lab, n = ndimage.label(ndimage.binary_dilation(coarse, iterations=2))
    blobs = []
    off = (np.arange(ss) + 0.5) / ss - 0.5
    for sl in ndimage.find_objects(lab):
        v0, v1 = max(sl[0].start * factor - 6, 0), min(sl[0].stop * factor + 6, H)
        u0, u1 = max(sl[1].start * factor - 6, 0), min(sl[1].stop * factor + 6, W)
        V, U = np.meshgrid(np.arange(v0, v1, dtype=float), np.arange(u0, u1, dtype=float), indexing="ij")
        c = np.c_[U.ravel(), V.ravel()]
        sub = (c[:, None, :] + np.stack(np.meshgrid(off, off, indexing="xy"), -1).reshape(1, -1, 2)).reshape(-1, 2)
        w = orange(render_pixels(cam, sub, objs)).reshape(-1, ss * ss).mean(1).reshape(v1 - v0, u1 - u0)
        lab2, n2 = ndimage.label(w > 0.05)
        for k in range(1, n2 + 1):
            m = lab2 == k
            if w[m].sum() < 1.0:
                continue                                    # a sliver too faint to measure
            vv, uu = np.nonzero(m)
            blobs.append((np.sum(w[m] * (uu + u0)) / w[m].sum(), np.sum(w[m] * (vv + v0)) / w[m].sum(), w[m].sum()))
    return np.array(blobs).reshape(-1, 3)


# ---- perception ---------------------------------------------------------------------------------------------------

class Kalman:
    """Constant velocity; state (position, velocity)."""

    def __init__(self, q=0.05, r=0.002):
        self.x, self.P, self.q, self.r = None, None, q, r

    def predict(self, dt):
        if self.x is None:
            return None
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        Q = self.q ** 2 * np.block([[dt ** 3 / 3 * np.eye(3), dt ** 2 / 2 * np.eye(3)],
                                    [dt ** 2 / 2 * np.eye(3), dt * np.eye(3)]])
        self.x, self.P = F @ self.x, F @ self.P @ F.T + Q
        return self.x[:3]

    def update(self, z):
        if self.x is None:
            self.x, self.P = np.r_[z, 0, 0, 0], np.diag([self.r ** 2] * 3 + [0.05 ** 2] * 3)
            return
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        S = H @ self.P @ H.T + self.r ** 2 * np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    def ahead(self, t):
        return self.x[:3] + t * self.x[3:]


class Refractive:
    name = "refractive"

    def __init__(self, cams):
        self.cams = cams

    def locate(self, blobs, pred, gate=40.0):
        cams, pix = [], []
        for cam, b in zip(self.cams, blobs):
            if not len(b):
                continue
            if pred is None:
                cams.append(cam)
                pix.append(b[np.argmax(b[:, 2]), :2])
                continue
            im = rc.project(pred, cam, TANK)
            if not len(im):
                continue                                    # the ball is in this camera's shadow
            for p in im.uv[np.isfinite(im.uv).all(1)]:
                dist = np.linalg.norm(b[:, :2] - p, axis=1)
                if dist.min() < gate:
                    cams.append(cam)
                    pix.append(b[np.argmin(dist), :2])      # every predicted image, so both when it is seen twice
        if len(cams) < 2 or len({id(c) for c in cams}) < 2:
            return None
        try:
            X, info = rc.triangulate(cams, np.array(pix), TANK)
        except ValueError:
            return None
        return X if np.all(np.isfinite(X)) else None


class AFRTPerception(Refractive):
    """Pinhole + AFRT: the refractive perception, but each camera's predicted image of the ball comes from one AFRT
    start (Belden 2013), so at most one image, and none where the iteration fails.  The ball is triangulated from the
    blobs' traced rays (least squares, no refinement by project())."""
    name = "afrt"

    def locate(self, blobs, pred, gate=40.0):
        from refractive_camera.baselines import afrt
        cams, pix = [], []
        for cam, b in zip(self.cams, blobs):
            if not len(b):
                continue
            if pred is None:
                cams.append(cam)
                pix.append(b[np.argmax(b[:, 2]), :2])
                continue
            d, _, _ = afrt(TANK, cam.center, pred)
            if d is None:
                continue                                    # AFRT gives no image: this camera waits
            tr = rc.trace(TANK, cam.center, d)
            if not tr.ok[0] or tr.miss_distance(np.asarray(pred, float)[None])[0] > 1e-9:
                continue
            p, front = cam.pixels_from_directions((d @ cam.R.T)[None])
            if not front[0] or not np.all(np.isfinite(p)):
                continue
            dist = np.linalg.norm(b[:, :2] - p[0], axis=1)
            if dist.min() < gate:
                cams.append(cam)
                pix.append(b[np.argmin(dist), :2])
        if len(cams) < 2 or len({id(c) for c in cams}) < 2:
            return None
        try:
            X, info = rc.triangulate(cams, np.array(pix), TANK, refine=False)
        except ValueError:
            return None
        return X if np.all(np.isfinite(X)) else None


class Pinhole:
    name = "pinhole"

    def __init__(self, fits):
        self.cams = fits

    def locate(self, blobs, pred, gate=40.0):
        A, bvec, n = np.zeros((3, 3)), np.zeros(3), 0
        for cam, b in zip(self.cams, blobs):
            if not len(b):
                continue
            if pred is None:
                j = np.argmax(b[:, 2])
            else:
                p, _ = cam.project_points(pred)
                dist = np.linalg.norm(b[:, :2] - p, axis=1)
                if dist.min() > gate:
                    continue
                j = np.argmin(dist)
            o, d = cam.rays(b[j, :2][None])
            if not np.all(np.isfinite(d)):
                continue
            M = np.eye(3) - np.outer(d[0], d[0])
            A += M
            bvec += M @ o[0]
            n += 1
        if n < 2:
            return None
        return np.linalg.solve(A, bvec)


def measure(per, blobs, pred, gap):
    """One frame's 3-D measurement, the same for every perception: blobs within a gate of the predicted images, the
    gate widening by 20 px per frame without a measurement (up to 300 px); if that finds nothing, the ball is
    re-acquired from the largest blob in each camera."""
    X = per.locate(blobs, pred, min(40.0 + 20.0 * gap, 300.0)) if pred is not None else None
    if X is None:
        X = per.locate(blobs, None)
    return X


def calibrate_pinhole(cam, rng):
    """The camera's own OpenCV model fitted through the tank to a 3-D target in its middle (r <= 80 mm)."""
    from scipy.optimize import least_squares
    n = 3000
    r = 0.08 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(-np.pi, np.pi, n)
    X = np.c_[arm.TANK_XY[0] + r * np.cos(a), arm.TANK_XY[1] + r * np.sin(a), rng.uniform(0.14, 0.32, n)]
    im = rc.project(X, cam, TANK)
    one = np.flatnonzero(im.counts() == 1)
    rows = np.array([im.rows_of(i)[0] for i in one])
    keep = im.in_image[rows]
    X, uv = X[one[keep]], im.uv[rows[keep]]
    uv = uv + rng.normal(0, 0.1, uv.shape)

    def unpack(p):
        th = np.linalg.norm(p[9:12])
        if th > 0:
            k = p[9:12] / th
            Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
        else:
            dR = np.eye(3)
        return rc.Camera([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dR @ cam.R, p[12:15], p[4:9], SIZE)

    def res(p):
        u, _ = unpack(p).project_points(X)
        return np.nan_to_num((u - uv).ravel(), nan=1e3)

    p0 = np.r_[cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2], np.zeros(5), np.zeros(3), cam.t]
    sol = least_squares(res, p0, x_scale="jac", method="lm", max_nfev=3000)
    return unpack(sol.x), float(np.sqrt(np.mean(sol.fun ** 2)))


# ---- one trial ----------------------------------------------------------------------------------------------------

def ball_path(seed):
    """Even seeds stay in the open water (40-120 mm from the axis), odd seeds swim out to 136-160 mm, where the wall
    shows a point twice or hides it.  The ball spirals out from the axis over 4 s, then circles."""
    rng = np.random.default_rng(seed)
    r_max = rng.uniform(0.04, 0.12) if seed % 2 == 0 else rng.uniform(0.136, 0.160)
    th0, om = rng.uniform(-np.pi, np.pi), rng.choice([-1, 1]) * rng.uniform(0.15, 0.30)
    z0, ps = rng.uniform(0.21, 0.26), rng.uniform(0, 2 * np.pi)

    def at(t):
        s = np.clip(t / 4.0, 0, 1)
        r = 0.02 + (r_max - 0.02) * s * s * (3 - 2 * s)
        th = th0 + om * t
        return np.array([arm.TANK_XY[0] + r * np.cos(th), arm.TANK_XY[1] + r * np.sin(th),
                         z0 + 0.03 * np.sin(0.5 * t + ps)])
    return at, r_max


R_REACH = arm.R_IN - 0.010                # the tool centre point stays this close to the axis, clear of the wall
COMMIT = 0.030                           # from this far, the tool stops listening to the cameras (its own fingers
                                         # would hide the ball) and closes on the filter's prediction
LEAD = 0.10                              # the arm's servo lags its goal by about this much (s): the goal leads


def _reachable(p):
    """The point, moved towards the axis if the tool centre point could not get there without touching the wall."""
    q = np.array(p, float)
    rel = q[:2] - arm.TANK_XY
    r = np.hypot(*rel)
    if r > R_REACH:
        q[:2] = arm.TANK_XY + rel * R_REACH / r
    return q


def run_trial(args):
    seed, kind, record = args
    import mujoco
    m = arm.build_model()
    d = mujoco.MjData(m)
    ik = arm.IK(m)
    q_park, _ = ik.park()
    d.qpos[:7], d.qpos[7:9] = q_park, arm.OPEN
    d.ctrl[:7], d.ctrl[7:9] = q_park, arm.OPEN
    mujoco.mj_forward(m, d)
    tank_ids = arm.tank_geoms(m)
    cams = cameras()
    if kind == "refractive":
        per = Refractive(cams)
    elif kind == "afrt":
        per = AFRTPerception(cams)
    else:
        fits = json.loads((CACHE / "pinhole_fits.json").read_text())
        per = Pinhole([rc.Camera(f["K"], f["R"], f["t"], f["dist"], SIZE) for f in fits])
    path, r_max = ball_path(seed)
    kf = Kalman()
    steps = int(round(DT_CTRL / m.opt.timestep))
    tcp = d.site("tcp")
    tool_rot = tcp.xmat.reshape(3, 3).copy()
    goal = tcp.xpos.copy()
    state, t_state, result, committed = "track", 0.0, None, False
    errs, frames, wall = [], [], 0
    seen = np.zeros(4, int)                          # frames in which the ball shows 0, 1, 2, 3+ images in a camera
    q_target = q_park.copy()
    gap = 0
    close_err = None
    for k in range(int(14.0 / DT_CTRL)):
        t = k * DT_CTRL
        ball = path(t) if result is None or not result["caught"] else tcp.xpos + tcp.xmat.reshape(3, 3) @ result["offset"]
        d.mocap_pos[0] = ball
        mujoco.mj_forward(m, d)
        objs = inside_objects(ball, arm.tool_parts(m, d))
        blobs = [detect(c, objs) for c in cams]
        if result is None:
            for c in cams:
                seen[min(int(rc.project(ball, c, TANK).in_image.sum()), 3)] += 1
        pred = kf.predict(DT_CTRL)
        X = measure(per, blobs, pred, gap) if not committed else None
        gap = 0 if X is not None else gap + 1
        if X is not None:
            kf.update(X)
        est = kf.x[:3].copy() if kf.x is not None else None
        if est is not None and state in ("track", "approach", "descend"):     # while the cameras are used
            errs.append((t, float(np.linalg.norm(est - ball)), float(np.hypot(*(ball[:2] - arm.TANK_XY)))))
        # the tool
        if kf.x is not None and state != "done":
            if state == "track" and t >= 4.5:                   # the ball has reached its circle
                state, t_state = "approach", t
            if state == "approach":                               # over the ball, at the parking depth
                aim = _reachable(kf.ahead(1.0))
                aim[2] = arm.PARK_Z
                if np.linalg.norm(goal[:2] - aim[:2]) < 0.01 and t - t_state > 0.5:
                    state, t_state = "descend", t
            elif state == "descend":
                aim = _reachable(kf.ahead(0.3))
                if np.linalg.norm(tcp.xpos - aim) < COMMIT or t - t_state > 4.0:
                    committed = True                              # from here on the cameras are not used
                    state, t_state = "final", t
            elif state == "final":                                # onto the predicted ball
                aim = _reachable(kf.ahead(LEAD))
                if np.linalg.norm(tcp.xpos - _reachable(kf.x[:3])) < 0.004 or t - t_state > 0.8:
                    state, t_state = "close", t
                    close_err = 1e3 * float(np.linalg.norm(kf.x[:3] - ball))
            elif state == "close":                                # the fingers close in about 0.05 s
                aim = _reachable(kf.ahead(LEAD))
                if t - t_state >= DT_CTRL - 1e-9:
                    R = tcp.xmat.reshape(3, 3)
                    off = R.T @ (ball - tcp.xpos)
                    between = (abs(off[1]) <= arm.OPEN - arm.BALL_R and abs(off[0]) <= arm.FINGER[0] + 0.002
                               and abs(off[2]) <= arm.FINGER[2] - arm.BALL_R)
                    result = {"caught": bool(between and wall == 0), "between_fingers": bool(between),
                              "offset_mm": (1e3 * off).tolist(), "offset": off, "t": t,
                              "r_mm": 1e3 * float(np.hypot(*(ball[:2] - arm.TANK_XY))), "z_mm": 1e3 * float(ball[2]),
                              "aim_err_mm": close_err}
                    state, t_state = "lift", t
            if state == "lift":
                aim = goal.copy()
                aim[2] = 0.43                                     # out of the water, still inside the tank
            if state in ("approach", "descend", "final", "close", "lift"):
                speed = {"approach": 0.25, "descend": 0.10, "final": 0.15, "close": 0.15, "lift": 0.12}[state]
                step = aim - goal
                n = np.linalg.norm(step)
                goal = goal + (step if n <= speed * DT_CTRL else step / n * speed * DT_CTRL)
                rel = goal[:2] - arm.TANK_XY
                if np.hypot(*rel) > R_REACH:                     # keep the tool clear of the wall
                    goal[:2] = arm.TANK_XY + rel * R_REACH / np.hypot(*rel)
                az = np.arctan2(rel[1], rel[0])
                if state in ("approach", "descend", "final"):
                    tool_rot = arm.nearest_grip(az, tcp.xmat.reshape(3, 3))
                q_target, _ = ik.solve(d.qpos, goal, tool_rot)
        d.ctrl[:7] = q_target
        d.ctrl[7:9] = 0.0 if state in ("close", "lift") else arm.OPEN
        if record:
            frames.append({"t": t, "qpos": d.qpos.copy(), "ball": ball.copy(), "blobs": [b.copy() for b in blobs],
                           "est": None if est is None else est.copy(), "state": state, "committed": committed})
        for _ in range(steps):
            mujoco.mj_step(m, d)
            wall += arm.tank_contacts(m, d, tank_ids)
        if state == "lift" and t - t_state > 2.5:
            break
    if result is None:
        result = {"caught": False, "between_fingers": False, "offset_mm": None, "t": None, "r_mm": None,
                  "z_mm": None, "aim_err_mm": None}
    result.pop("offset", None)
    result.update({"seed": seed, "kind": kind, "r_max_mm": 1e3 * r_max, "near_wall": bool(seed % 2),
                   "wall_contacts": int(wall), "images_seen": seen.tolist(),
                   "track_err_mm": {"median": 1e3 * float(np.median([e[1] for e in errs])) if errs else None,
                                    "max": 1e3 * float(np.max([e[1] for e in errs])) if errs else None}})
    if record:
        np.save(CACHE / f"frames_{kind}_{seed}.npy", np.array(frames, dtype=object), allow_pickle=True)
    return result


# ---- main ---------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-seed", type=int, default=None)
    ap.add_argument("--kinds", default="refractive,afrt,pinhole")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    arm.fetch_panda()
    lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    cams = cameras()
    rng = np.random.default_rng(11)
    fits = [calibrate_pinhole(c, rng) for c in cams]
    (CACHE / "pinhole_fits.json").write_text(json.dumps([{"K": f.K.tolist(), "R": f.R.tolist(), "t": f.t.tolist(),
                                                           "dist": f.dist.tolist()} for f, _ in fits]))
    say("Demo D: pinhole + distortion fits through the tank (target r <= 80 mm): rms "
        + ", ".join(f"{r:.2f}" for _, r in fits) + " px")
    kinds = args.kinds.split(",")
    jobs = [(s, kind, False) for s in range(args.trials) for kind in kinds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run_trial, jobs))
    say(f"{len(jobs)} trials in {time.perf_counter() - t0:.0f} s")
    rpath = HERE / "results.json"
    out = json.loads(rpath.read_text()) if rpath.exists() else {}
    for kind in kinds:
        rr = [r for r in res if r["kind"] == kind]
        caught = np.array([r["caught"] for r in rr])
        near = np.array([r["near_wall"] for r in rr])
        out[kind] = {"trials": rr, "catch_rate": float(caught.mean()),
                     "catch_rate_near_wall": float(caught[near].mean()) if near.any() else None,
                     "catch_rate_centre": float(caught[~near].mean()) if (~near).any() else None,
                     "track_err_median_mm": float(np.median([r["track_err_mm"]["median"] for r in rr
                                                             if r["track_err_mm"]["median"] is not None])),
                     "wall_contact_trials": int(sum(r["wall_contacts"] > 0 for r in rr))}
        aim = np.array([r["aim_err_mm"] if r["aim_err_mm"] is not None else np.nan for r in rr])
        say(f"{kind:10s}: caught {caught.sum()} of {len(rr)} ({100 * caught.mean():.0f} %); in the open water "
            f"{caught[~near].sum()} of {(~near).sum()}, near the wall {caught[near].sum()} of {near.sum()}; "
            f"error of the estimate as the fingers close: median {np.nanmedian(aim):.1f} mm, max {np.nanmax(aim):.1f} mm; "
            f"median tracking error {out[kind]['track_err_median_mm']:.2f} mm; trials touching the tank: "
            f"{out[kind]['wall_contact_trials']}")
    rpath.write_text(json.dumps(out, indent=1))
    with open(HERE / "demo_d_out.txt", "a") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
