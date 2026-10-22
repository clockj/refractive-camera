"""The robot side of Demo D: a Franka Panda (MuJoCo Menagerie) beside a water-filled acrylic tank, carrying a slim
long-reach gripper, with inverse kinematics for its tool centre point and a scripted target ball.

The Panda's own hand is 208 mm long along its closing direction: in this tank it fits only around points within about
93 mm of the axis.  The slim tool reaches the whole tank: a 14 mm shaft 30 cm long, with two thin fingers at its end
that close along the tank's tangent.  The tank's wall, floor and stones are collision geometry, so that touching them
shows as contacts (and counts as a failed catch)."""
import json
import os
import urllib.request
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ["REFCAM_CACHE"]) / "demo_d_arm" if "REFCAM_CACHE" in os.environ else HERE / "_cache"
PANDA = CACHE / "franka_emika_panda"
MENAGERIE = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/franka_emika_panda"
MODEL_FILES = ("panda_nohand.xml", "LICENSE", "README.md")

# The tank: acrylic cylinder on a stand, axis vertical; lengths in metres
TANK_XY = np.array([0.55, 0.0])
BASE_Z = 0.25                            # the arm stands on a pedestal beside the tank
R_OUT, R_IN = 0.18, 0.172
Z_FLOOR, Z_WATER, Z_RIM = 0.10, 0.40, 0.44
BALL_R = 0.008
# Stones on the tank's floor (offset from the axis, radius, colour); half sunk in the sand
STONES = [((0.075, 0.065), 0.026, (0.45, 0.55, 0.45)), ((-0.095, 0.035), 0.021, (0.45, 0.50, 0.62)),
          ((0.020, -0.105), 0.030, (0.70, 0.64, 0.52)), ((-0.055, -0.080), 0.019, (0.36, 0.36, 0.40)),
          ((0.115, -0.035), 0.023, (0.55, 0.58, 0.35))]

# The slim tool, in the flange frame (z away from the arm); lengths from the flange
ADAPTER = (0.032, 0.008)                 # radius, half-height of the plate on the flange
SHAFT_R, SHAFT_TOP, SHAFT_BOTTOM = 0.007, 0.016, 0.316
WRIST = (0.008, 0.024, 0.008)            # half-sizes of the block that carries the fingers
FINGER = (0.007, 0.0025, 0.025)          # finger half-sizes: across, thickness (along the closing axis), length
FINGER_TOP = SHAFT_BOTTOM + 2 * WRIST[2]
TCP_OFFSET = FINGER_TOP + FINGER[2]      # the grasp point, halfway down the fingers: 0.357 m from the flange
OPEN = 0.020                             # each finger's inner face from the centre line, open (a 40 mm gap)
PARK_Z = 0.33                            # the tool waits in the water above the ball's depths, on the axis
TOOL_GEOMS = ("shaft", "wrist", "left_pad", "right_pad")


def stone_centres():
    return [np.array([TANK_XY[0] + dx, TANK_XY[1] + dy, Z_FLOOR + 0.004 + 0.4 * r]) for (dx, dy), r, _ in STONES]


def fetch_panda():
    """Download the Panda model without its hand (MuJoCo Menagerie, Apache-2.0) into the cache, once."""
    if all((PANDA / f).exists() for f in MODEL_FILES):
        return PANDA
    (PANDA / "assets").mkdir(parents=True, exist_ok=True)
    for sub in ("", "/assets"):
        for it in json.load(urllib.request.urlopen(MENAGERIE + sub)):
            if it["type"] != "file" or (sub == "" and it["name"] not in MODEL_FILES):
                continue
            dst = PANDA / sub.strip("/") / it["name"]
            if not dst.exists():
                urllib.request.urlretrieve(it["download_url"], dst)
    return PANDA


def _add_tool(spec):
    """The slim gripper on the flange (Menagerie's attachment body): plate, shaft, wrist block, two fingers on
    slides, the tool centre point between the finger pads."""
    G = mujoco.mjtGeom
    dark, metal, pad = [0.18, 0.18, 0.2, 1], [0.70, 0.72, 0.76, 1], [0.93, 0.93, 0.95, 1]
    tool = spec.body("attachment").add_body(name="tool")
    tool.add_geom(name="plate", type=G.mjGEOM_CYLINDER, pos=[0, 0, ADAPTER[1]], size=[ADAPTER[0], ADAPTER[1], 0],
                  rgba=dark, mass=0.1)
    tool.add_geom(name="shaft", type=G.mjGEOM_CYLINDER, pos=[0, 0, 0.5 * (SHAFT_TOP + SHAFT_BOTTOM)],
                  size=[SHAFT_R, 0.5 * (SHAFT_BOTTOM - SHAFT_TOP), 0], rgba=metal, mass=0.12)
    tool.add_geom(name="wrist", type=G.mjGEOM_BOX, pos=[0, 0, SHAFT_BOTTOM + WRIST[2]], size=list(WRIST), rgba=dark,
                  mass=0.05)
    tool.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0], rgba=[0, 1, 0, 0])
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        f = tool.add_body(name=f"{side}_finger", pos=[0, 0, FINGER_TOP])
        j = f.add_joint(name=f"{side}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, sgn, 0], range=[0, OPEN],
                        damping=2.0, armature=0.01)
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        f.add_geom(name=f"{side}_pad", type=G.mjGEOM_BOX, pos=[0, sgn * FINGER[1], FINGER[2]], size=list(FINGER),
                   rgba=pad, mass=0.02)
        a = spec.add_actuator(name=f"{side}_finger", target=f"{side}_slide", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=300.0, kv=8.0)
        a.ctrlrange = [0.0, OPEN]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return tool


def build_model(ghost=False):
    """The scene; with `ghost`, a translucent second ball (a mocap body) marks where a perception puts the ball."""
    spec = mujoco.MjSpec.from_file(str(fetch_panda() / "panda_nohand.xml"))
    for k in list(spec.keys):
        spec.delete(k)                                         # the arm is parked by inverse kinematics instead
    spec.body("link0").pos = [0.0, 0.0, BASE_Z]
    G = mujoco.mjtGeom
    wb = spec.worldbody
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.25, 0.25, 0.27]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.37]
    wb.add_light(pos=[0.6, -0.5, 1.8], dir=[-0.05, 0.25, -1], diffuse=[0.7, 0.7, 0.68], castshadow=True)
    wb.add_light(pos=[-0.6, 0.8, 1.2], dir=[0.6, -0.6, -0.6], diffuse=[0.25, 0.27, 0.32], castshadow=False)
    grid = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                            rgb1=[0.86, 0.86, 0.84], rgb2=[0.80, 0.80, 0.78], width=512, height=512)
    del grid
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.08)
    wb.add_geom(type=G.mjGEOM_PLANE, size=[2, 2, 0.05], material="floor", contype=0, conaffinity=0)
    cx, cy = TANK_XY
    vis = dict(contype=0, conaffinity=0)
    wb.add_geom(type=G.mjGEOM_CYLINDER, pos=[0, 0, BASE_Z / 2], size=[0.11, BASE_Z / 2, 0], rgba=[0.30, 0.31, 0.34, 1],
                **vis)                                                                 # the pedestal
    wb.add_geom(type=G.mjGEOM_BOX, pos=[cx, cy, Z_FLOOR / 2], size=[0.22, 0.22, Z_FLOOR / 2],
                rgba=[0.30, 0.31, 0.34, 1], **vis)
    wb.add_geom(name="tank_floor", type=G.mjGEOM_CYLINDER, pos=[cx, cy, Z_FLOOR + 0.002], size=[R_IN, 0.002, 0],
                rgba=[0.80, 0.74, 0.60, 1])                                            # sand on the floor
    wb.add_geom(type=G.mjGEOM_CYLINDER, pos=[cx, cy, (Z_FLOOR + Z_WATER) / 2], size=[R_IN, (Z_WATER - Z_FLOOR) / 2, 0],
                rgba=[0.30, 0.58, 0.92, 0.20], **vis)                                  # water
    wb.add_geom(type=G.mjGEOM_CYLINDER, pos=[cx, cy, (Z_FLOOR + Z_RIM) / 2], size=[R_OUT, (Z_RIM - Z_FLOOR) / 2, 0],
                rgba=[0.85, 0.93, 1.0, 0.13], **vis)                                   # acrylic wall
    wb.add_geom(type=G.mjGEOM_CYLINDER, pos=[cx, cy, Z_RIM - 0.002], size=[R_OUT + 0.002, 0.002, 0],
                rgba=[0.75, 0.85, 0.95, 0.35], **vis)                                  # the rim, for the eye
    # the wall as collision geometry: a ring of thin boxes between the two radii (invisible)
    n_seg = 72
    r_mid, half_t = 0.5 * (R_IN + R_OUT), 0.5 * (R_OUT - R_IN)
    for k in range(n_seg):
        th = 2 * np.pi * k / n_seg
        wb.add_geom(name=f"tank_wall_{k}", type=G.mjGEOM_BOX, group=3, rgba=[0, 0, 0, 0],
                    pos=[cx + r_mid * np.cos(th), cy + r_mid * np.sin(th), 0.5 * (Z_FLOOR + Z_RIM)],
                    size=[half_t, 1.08 * r_mid * np.pi / n_seg, 0.5 * (Z_RIM - Z_FLOOR)],
                    quat=[np.cos(th / 2), 0, 0, np.sin(th / 2)])
    for k, (c, ((_, _), r, col)) in enumerate(zip(stone_centres(), STONES)):
        wb.add_geom(name=f"tank_stone_{k}", type=G.mjGEOM_SPHERE, pos=list(c), size=[r, 0, 0], rgba=list(col) + [1])
    ball = wb.add_body(name="ball", mocap=True, pos=[cx, cy, 0.25])
    ball.add_geom(type=G.mjGEOM_SPHERE, size=[BALL_R, 0, 0], rgba=[1.0, 0.45, 0.05, 1], **vis)
    if ghost:
        est = wb.add_body(name="estimate", mocap=True, pos=[cx, cy, 0.25])
        est.add_geom(type=G.mjGEOM_SPHERE, size=[2.0 * BALL_R, 0, 0], rgba=[0.2, 0.9, 0.3, 0.3], **vis)
    tex = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.92, 0.94, 0.97],
                           rgb2=[0.42, 0.47, 0.56], width=256, height=256)
    del tex
    _add_tool(spec)
    for name in ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "tool", "left_finger", "right_finger"):
        spec.body(name).gravcomp = 1.0                 # the arm's controller compensates gravity, as a real one does
    wb.add_camera(name="overview", pos=[1.05, -0.62, 0.78], xyaxes=look_at_axes([1.05, -0.62, 0.78], [0.42, 0.0, 0.33]))
    return spec.compile()


def look_at_axes(pos, target):
    """MuJoCo camera xyaxes (right, up) for a camera at pos looking at target, world z up."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return list(right) + list(np.cross(right, f))


def tank_geoms(m):
    """Ids of the tank's collision geometry: wall, floor, stones."""
    return {i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("tank_")}


def tank_contacts(m, d, tank_ids):
    """How many contacts the robot makes with the tank (nothing else in the scene collides)."""
    return sum(1 for c in d.contact[:d.ncon] if (c.geom1 in tank_ids) != (c.geom2 in tank_ids))


def tool_parts(m, d):
    """World pose of the tool's parts in the water's reach: (name, centre, rotation, size) per geom."""
    out = []
    for name in TOOL_GEOMS:
        g = m.geom(name)
        out.append((name, d.geom(name).xpos.copy(), d.geom(name).xmat.reshape(3, 3).copy(), g.size.copy()))
    return out


class IK:
    """Damped least squares for the tool centre point's position and orientation (fingers down)."""

    def __init__(self, m):
        self.m = m
        self.d = mujoco.MjData(m)
        self.site = m.site("tcp").id
        self.lo, self.hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]

    def solve(self, q0, pos, R_target, iters=60, damping=1e-3, tol=1e-6):
        m, d = self.m, self.d
        q = np.array(q0[:7], float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            e = np.r_[pos - p, 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))]
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, np.linalg.norm(e[:3])

    def park(self, pos=None):
        """Joint angles with the tool centre point on the tank's axis, fingers down, in the water at PARK_Z."""
        pos = np.r_[TANK_XY, PARK_Z] if pos is None else pos
        q = np.array([0.0, 0.2, 0.0, -1.9, 0.0, 2.1, 0.785])
        for _ in range(3):
            q, err = self.solve(q, pos, fingers_down(np.pi / 2), iters=200)
        return q, err


def fingers_down(azimuth):
    """Tool orientation with the fingers pointing down and closing along the tank's tangent at `azimuth`."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def nearest_grip(azimuth, R_now):
    """The two fingers are alike, so a turn by pi about the tool's axis grips the same way: the orientation of the two
    closer to the current one."""
    a, b = fingers_down(azimuth), fingers_down(azimuth + np.pi)
    return a if np.trace(R_now.T @ a) >= np.trace(R_now.T @ b) else b
