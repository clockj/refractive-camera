"""Demo D, parts 2 and 3: tracking the ball where the wall shows it twice or hides it.

In this tank (outer radius 180 mm, water n = 1.33) a camera sees points farther than about 135 mm from the axis
(180 mm / 1.33) in three ways, depending on where they are:
- twice: the band of double images;
- not at all: the wall's shadow;
- once, at the edge of either.

The ball swims there, 140-162 mm from the axis, on 20 random paths of 16 s, and four perceptions track it, each with
its Kalman filter.  There is no arm: what is measured is what each perception knows about the ball.
  * part 2: ours, pinhole + AFRT and pinhole + distortion, each blob's centroid taken as the image of the ball's centre;
  * part 3, the whole ball: ours fits the ball's centre to its blobs, predicted from every image of its surface and
    the Jacobian (ball_image.py).  And where one camera shows the ball twice, that camera alone locates it.

    python examples/demo_d_arm/fold_zone.py --workers 6      # about 30 minutes
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE)]
import refractive_camera as rc  # noqa: E402
from refractive_camera.baselines import afrt  # noqa: E402
import arm  # noqa: E402
import ball_image  # noqa: E402
import demo_d as D  # noqa: E402

DURATION, PATHS = 16.0, 20
KINDS = ("refractive", "refractive_ball", "afrt", "pinhole")
LABELS = {"refractive": "ours, blob centroids", "refractive_ball": "ours, whole ball",
          "afrt": "pinhole + AFRT", "pinhole": "pinhole + distortion"}


class RefractiveBall(D.Refractive):
    """Ours, fitting the ball's centre to its blobs (ball_image.fit) from the centroid triangulation."""
    name = "refractive_ball"

    def locate(self, blobs, pred, gate=40.0):
        X0 = super().locate(blobs, pred, gate)
        if X0 is None:
            return None
        obs = [(c, ball_image.blobs_near(X0, c, D.TANK, b)) for c, b in zip(self.cams, blobs)]
        obs = [(c, b) for c, b in obs if len(b)]
        X, rms = ball_image.fit(obs, X0, D.TANK, D.BALL_RGB)
        return X if np.all(np.isfinite(X)) and rms < 2.0 else X0


def one_camera(cam, b):
    """Where one camera shows the ball as two blobs: the two rays' crossing (raw), then the whole-ball fit to that
    camera's blobs alone; each accepted only if the point reprojects onto both blobs."""
    if len(b) < 2:
        return None, None
    pix = b[np.argsort(-b[:, 2])[:2], :2]
    try:
        X1, _ = rc.triangulate([cam, cam], pix, D.TANK)
    except ValueError:
        return None, None
    if not np.all(np.isfinite(X1)):
        return None, None
    im = rc.project(X1, cam, D.TANK)
    if int(im.in_image.sum()) != 2:
        return None, None                                   # the point does not explain two blobs
    Xf, rms = ball_image.fit([(cam, pix)], X1, D.TANK, D.BALL_RGB)
    return X1, (Xf if np.all(np.isfinite(Xf)) and rms < 1.0 else None)


def path(seed):
    rng = np.random.default_rng(1000 + seed)
    r0, ph_r = rng.uniform(0.146, 0.155), rng.uniform(0, 2 * np.pi)
    th0, om = rng.uniform(-np.pi, np.pi), rng.choice([-1, 1]) * rng.uniform(0.25, 0.40)
    z0, ps = rng.uniform(0.19, 0.26), rng.uniform(0, 2 * np.pi)

    def at(t):
        r = r0 + 0.006 * np.sin(0.6 * t + ph_r)
        th = th0 + om * t
        return np.array([arm.TANK_XY[0] + r * np.cos(th), arm.TANK_XY[1] + r * np.sin(th),
                         z0 + 0.03 * np.sin(0.5 * t + ps)])
    return at


def track(args):
    seed, kind = args
    cams = D.cameras()
    if kind == "refractive":
        per = D.Refractive(cams)
    elif kind == "refractive_ball":
        per = RefractiveBall(cams)
    elif kind == "afrt":
        per = D.AFRTPerception(cams)
    else:
        fits = json.loads((D.CACHE / "pinhole_fits.json").read_text())
        per = D.Pinhole([rc.Camera(f["K"], f["R"], f["t"], f["dist"], D.SIZE) for f in fits])
    at = path(seed)
    kf = D.Kalman()
    rows = []
    gap = 0
    for k in range(int(DURATION / D.DT_CTRL)):
        t = k * D.DT_CTRL
        ball = at(t)
        objs = D.inside_objects(ball, [])
        blobs = [D.detect(c, objs) for c in cams]
        pred = kf.predict(D.DT_CTRL)
        X = D.measure(per, blobs, pred, gap)
        gap = 0 if X is not None else gap + 1
        if X is not None:
            kf.update(X)
        row = {"t": t, "measured": X is not None,
               "err_mm": None if kf.x is None else 1e3 * float(np.linalg.norm(kf.x[:3] - ball)),
               "meas_err_mm": None if X is None else 1e3 * float(np.linalg.norm(X - ball)),
               "r_mm": 1e3 * float(np.hypot(*(ball[:2] - arm.TANK_XY)))}
        if kind == "refractive":                          # what each camera really shows, and what AFRT finds
            true_images, afrt_found = [], []
            for c in cams:
                im = rc.project(ball, c, D.TANK)
                true_images.append(int(im.in_image.sum()))
                d, _, _ = afrt(D.TANK, c.center, ball)
                ok = d is not None
                if ok:
                    tr = rc.trace(D.TANK, c.center, d)
                    ok = bool(tr.ok[0] and tr.miss_distance(ball[None])[0] < 1e-9)
                afrt_found.append(ok)
            row.update(true_images=true_images, afrt_found=afrt_found)
        if kind == "refractive_ball":                     # one camera alone, where it shows two blobs
            one = []
            for ci, (c, b) in enumerate(zip(cams, blobs)):
                if len(b) >= 2:
                    X1, Xf = one_camera(c, b)
                    one.append((ci, None if X1 is None else 1e3 * float(np.linalg.norm(X1 - ball)),
                                None if Xf is None else 1e3 * float(np.linalg.norm(Xf - ball))))
            row["one_camera"] = one
        rows.append(row)
    return {"seed": seed, "kind": kind, "rows": rows}


def summarise(res, say):
    out = {}
    for kind in KINDS:
        runs = [r for r in res if r["kind"] == kind]
        rows = [x for r in runs for x in r["rows"] if x["t"] >= 1.0]          # after the filter has started
        err = np.array([x["err_mm"] for x in rows], float)
        meas = np.array([x["meas_err_mm"] if x["meas_err_mm"] is not None else np.nan for x in rows], float)
        lost = [r["seed"] for r in runs if max(x["err_mm"] or 0 for x in r["rows"] if x["t"] >= 1.0) > 20]
        out[kind] = {"frames": len(rows), "measured": float(np.mean(np.isfinite(meas))),
                     "meas_err_median_mm": float(np.nanmedian(meas)), "meas_err_p95_mm": float(np.nanpercentile(meas, 95)),
                     "err_median_mm": float(np.median(err)), "err_p95_mm": float(np.percentile(err, 95)),
                     "err_max_mm": float(np.max(err)), "frames_over_5mm": float(np.mean(err > 5)),
                     "paths_lost": lost}
        say(f"{LABELS[kind]:22s}: a 3-D measurement in {100 * np.mean(np.isfinite(meas)):5.1f} % of frames, its error "
            f"median {np.nanmedian(meas):.2f} mm (95th pct {np.nanpercentile(meas, 95):.2f}); the filter's error median "
            f"{np.median(err):.2f} mm, 95th pct {np.percentile(err, 95):.2f} mm, max {np.max(err):.1f} mm; paths off by "
            f"more than 20 mm: {len(lost)} of {len(runs)}")
    ref = [x for r in res if r["kind"] == "refractive" for x in r["rows"]]
    ti = np.array([x["true_images"] for x in ref])
    af = np.array([x["afrt_found"] for x in ref])
    say(f"\nWhat the cameras show over all frames (camera-frames {ti.size}): no image {100 * np.mean(ti == 0):.1f} %, "
        f"one {100 * np.mean(ti == 1):.1f} %, two {100 * np.mean(ti == 2):.1f} %")
    say(f"AFRT finds an image in {100 * np.mean(af[ti >= 1]):.1f} % of the camera-frames where the ball is visible; "
        f"both images where it is seen twice: never (one start returns one ray)")
    out["camera_frames"] = {"none": float(np.mean(ti == 0)), "one": float(np.mean(ti == 1)),
                            "two": float(np.mean(ti == 2)), "afrt_finds_visible": float(np.mean(af[ti >= 1]))}
    one = [o for r in res if r["kind"] == "refractive_ball" for x in r["rows"] for o in x["one_camera"]]
    raw = np.array([o[1] if o[1] is not None else np.nan for o in one], float)
    fit = np.array([o[2] if o[2] is not None else np.nan for o in one], float)
    say(f"\nOne camera alone, in the {len(one)} camera-frames where it shows the ball as two blobs:")
    say(f"  the two rays' crossing: a point that explains both blobs in {100 * np.mean(np.isfinite(raw)):.0f} %, error "
        f"median {np.nanmedian(raw):.1f} mm")
    say(f"  then the whole-ball fit: accepted in {100 * np.mean(np.isfinite(fit)):.0f} %, error median "
        f"{np.nanmedian(fit):.2f} mm, 90th pct {np.nanpercentile(fit, 90):.2f} mm; within 1 mm "
        f"{100 * np.nanmean(fit < 1):.0f} % of those")
    say("  pinhole + AFRT: one image per camera, so one ray, not a point; pinhole + distortion: two straight lines "
        "through the camera's centre, which meet only there")
    out["one_camera"] = {"camera_frames": len(one), "raw_accepted": float(np.mean(np.isfinite(raw))),
                         "raw_median_mm": float(np.nanmedian(raw)), "fit_accepted": float(np.mean(np.isfinite(fit))),
                         "fit_median_mm": float(np.nanmedian(fit)), "fit_p90_mm": float(np.nanpercentile(fit, 90)),
                         "fit_within_1mm": float(np.nanmean(fit < 1))}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    D.CACHE.mkdir(exist_ok=True)
    if not (D.CACHE / "pinhole_fits.json").exists():
        raise SystemExit("run demo_d.py first: it calibrates the pinhole cameras")
    lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    t0 = time.perf_counter()
    jobs = [(s, k) for k in ("refractive_ball",) + tuple(k for k in KINDS if k != "refractive_ball")
            for s in range(PATHS)]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(track, jobs))
    say(f"Demo D, parts 2-3: {PATHS} paths of {DURATION:.0f} s, 140-162 mm from the axis; {time.perf_counter() - t0:.0f} s")
    out = summarise(res, say)
    (D.CACHE / "fold_zone_runs.json").write_text(json.dumps(res))
    (HERE / "fold_zone.json").write_text(json.dumps(out, indent=1))
    (HERE / "fold_zone_out.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
