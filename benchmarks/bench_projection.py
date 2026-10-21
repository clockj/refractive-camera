"""Benchmark: forward projection through the paper's reference port, by our search and by AFRT.

At the reference port of S. Zhong et al., arXiv:2610.17413 (an acrylic tube of outer radius 100 mm and 6 mm wall,
filled with water and seen from air, the camera 350 mm from the axis), and at the same port made spherical, this
measures how long one projection takes and whether it returns every image of the point:

  * ours, compiled  - refractive_camera.project(backend="cpp", threads=1): the complete search;
  * ours, Python    - the same search in pure Python (backend="python"), on 40 points per set;
  * AFRT            - alternating forward ray tracing (Belden 2013) from one start, the straight line, as published.

Point sets: uniform in the water; steep rays (points far above or below the camera); the two-image band; next to the
edge of the shadow (the image count changes within 0.1 mm radially); near the axis (within 2 mm of the cylinder's
axis, or of the sphere's line through its centre and the camera).

The truth at a point is every ray found by our search or by a brute-force search that uses tracing alone
(refractive_camera.tracing.search_images), kept only if its traced path passes within 1e-9 r_P of the point; a ray of
AFRT that lands within its own tolerance (1e-9 m, also the paper's acceptance) but matches none of them would be added
and listed.  A returned ray is an image when its direction is within 1e-7 rad of a true ray's (2e-4 px at f = 2000
px); each method's answer is then scored: every image, some images, no answer (nothing at a point that has images), a
false image (a returned ray that is no image), or a correct empty answer.  How precisely each method's rays land is
reported apart from that.

Timings are per point, each call timed on its own with time.perf_counter: the compiled search through the public
project() (so including its Python wrapper) and, directly, the compiled core alone; AFRT; each the best of 3 calls.  The
Python search is timed once per point.  The batch throughput of the compiled backend, wrapper amortised, is given too,
and so is how busy the machine was (whole-machine CPU use, sampled at each set), since other load inflates timings.

    python benchmarks/bench_projection.py            about 15 minutes
    python benchmarks/bench_projection.py --quick    a minute, small sets (writes *_quick.* files)
    python benchmarks/bench_projection.py --redraw   the figure again, from the saved JSON
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import refractive_camera as rc  # noqa: E402
from refractive_camera.baselines import afrt  # noqa: E402
from refractive_camera.tracing import search_images  # noqa: E402

RADII, INDICES = (0.100, 0.094), (1.0, 1.49, 1.33)
CENTRE = np.array([0.35, 0.0, 0.0])
SETS = ("uniform", "steep", "band", "edge", "axis")
OUTCOMES = ("every image", "some images", "no answer", "false image", "correct empty")
TRUTH_TOL = 1e-9                # a ray enters the truth if its trace passes within this of the point, relative to r_P
ACCEPT_M = 1e-9                 # m: AFRT's stopping tolerance and the paper's acceptance (E1)
SAME_RAY = 1e-7                 # rad: a returned ray this close in direction to a true ray is that image
GRID = (600, 400)               # brute-force search grid
PY_POINTS = 40                  # points per set for the Python backend

QUICK = "--quick" in sys.argv
SIZES = dict(uniform=500, steep=100, band=100, edge=100, axis=100)
TRUTH_SIZES = dict(uniform=150, steep=100, band=100, edge=100, axis=100)
if QUICK:
    SIZES = {k: 12 for k in SIZES}
    TRUTH_SIZES = {k: 6 for k in TRUTH_SIZES}
    PY_POINTS = 4
SUFFIX = "_quick" if QUICK else ""

_log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log.append(s)


def cpu_name():
    try:
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def cpu_load():
    """Whole-machine CPU use (%) now, or None: on Windows the processors' load, elsewhere the 1-minute load average."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | "
                                  "Measure-Object -Property LoadPercentage -Average).Average"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            return float(out)
        if hasattr(os, "getloadavg"):
            return 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1)
    except Exception:
        pass
    return None


# ------------------------------------------------------------------------------------------------- the point sets
def point_sets(window, cam, seed):
    """The five sets, deterministic; band and edge points are drawn from a pool by the compiled backend's counts."""
    rng = np.random.default_rng(seed)
    R = window.radii[-1]
    sphere = window.kind == "sphere"

    def disc(m, zlo, zhi):
        r = R * np.sqrt(rng.uniform(0, 1, m)) * (1 - 1e-9)
        th = rng.uniform(-np.pi, np.pi, m)
        z = rng.uniform(zlo, zhi, m) * rng.choice([-1.0, 1.0], m)
        if sphere:                                      # keep the point inside the sphere
            z = np.sign(z) * np.minimum(np.abs(z), np.sqrt(np.maximum(R * R - r * r, 0)) * 0.999)
        return np.c_[r * np.cos(th), r * np.sin(th), z]

    def counts(X):
        return rc.project(X, cam, window, backend="cpp", threads=0).counts(on_sensor=False)

    sets = {}
    if sphere:
        v = rng.normal(size=(SIZES["uniform"], 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sets["uniform"] = v * (R * rng.uniform(0, 1, len(v)) ** (1 / 3) * (1 - 1e-9))[:, None]
        sets["steep"] = disc(SIZES["steep"], 0.05, 0.09)
    else:
        sets["uniform"] = disc(SIZES["uniform"], 0.0, 0.1)
        sets["steep"] = disc(SIZES["steep"], 0.2, 0.5)
    pool = disc(100000, 0.0, 0.03)
    c = counts(pool)
    sets["band"] = pool[c == 2][:SIZES["band"]]
    r = np.hypot(pool[:, 0], pool[:, 1])
    inner = pool * ((r - 1e-4) / r)[:, None]
    inner[:, 2] = pool[:, 2]
    sets["edge"] = pool[c != counts(inner)][:SIZES["edge"]]
    m = SIZES["axis"]
    rr, ph = 0.002 * np.sqrt(rng.uniform(0, 1, m)), rng.uniform(-np.pi, np.pi, m)
    if sphere:                                          # near the line through the centre and the camera (x axis)
        x = rng.uniform(-1, 1, m) * np.sqrt(R * R - rr * rr) * 0.999
        sets["axis"] = np.c_[x, rr * np.cos(ph), rr * np.sin(ph)]
    else:
        sets["axis"] = np.c_[rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-0.1, 0.1, m)]
    return sets


# ------------------------------------------------------------------------------------------------------ methods
def ours(window, cam, backend):
    def f(x):
        im = rc.project(x, cam, window, backend=backend, threads=1)
        rays = im.branch != "ring"
        return im.direction[rays], str(im.status[0]), int((~rays).sum())
    return f


def core_only(window, cam):
    """The compiled core called directly on one point, in the window's frame: the search without project()'s Python."""
    import refractive_camera.projection as rp
    f = rp._core.project_cylinder if window.kind == "cylinder" else rp._core.project_sphere
    n, R = list(window.indices), list(window.radii)
    P0 = window.to_local(cam.center)

    def g(x):
        return f(n, R, P0, window.to_local(np.asarray(x, float)[None]), 1e-9, 1)
    return g


def afrt_method(window):
    def f(x):
        d, status, it = afrt(window, CENTRE, x)
        return (np.zeros((0, 3)) if d is None else d[None, :]), status, it
    return f


def timed(f, X, repeat):
    times, out = np.empty(len(X)), []
    for i, x in enumerate(X):
        best, res = np.inf, None
        for _ in range(repeat):
            t0 = time.perf_counter()
            res = f(x)
            best = min(best, time.perf_counter() - t0)
        times[i] = best
        out.append(res)
    return times, out


# ------------------------------------------------------------------------------------------------------ scoring
def landing(window, dirs, x):
    """Distance from the point to each direction's ray, traced from the camera centre (inf where it misses)."""
    if len(dirs) == 0:
        return np.zeros(0)
    tr = rc.trace(window, np.repeat(CENTRE[None], len(dirs), 0), dirs)
    return tr.miss_distance(np.repeat(np.asarray(x)[None], len(dirs), 0))


def dedupe(dirs):
    out = []
    for d in dirs:
        if not any(np.linalg.norm(d - e) < SAME_RAY for e in out):
            out.append(d)
    return np.array(out).reshape(-1, 3)


def matches(dirs, truth):
    """For each direction, whether it is one of the true rays (within SAME_RAY)."""
    if len(truth) == 0:
        return np.zeros(len(dirs), bool)
    return np.array([np.min(np.linalg.norm(truth - d, axis=1)) < SAME_RAY for d in dirs], bool)


def score(dirs, truth):
    """One method's answer at one point against the truth: its outcome."""
    if len(dirs) == 0:
        return "correct empty" if len(truth) == 0 else "no answer"
    if not matches(dirs, truth).all():
        return "false image"
    found = int(matches(truth, dirs).sum())
    return "every image" if found == len(truth) else "some images"


def pct(outcomes):
    n = max(len(outcomes), 1)
    return {o: 100.0 * sum(v == o for v in outcomes) / n for o in OUTCOMES}


def precision(m):
    """Median and largest landing distance (m) of the true images a method returned."""
    m = np.asarray(m, float)
    return {"median": float(np.median(m)) if m.size else None, "max": float(np.max(m)) if m.size else None,
            "n": int(m.size)}


def stats_us(t):
    return {"median_us": float(np.median(t) * 1e6), "p95_us": float(np.percentile(t, 95) * 1e6), "n": int(len(t))}


# --------------------------------------------------------------------------------------------------------- main
def run_window(window, cam, seed):
    name = window.kind
    say(f"\n=== {name}: radii {window.radii} m, indices {window.indices}, camera centre {CENTRE.tolist()} m ===")
    t0 = time.perf_counter()
    sets = point_sets(window, cam, seed)
    say("point sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    res = {"sets": {}}
    all_points = np.vstack([sets[k] for k in SETS])
    for threads, label in ((1, "one thread"), (0, "all cores")):
        best = np.inf
        for _ in range(3):
            t = time.perf_counter()
            rc.project(all_points, cam, window, backend="cpp", threads=threads)
            best = min(best, time.perf_counter() - t)
        res[f"batch_us_per_point_{'1' if threads == 1 else 'all'}_threads"] = 1e6 * best / len(all_points)
        say(f"compiled backend, batch of {len(all_points)} points, {label}: {1e6 * best / len(all_points):.1f} us/point")
    disagreements, loads = [], []
    for key in SETS:
        X = sets[key]
        loads.append(cpu_load())
        t_cpp, out_cpp = timed(ours(window, cam, "cpp"), X, 3)
        t_core, _ = timed(core_only(window, cam), X, 3)
        t_afrt, out_afrt = timed(afrt_method(window), X, 3)
        npy = min(PY_POINTS, len(X))
        t_py, out_py = timed(ours(window, cam, "python"), X[:npy], 1)
        py_same = sum(len(a[0]) == len(b[0]) and all(np.min(np.linalg.norm(b[0] - d, axis=1)) < SAME_RAY
                                                     for d in a[0])
                      for a, b in zip(out_cpp[:npy], out_py))
        capped = sum(o[1] == "capped" for o in out_cpp)
        rings = sum(o[2] for o in out_cpp)
        # the truth, on the scored subset
        nt = min(TRUTH_SIZES[key], len(X))
        outcomes = {"ours": [], "afrt": []}
        miss = {"ours": [], "afrt": []}
        n_true, afrt_status = [], []
        rP = window.radial_distance(CENTRE)
        for i in range(nt):
            x = X[i]
            cands = [c for c in (out_cpp[i][0], search_images(window, CENTRE, x, grid=GRID)) if len(c)]
            cands = np.vstack(cands) if cands else np.zeros((0, 3))
            truth = dedupe(cands[landing(window, cands, x) <= TRUTH_TOL * rP]) if len(cands) else np.zeros((0, 3))
            for d, m in zip(out_afrt[i][0], landing(window, out_afrt[i][0], x)):
                if m <= ACCEPT_M and not matches(d[None], truth)[0]:
                    truth = np.vstack([truth, d[None]])
                    disagreements.append(dict(window=name, set=key, point=x.tolist(), note="AFRT found a ray that "
                                              "neither our search nor the brute force did", landing_m=float(m)))
            n_true.append(len(truth))
            for method, dirs in (("ours", out_cpp[i][0]), ("afrt", out_afrt[i][0])):
                outcomes[method].append(score(dirs, truth))
                ok = matches(dirs, truth)
                if ok.any():
                    miss[method] += landing(window, dirs[ok], x).tolist()
            afrt_status.append(out_afrt[i][1])
            if outcomes["ours"][-1] not in ("every image", "correct empty"):
                disagreements.append(dict(window=name, set=key, point=x.tolist(), ours=len(out_cpp[i][0]),
                                          truth=len(truth), outcome=outcomes["ours"][-1]))
        res["sets"][key] = {
            "points": int(len(X)), "scored": int(nt),
            "true_image_counts": {str(k): int(v) for k, v in zip(*np.unique(n_true, return_counts=True))},
            "ours": {"outcomes_pct": pct(outcomes["ours"]), "time": stats_us(t_cpp), "capped": int(capped),
                     "rings": int(rings), "landing_m": precision(miss["ours"])},
            "ours_core": {"time": stats_us(t_core)},
            "ours_python": {"time": stats_us(t_py), "same_as_compiled": int(py_same), "of": int(npy)},
            "afrt": {"outcomes_pct": pct(outcomes["afrt"]), "time": stats_us(t_afrt),
                     "landing_m": precision(miss["afrt"]),
                     "status_on_scored": {s: afrt_status.count(s) for s in ("converged", "cap", "breakdown")}},
        }
        say(f"  {key:8s} done ({len(X)} points timed, {nt} scored against the truth; {time.perf_counter() - t0:.0f} s)")
    loads.append(cpu_load())
    known = [v for v in loads if v is not None]
    res["machine_cpu_load_pct"] = {"samples": loads, "mean": float(np.mean(known)) if known else None,
                                   "max": float(np.max(known)) if known else None}
    res["disagreements_with_truth"] = disagreements
    res["seconds"] = time.perf_counter() - t0
    return res


def table(name, res):
    say(f"\n{name}: outcome on the points scored against the truth, percent of points")
    head = "  ".join(f"{o.split()[0][:5]:>5s}" for o in OUTCOMES)
    say(f"  {'set':8s} {'pts':>4s} | {'ours (compiled)':^36s} | {'AFRT, one start':^36s} | true images")
    say(f"  {'':8s} {'':>4s} | {head} | {head} |")
    for key in SETS:
        s = res["sets"][key]
        a = "  ".join(f"{s['ours']['outcomes_pct'][o]:5.1f}" for o in OUTCOMES)
        b = "  ".join(f"{s['afrt']['outcomes_pct'][o]:5.1f}" for o in OUTCOMES)
        counts = ", ".join(f"{k}: {v}" for k, v in s["true_image_counts"].items())
        say(f"  {key:8s} {s['scored']:4d} | {a} | {b} | {counts}")
    say(f"  (columns: {', '.join(OUTCOMES)})")
    worst, med = {}, {}
    for m in ("ours", "afrt"):
        lm = [res["sets"][k][m]["landing_m"] for k in SETS]
        worst[m] = max((v["max"] or 0.0) for v in lm)
        meds = [v["median"] for v in lm if v["median"] is not None]
        med[m] = float(np.median(meds)) if meds else float("nan")
    say(f"  precision of the true images returned (distance from the point to the traced ray): ours median "
        f"{med['ours']:.1e} m, largest {worst['ours']:.1e} m; AFRT median {med['afrt']:.1e} m, largest "
        f"{worst['afrt']:.1e} m")
    load = res["machine_cpu_load_pct"]
    busy = "" if load["mean"] is None else f"; machine CPU use during the run {load['mean']:.0f} % mean, {load['max']:.0f} % max"
    say(f"\n{name}: time per point, median / p95, single calls{busy}")
    say(f"  {'set':8s} | {'ours, project() (us)':>20s} | {'ours, core (us)':>16s} | {'ours, Python (ms)':>18s} | "
        f"{'AFRT, Python (us)':>18s} | Python = compiled")
    for key in SETS:
        s = res["sets"][key]
        c, k, p, a = s["ours"]["time"], s["ours_core"]["time"], s["ours_python"]["time"], s["afrt"]["time"]
        say(f"  {key:8s} | {c['median_us']:8.0f} / {c['p95_us']:8.0f} | {k['median_us']:6.0f} / {k['p95_us']:6.0f} | "
            f"{p['median_us'] / 1e3:7.2f} / {p['p95_us'] / 1e3:7.2f} | {a['median_us']:7.0f} / {a['p95_us']:7.0f} | "
            f"{s['ours_python']['same_as_compiled']}/{s['ours_python']['of']}")


def headline(results):
    say("\n=== headline ===")
    for name, res in results.items():
        def pooled(method, keys):
            num = sum(res["sets"][k][method]["outcomes_pct"][o] * res["sets"][k]["scored"]
                      for k in keys for o in ("every image", "correct empty"))
            return num / sum(res["sets"][k]["scored"] for k in keys)
        hard = ("steep", "band", "edge", "axis")
        say(f"{name}: complete and correct answer (every image, or a correct empty one):")
        say(f"    uniform points:   ours {pooled('ours', ('uniform',)):5.1f} %   AFRT {pooled('afrt', ('uniform',)):5.1f} %")
        say(f"    difficult points: ours {pooled('ours', hard):5.1f} %   AFRT {pooled('afrt', hard):5.1f} %")
        u = res["sets"]["uniform"]
        say(f"    median time, uniform set: ours compiled {u['ours']['time']['median_us']:.0f} us through project() "
            f"({u['ours_core']['time']['median_us']:.0f} us in the core), ours Python "
            f"{u['ours_python']['time']['median_us'] / 1e3:.1f} ms, AFRT (Python) {u['afrt']['time']['median_us']:.0f} us;"
            f" compiled batch {res['batch_us_per_point_1_threads']:.0f} us/point on one thread, "
            f"{res['batch_us_per_point_all_threads']:.1f} on all cores")
        d = res["disagreements_with_truth"]
        say(f"    points where ours is not complete against the truth: {len(d)}")
        for x in d:
            say(f"      {x}")
    say("\nNotes: AFRT returns at most one ray per start, so it cannot give both images of a two-image point; a "
        "converged AFRT ray is a true image, and on easy points one start is often all it needs.  Times of AFRT here "
        "are of a Python implementation and those of 'ours, compiled' of C++ behind a Python call, so they are not a "
        "like-for-like comparison of the two algorithms' cost; the paper compares both in C++ (its Section 9.2.1).")


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    colours = {"every image": "#0ca30c", "correct empty": "#0ca30c", "some images": "#fab219",
               "no answer": "#ec835a", "false image": "#d03b3b"}
    method_colours = {"ours, compiled": "#2a78d6", "ours, Python": "#eb6834", "AFRT, Python": "#1baf7a"}
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink, "axes.titlesize": 9, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(len(results), 2, figsize=(10.5, 3.3 * len(results)), facecolor=surface,
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    axes = np.atleast_2d(axes)
    for row, (name, res) in enumerate(results.items()):
        ax = axes[row, 0]
        ax.set_facecolor(surface)
        xs, labels = [], []
        for i, key in enumerate(SETS):
            for j, (method, short) in enumerate((("ours", "ours"), ("afrt", "AFRT"))):
                x = i * 3 + j
                xs.append(x)
                labels.append(f"{short}")
                bottom = 0.0
                p = res["sets"][key][method]["outcomes_pct"]
                for o in OUTCOMES:
                    if p[o] <= 0:
                        continue
                    ax.bar(x, p[o], bottom=bottom, width=0.85, color=colours[o], edgecolor=surface, linewidth=1.0,
                           hatch="////" if o == "correct empty" else None)
                    bottom += p[o]
            ax.text(i * 3 + 0.5, -16, key, ha="center", va="top", color=ink, fontsize=8)
        ax.set_xticks(xs, labels, fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("points (%)")
        ax.set_title(f"{name}: is the answer complete and correct?", loc="left")
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if row == 0:
            handles = [Patch(facecolor=colours[o], edgecolor=surface, hatch="////" if o == "correct empty" else None,
                             label=o) for o in OUTCOMES]
            ax.legend(handles=handles, ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.10), frameon=False,
                      fontsize=7, handlelength=1.4, columnspacing=1.0)
        ax = axes[row, 1]
        ax.set_facecolor(surface)
        width = 0.26
        for j, (label, getter) in enumerate((("ours, compiled", lambda s: s["ours"]["time"]),
                                             ("ours, Python", lambda s: s["ours_python"]["time"]),
                                             ("AFRT, Python", lambda s: s["afrt"]["time"]))):
            med = np.array([getter(res["sets"][k])["median_us"] for k in SETS])
            p95 = np.array([getter(res["sets"][k])["p95_us"] for k in SETS])
            x = np.arange(len(SETS)) + (j - 1) * width
            ax.errorbar(x, med, yerr=[np.zeros_like(med), p95 - med], fmt="o", ms=5, color=method_colours[label],
                        ecolor=method_colours[label], elinewidth=1.2, capsize=2, label=label)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(SETS)), SETS)
        ax.set_ylabel("time per point (\u00b5s), median and p95")
        ax.set_title(f"{name}: time per point", loc="left")
        ax.grid(axis="y", which="major", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if row == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.10), ncol=3, frameon=False, fontsize=7)
    fig.text(0.01, 0.005, "Reference port: acrylic tube R = 100/94 mm, water inside, camera in air 350 mm from the axis. "
             "Times of single calls (ours through project()), on a machine also running other jobs.\n"
             "Truth: the rays of our search and of a brute-force tracing search that pass within 1e-9 r_P of the point; "
             "a returned ray is an image if within 1e-7 rad of a true ray.", fontsize=6.5, color=muted)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)


def main():
    if "--redraw" in sys.argv:
        saved = json.loads((HERE / f"bench_projection{SUFFIX}.json").read_text(encoding="utf-8"))
        figure(saved["results"], HERE / f"bench_projection{SUFFIX}.png")
        return
    t0 = time.perf_counter()
    cam = rc.Camera.look_at(CENTRE, (0.0, 0.0, 0.0), f=2000.0, size=(1280, 1024))
    windows = (rc.CylindricalWindow(RADII, INDICES), rc.SphericalWindow(RADII, INDICES))
    say(f"bench_projection{' --quick' if QUICK else ''}, {time.strftime('%Y-%m-%d %H:%M')}")
    import refractive_camera.projection as rp
    machine = {"cpu": cpu_name(), "os": platform.platform(), "python": sys.version.split()[0],
               "numpy": np.__version__, "backends": list(rc.available_backends()),
               "compiled_core": Path(rp._core.__file__).name if rp._core is not None else None}
    say(f"machine: {machine['cpu']}; {machine['os']}; Python {machine['python']}, NumPy {machine['numpy']}; "
        f"backends {machine['backends']} ({machine['compiled_core']})")
    say(f"truth: rays of our search and of a brute-force tracing search (grid {GRID}) passing within {TRUTH_TOL:g} r_P "
        f"of the point; a returned ray is an image if within {SAME_RAY:g} rad of a true ray's direction")
    if "cpp" not in rc.available_backends():
        raise SystemExit("the compiled backend is not built: build it first (pip install . , or "
                         "python setup.py build_ext --inplace)")
    results = {}
    for seed, w in zip((20261021, 20261022), windows):
        results[w.kind] = run_window(w, cam, seed)
        table(w.kind, results[w.kind])
    headline(results)
    total = time.perf_counter() - t0
    say(f"\ntotal {total / 60:.1f} min")
    out = {"machine": machine, "settings": {"truth_tol_rel": TRUTH_TOL, "afrt_accept_m": ACCEPT_M,
                                            "same_ray_rad": SAME_RAY, "grid": GRID,
                                            "sizes": SIZES, "truth_sizes": TRUTH_SIZES, "python_points": PY_POINTS},
           "results": results, "minutes": total / 60}
    (HERE / f"bench_projection{SUFFIX}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    figure(results, HERE / f"bench_projection{SUFFIX}.png")
    (HERE / f"bench_projection{SUFFIX}_out.txt").write_text("\n".join(_log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
