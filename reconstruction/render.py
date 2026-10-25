"""Renders a reconstructed flight as a video: both camera views on top, the 3D scene with the trajectory below.

Usage: python render.py SESSION [--run full-n640] [--start 0] [--end END] [--out FILE.mp4]
Input: data/drone/reconstruction/<run>/<SESSION>/trajectory.csv (from triangulate_session.py), the detection boxes
and the session frames. Output: <SESSION>.mp4 next to trajectory.csv (H.264, 1920x1320, 30 fps; needs ffmpeg).
World frame: z up from the floor, x along the baseline from camera A to B, y into the room, metres.
Top views and 3D path share one colour scale, the drone's height.
"""
import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from stereo import LABEL, Rig

DATA = Path(__file__).resolve().parent.parent / "data" / "drone"
UP_A = np.array([-0.035, -0.931, 0.364])  # up in camera A's frame, from vertical room edges (A and B agree within 1 deg)
CAM_HEIGHT = 1.0  # m, cameras above the floor (tape-measured roughly)
VW, VH, H3 = 960, 600, 720  # each camera view, and the 3D strip below them
TAIL, TRAIL = 3.0, 2.0  # s of bright path in 3D and of trail in the camera views
ZMAX = 2.6  # m, top of the height colour scale
CMAP = plt.get_cmap("plasma")
CAM_COLOR = {"A": "#1f9bd1", "B": "#e8833a"}

p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--run", default="full-n640")
p.add_argument("--start", type=float, default=0.0, help="s")
p.add_argument("--end", type=float, help="s (default: end of the recording)")
p.add_argument("--out", help="default: <SESSION>.mp4 next to trajectory.csv")
args = p.parse_args()
DIR = DATA / "reconstruction" / args.run / args.session

rig = Rig()
up = UP_A / np.linalg.norm(UP_A)
mid = (rig.center("A") + rig.center("B")) / 2
ex = rig.center("B") - rig.center("A")
ex -= (ex @ up) * up
ex /= np.linalg.norm(ex)
ey = np.cross(up, ex)
TO_W = np.array([ex, ey, up])  # rows: world axes in A's frame


def world(P):
    """Points (N, 3) in A's frame, mm -> world, m."""
    return (np.asarray(P) - mid) @ TO_W.T / 1000 + [0, 0, CAM_HEIGHT]


def from_world(W):
    """World points (N, 3), m -> A's frame, mm."""
    return (np.asarray(W) - [0, 0, CAM_HEIGHT]) * 1000 @ TO_W + mid


rows = list(csv.DictReader(open(DIR / "trajectory.csv")))
pairs = [r["pair"] for r in rows]
t = np.array([float(r["t_s"]) for r in rows])
S = np.array([[float(r[k]) if r[k] else np.nan for k in ("xs_mm", "ys_mm", "zs_mm")] for r in rows])
Wp = world(S)  # NaN rows stay NaN
boxes = {}
for r in csv.DictReader(open(DATA / "detections" / args.run / args.session / "detections.csv")):
    boxes.setdefault(r["pair"], {})[LABEL[int(r["cam"])]] = [int(r[k]) for k in ("x1", "y1", "x2", "y2")]
# trail pixels in each view, projected once for the whole flight
have = ~np.isnan(Wp[:, 0])
px = {c: np.full((len(t), 2), np.nan) for c in "AB"}
for c in "AB":
    px[c][have] = rig.project(c, S[have])


def bgr(z):
    r, g, b, _ = CMAP(np.clip(z / ZMAX, 0, 1))
    return int(b * 255), int(g * 255), int(r * 255)


def camera_view(c, k):
    i = next(i for i, l in LABEL.items() if l == c)
    img = cv2.cvtColor(cv2.imread(str(DATA / args.session / f"{pairs[k]}_cam{i}.png"), cv2.IMREAD_GRAYSCALE),
                       cv2.COLOR_GRAY2BGR)
    if c in boxes.get(pairs[k], {}):
        x1, y1, x2, y2 = boxes[pairs[k]][c]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    lo = np.searchsorted(t, t[k] - TRAIL)
    for j in range(lo, k):
        if have[j] and have[j + 1]:
            a, b = (np.round(px[c][j:j + 2] * 4).astype(int))  # 2 bits of subpixel precision
            alpha = (j - lo) / max(k - lo, 1)
            cv2.line(img, tuple(a), tuple(b), bgr(Wp[j, 2]), 1 + round(3 * alpha), cv2.LINE_AA, shift=2)
    if have[k]:
        cv2.circle(img, tuple(np.round(px[c][k] * 4).astype(int)), 6 * 4, bgr(Wp[k, 2]), 2, cv2.LINE_AA, shift=2)
    img = cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA)
    color = tuple(int(CAM_COLOR[c][i:i + 2], 16) for i in (5, 3, 1))
    cv2.putText(img, f"camera {c}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return img


# --- static 3D scene ---
fig = plt.figure(figsize=(2 * VW / 100, H3 / 100), dpi=100, facecolor="white")
ax = fig.add_axes([-0.1, -0.1, 1.2, 1.32], projection="3d")  # larger than the figure: crops the empty 3D margins
ax.set_axis_off()
lim = np.nanpercentile(Wp, [0, 100], axis=0)
xlo, xhi = min(lim[0, 0], -1.2) - 0.3, max(lim[1, 0], 1.2) + 0.3
ylo, yhi = -0.3, lim[1, 1] + 0.3
for gx in np.arange(np.floor(xlo * 2) / 2, xhi + 1e-6, 0.5):
    ax.plot([gx, gx], [ylo, yhi], [0, 0], color="#d5d5d5", lw=0.6)
for gy in np.arange(np.ceil(ylo * 2) / 2, yhi + 1e-6, 0.5):
    ax.plot([xlo, xhi], [gy, gy], [0, 0], color="#d5d5d5", lw=0.6)
centers = {}
for c in "AB":
    R, T = rig.pose[c]
    corners = np.array([[x, y, 1] for x, y in ((-.6, -.4), (.6, -.4), (.6, .4), (-.6, .4))]) * 350  # 0.35 m deep
    fr = world((R.T @ (corners.T - T)).T)
    o = world(rig.center(c)[None])[0]
    centers[c] = o
    for k in range(4):
        ax.plot(*np.c_[o, fr[k]], color=CAM_COLOR[c], lw=1.4)
        ax.plot(*np.c_[fr[k], fr[(k + 1) % 4]], color=CAM_COLOR[c], lw=1.4)
    ax.plot(*np.c_[o, [o[0], o[1], 0]], color=CAM_COLOR[c], lw=2.2, alpha=0.6)  # stand down to the floor
    ax.text(o[0], o[1], o[2] + 0.35, c, color=CAM_COLOR[c], fontsize=15, fontweight="bold", ha="center")
ax.plot(*np.c_[centers["A"], centers["B"]], color="#999999", lw=0.8, ls="--")
ax.set_xlim(xlo, xhi), ax.set_ylim(ylo, yhi), ax.set_zlim(0, ZMAX)
ax.set_box_aspect((xhi - xlo, yhi - ylo, ZMAX))
info = fig.text(0.012, 0.93, "", fontsize=15, family="monospace", color="#333333")
fig.text(0.012, 0.03, "grid 0.5 m", fontsize=11, color="#999999")

# --- dynamic artists ---
old, = ax.plot([], [], [], color="#9a9a9a", lw=0.9)
shadow, = ax.plot([], [], [], color="#c8c8c8", lw=0.9)
tail = Line3DCollection([], lw=2.6)
ax.add_collection3d(tail, autolim=False)
drop, = ax.plot([], [], [], color="#777777", lw=0.9, ls="--")
rays = {c: ax.plot([], [], [], color=CAM_COLOR[c], lw=0.8, alpha=0.7)[0] for c in "AB"}
dot = ax.scatter([], [], [], s=70, edgecolor="black", linewidth=0.8, depthshade=False)
foot = ax.scatter([], [], [], s=18, color="#888888", depthshade=False)


def path(seg):
    """Path with NaN gaps kept as breaks."""
    return seg[:, 0], seg[:, 1], seg[:, 2]


def scene(k):
    lo = np.searchsorted(t, t[k] - TAIL)
    old.set_data_3d(*path(Wp[:lo + 1]))
    flat = Wp[:k + 1].copy()
    flat[:, 2] = 0.003
    shadow.set_data_3d(*path(flat))
    seg = np.stack([Wp[lo:k], Wp[lo + 1:k + 1]], 1)
    good = ~np.isnan(seg).any((1, 2))
    tail.set_segments(seg[good])
    alpha = np.linspace(0.25, 1, max(len(seg), 1))[good]
    col = CMAP(np.clip(seg[good, 1, 2] / ZMAX, 0, 1))
    col[:, 3] = alpha
    tail.set_color(col)
    if have[k]:
        w = Wp[k]
        drop.set_data_3d([w[0], w[0]], [w[1], w[1]], [0, w[2]])
        for c in "AB":
            rays[c].set_data_3d(*np.c_[centers[c], w])
        dot._offsets3d = ([w[0]], [w[1]], [w[2]])
        dot.set_color(CMAP(np.clip(w[2] / ZMAX, 0, 1)))
        dot.set_edgecolor("black")
        foot._offsets3d = ([w[0]], [w[1]], [0])
        dist = np.linalg.norm(w - [0, 0, CAM_HEIGHT])
        info.set_text(f"t {t[k]:5.1f} s   height {w[2]:4.2f} m   distance {dist:4.2f} m")
    else:
        drop.set_data_3d([], [], [])
        for c in "AB":
            rays[c].set_data_3d([], [], [])
        dot._offsets3d = foot._offsets3d = ([], [], [])
        info.set_text(f"t {t[k]:5.1f} s   drone not seen")
    ax.view_init(elev=22, azim=-90 + 14 * np.sin(2 * np.pi * t[k] / 24))
    fig.canvas.draw()
    return cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


out = Path(args.out) if args.out else DIR / f"{args.session}.mp4"
ks = np.flatnonzero((t >= args.start) & (t <= (args.end if args.end is not None else t[-1])))
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{2 * VW}x{VH + H3}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-crf", "20", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
for n, k in enumerate(ks):
    frame = np.vstack([np.hstack([camera_view("A", k), camera_view("B", k)]), scene(k)])
    ff.stdin.write(frame.tobytes())
    if n % 150 == 0:
        print(f"{n}/{len(ks)} frames")
ff.stdin.close()
ff.wait()
print(f"-> {out}")
