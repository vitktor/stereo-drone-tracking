"""Triangulates the drone from the YOLO boxes of a captured stereo session, one 3D point per synced pair.

Usage: python triangulate_session.py SESSION [--run full-n640] [--max-epi 5]
Input: data/drone/detections/<run>/<SESSION>/detections.csv (from detection/detect_session.py) and the session frames
Output: data/drone/reconstruction/<run>/<SESSION>/: trajectory.csv (one row per recorded pair), trajectory.png
(3D view + epipolar errors)
The point in each view is the centroid of the dark drone blob inside the box, which the two views agree on
better than the box centre. When the blob also takes in dark background (e.g. at a window edge) and the views no
longer agree, the box centres are tried instead and the pair closer to the epipolar line is kept.
Pairs still further than --max-epi px from the epipolar line, or whose box is cut by the image edge
(only part of the drone visible), are kept in the CSV but marked bad.
The good points are then smoothed (smooth.py), which also bridges short gaps and drops points off the path.
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from smooth import smooth
from stereo import LABEL, Rig

DATA = Path(__file__).resolve().parent.parent / "data" / "drone"
W, H = 1280, 800

p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--run", default="full-n640", help="detection run under data/drone/detections/")
p.add_argument("--max-epi", type=float, default=5.0, help="px from the epipolar line to accept a pair")
args = p.parse_args()
OUT = DATA / "reconstruction" / args.run / args.session
OUT.mkdir(parents=True, exist_ok=True)


def blob_centroid(img, box, pad=4):
    """Centroid of the largest dark region in the (slightly padded) box: the drone body and arms."""
    x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
    roi = cv2.GaussianBlur(img[y1:box[3] + pad, x1:box[2] + pad], (3, 3), 0)
    _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    return centroids[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] + (x1, y1)


def at_edge(box, margin=2):
    return box[0] <= margin or box[1] <= margin or box[2] >= W - margin or box[3] >= H - margin


boxes = {}  # pair -> {"A": box, "B": box}
for r in csv.DictReader(open(DATA / "detections" / args.run / args.session / "detections.csv")):
    boxes.setdefault(r["pair"], {})[LABEL[int(r["cam"])]] = [int(r[k]) for k in ("x1", "y1", "x2", "y2")]
stamps = {Path(r["cam0_file"]).stem.rsplit("_", 1)[0]: int(r["cam0_timestamp_ns"])
          for r in csv.DictReader(open(DATA / args.session / "pairs.csv"))}  # every recorded pair, in order

rig = Rig()
rows = []
for pair, ts in stamps.items():
    b = boxes.get(pair, {})
    if len(b) < 2:
        rows.append(dict(pair=pair, t=ts, point="", epi=np.nan, edge=False, ok=False, X=np.full(3, np.nan),
                         pt={c: np.full(2, np.nan) for c in "AB"}))
        continue
    pt = {c: blob_centroid(cv2.imread(str(DATA / args.session / f"{pair}_cam{i}.png"), cv2.IMREAD_GRAYSCALE), b[c])
          for i, c in LABEL.items()}
    epi, point = rig.epipolar_px(pt["A"][None], pt["B"][None])[0], "blob"
    if epi > args.max_epi:
        mid = {c: np.array([(b[c][0] + b[c][2]) / 2, (b[c][1] + b[c][3]) / 2]) for c in "AB"}
        epi_mid = rig.epipolar_px(mid["A"][None], mid["B"][None])[0]
        if epi_mid < epi:
            pt, epi, point = mid, epi_mid, "box"
    edge = at_edge(b["A"]) or at_edge(b["B"])
    rows.append(dict(pair=pair, t=ts, point=point, epi=epi, edge=edge, ok=epi <= args.max_epi and not edge,
                     X=rig.triangulate(pt["A"][None], pt["B"][None])[0], pt=pt))

t = np.array([(r["t"] - rows[0]["t"]) / 1e9 for r in rows])
both = np.array([r["point"] != "" for r in rows])
ok = np.array([r["ok"] for r in rows])
X = np.array([r["X"] for r in rows])
epi = np.array([r["epi"] for r in rows])
S, used = smooth(t, np.where(ok[:, None], X, np.nan))

fmt = lambda v, d: "" if np.isnan(v) else f"{v:.{d}f}"
with open(OUT / "trajectory.csv", "w") as f:
    f.write("pair,t_s,point,x_mm,y_mm,z_mm,epi_px,edge,ok,used,xs_mm,ys_mm,zs_mm,u_A,v_A,u_B,v_B\n")
    for r, ti, Xs, u in zip(rows, t, S, used):
        f.write(",".join([r["pair"], f"{ti:.3f}", r["point"], *(fmt(v, 1) for v in r["X"]), fmt(r["epi"], 2),
                          str(int(r["edge"])), str(int(r["ok"])), str(int(u)), *(fmt(v, 1) for v in Xs),
                          *(fmt(v, 2) for c in "AB" for v in r["pt"][c])]) + "\n")

n_edge = sum(r["edge"] for r in rows)
print(f"{args.session}: {len(rows)} pairs | box in both views {both.sum()} | good {ok.sum()} | "
      f"box at image edge {n_edge} | epipolar > {args.max_epi:g} px {both.sum() - ok.sum() - n_edge}")
print(f"good pairs: epipolar median {np.median(epi[ok]):.1f} px | distance from A "
      f"{np.linalg.norm(X[ok], axis=1).min() / 1000:.2f}-{np.linalg.norm(X[ok], axis=1).max() / 1000:.2f} m")
print(f"smoothing: {ok.sum() - used.sum()} off-path points dropped | path covers {(~np.isnan(S[:, 0])).sum()} pairs, "
      f"{((~np.isnan(S[:, 0])) & ~used).sum()} of them bridged | median distance raw-smoothed "
      f"{np.median(np.linalg.norm(X[used] - S[used], axis=1)):.1f} mm")

# Plot in A's frame turned so that up is roughly up: x right, z forward, -y up (until a floor frame exists)
show = lambda Q: (Q[..., 0], Q[..., 2], -Q[..., 1])
P, Ps = X / 1000, S / 1000
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 2, 1, projection="3d")
for c in "AB":
    R, T = rig.pose[c]
    corners = np.array([[x, y, 1] for x, y in ((-.6, -.4), (.6, -.4), (.6, .4), (-.6, .4))]) * 0.3  # frustum, 0.3 m deep
    world = (R.T @ (corners.T - T / 1000)).T  # camera -> A's frame, m
    o = rig.center(c) / 1000
    for k in range(4):
        ax.plot(*show(np.array([o, world[k]])), color="k", lw=0.8)
        ax.plot(*show(np.array([world[k], world[(k + 1) % 4]])), color="k", lw=0.8)
    ax.text(*show(o), f" {c}")
ax.scatter(*show(P[used]), color="gray", s=2, alpha=0.4, label="triangulated")
ax.scatter(*show(P[both & ~used]), color="red", marker="x", s=12, label="rejected")
for k in range(len(Ps) - 1):  # smoothed path, coloured by time
    if not np.isnan(Ps[k:k + 2]).any():
        ax.plot(*show(Ps[k:k + 2]), color=plt.cm.viridis(t[k] / t[-1]), lw=1.5)
ax.set_xlabel("x (m)"), ax.set_ylabel("z, forward from A (m)"), ax.set_zlabel("up (m)")
ax.set_aspect("equal")
ax.legend()
fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, t[-1]), "viridis"), ax=ax, shrink=0.6, label="time (s)")
ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(np.clip(epi[both], 0, 3 * args.max_epi), bins=40)
ax2.axvline(args.max_epi, color="red", ls="--")
ax2.set_xlabel(f"distance to epipolar line (px, clipped at {3 * args.max_epi:g})"), ax2.set_ylabel("pairs")
fig.suptitle(f"{args.session}: {used.sum()} of {len(rows)} stereo pairs triangulated")
fig.tight_layout()
fig.savefig(OUT / "trajectory.png", dpi=110)
print(f"-> {OUT}")
