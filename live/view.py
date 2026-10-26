"""Shows the Pi's live tracking (live/track.py --pub): both camera views with the detections, and the drone's path
from above and from the side.

Usage: python view.py [--pi 10.42.0.2] [--port 5556] [--trail 5]
Keys:  C clear the path   ESC quit
Positions are in the world frame of stereo.Rig.world: z up from the floor, x from camera A to B, y into the room, m.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reconstruction"))
from stereo import Rig  # noqa: E402

XR, YR, ZR = (-2.0, 2.0), (0.0, 4.5), (0.0, 2.6)  # m, shown part of the room
PPM = 110  # px per m in the plots
CAM_COLOR = {"A": (209, 155, 31), "B": (58, 131, 232)}  # BGR, as in reconstruction/render.py

p = argparse.ArgumentParser()
p.add_argument("--pi", default="10.42.0.2")
p.add_argument("--port", type=int, default=5556)
p.add_argument("--trail", type=float, default=5.0, help="s of path to show")
args = p.parse_args()

rig = Rig()
cams = {c: rig.world(rig.center(c)[None])[0] for c in "AB"}
s = zmq.Context().socket(zmq.SUB)
s.setsockopt(zmq.RCVHWM, 2)
s.connect(f"tcp://{args.pi}:{args.port}")
s.setsockopt(zmq.SUBSCRIBE, b"")

trail = deque()  # (t, filtered world point)
rate, rate_n, rate_t = 0.0, 0, time.time()


def height_color(z):
    v = np.uint8(np.clip((z - ZR[0]) / (ZR[1] - ZR[0]), 0, 1) * 255)
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[v]]), cv2.COLORMAP_PLASMA)[0, 0])


def panel(hr, vr, title, h_label, v_label, h_axis, v_axis):
    """Plot of the trail on world axes h_axis (right) and v_axis (up), with a 0.5 m grid and the cameras."""
    w, h = int((hr[1] - hr[0]) * PPM), int((vr[1] - vr[0]) * PPM)
    img = np.full((h + 40, w + 40, 3), 24, np.uint8)
    to_px = lambda P: (int(20 + (P[h_axis] - hr[0]) * PPM), int(20 + h - (P[v_axis] - vr[0]) * PPM))
    for g in np.arange(np.ceil(hr[0] * 2) / 2, hr[1] + 1e-6, 0.5):
        cv2.line(img, (int(20 + (g - hr[0]) * PPM), 20), (int(20 + (g - hr[0]) * PPM), 20 + h), (50, 50, 50), 1)
    for g in np.arange(np.ceil(vr[0] * 2) / 2, vr[1] + 1e-6, 0.5):
        cv2.line(img, (20, int(20 + h - (g - vr[0]) * PPM)), (20 + w, int(20 + h - (g - vr[0]) * PPM)), (50, 50, 50), 1)
    for c, o in cams.items():
        cv2.circle(img, to_px(o), 6, CAM_COLOR[c], -1)
        cv2.putText(img, c, (to_px(o)[0] + 8, to_px(o)[1] + 5), 0, 0.5, CAM_COLOR[c], 1)
    pts = [P for _, P in trail]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, to_px(a), to_px(b), height_color(b[2]), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(img, to_px(pts[-1]), 7, height_color(pts[-1][2]), -1)
    cv2.putText(img, f"{title}  (grid 0.5 m, {h_label} right, {v_label} up)", (20, 14), 0, 0.45, (200, 200, 200), 1)
    return img


while True:
    if not s.poll(1000):
        print("nothing from the Pi; is live/track.py running with --pub?")
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        continue
    meta, jpg = s.recv_multipart()
    m = json.loads(meta)
    frames = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)  # A | B, half size

    rate_n += 1
    if time.time() - rate_t >= 1:
        rate, rate_n, rate_t = rate_n / (time.time() - rate_t), 0, time.time()
    if trail and m["t"] < trail[-1][0]:
        trail.clear()  # track.py restarted
    if m["Xf"] is not None:
        trail.append((m["t"], np.array(m["Xf"])))
    while trail and trail[0][0] < m["t"] - args.trail:
        trail.popleft()

    half_w = frames.shape[1] // 2
    for i, c in enumerate("AB"):
        if m["box"][c]:
            (x1, y1, x2, y2), conf = m["box"][c]
            box_color = (0, 220, 0) if m["ok"] else (0, 200, 255)
            cv2.rectangle(frames, (i * half_w + x1 // 2, y1 // 2), (i * half_w + x2 // 2, y2 // 2), box_color, 2)
            cv2.putText(frames, f"{conf:.2f}", (i * half_w + x1 // 2, y1 // 2 - 6), 0, 0.5, box_color, 1)
        if m["pt"]:
            u, v = m["pt"][c]
            cv2.drawMarker(frames, (int(i * half_w + u / 2), int(v / 2)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(frames, c, (i * half_w + 10, 28), 0, 0.9, CAM_COLOR[c], 2)

    top = panel(XR, YR, "from above", "x", "y", 0, 1)
    side = panel(XR, ZR, "from the side", "x", "z", 0, 2)
    plots = np.vstack([side, top])
    view = np.full((max(frames.shape[0], plots.shape[0]), frames.shape[1] + plots.shape[1], 3), 24, np.uint8)
    view[:frames.shape[0], :frames.shape[1]] = frames
    view[:plots.shape[0], frames.shape[1]:] = plots
    lines = [f"{rate:.1f} pairs/s   latency {m['latency_ms']:.0f} ms   YOLO {m['detect_ms']:.0f} ms (both views)"]
    if m["epi"] is not None:
        lines.append(f"epipolar {m['epi']:.1f} px  {'ok' if m['ok'] else 'rejected'}  {'' if m['used'] else '(not used)'}")
    lines.append("no track" if m["Xf"] is None else "drone  x {:+.2f}  y {:.2f}  z {:.2f} m".format(*m["Xf"]))
    for k, line in enumerate(lines):
        cv2.putText(view, line, (12, frames.shape[0] + 34 + 32 * k), 0, 0.8, (230, 230, 230), 2)
    cv2.imshow("live", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        trail.clear()
    if key == 27:
        break
cv2.destroyAllWindows()
