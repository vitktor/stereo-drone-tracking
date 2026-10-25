"""Live view from the Pi; saves frames for the drone detection dataset.
On the Pi run calibration/pi_stream.py (one camera) or test-sync/sync_stream.py (synced pair); the stream type is detected.

Usage: python capture.py [--session NAME] [--interval 1] [--auto [--delay 30]]
Keys:  SPACE save frame(s)   A toggle auto-save (every --interval s)   ESC quit
Frames go to data/drone/<session>/: one camera <session>_NNN.png, stereo <session>_NNN_cam0.png + <session>_NNN_cam1.png
(same NNN = same synced pair; sensor timestamps and offsets are logged in pairs.csv).
One folder per session so train/val can be split by session.
"""
import argparse
import re
import time
import tkinter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import zmq

PI_IP, W, H = "10.42.0.2", 1280, 800
DATA = Path(__file__).resolve().parent.parent / "data" / "drone"
root = tkinter.Tk()
SCREEN_W = root.winfo_screenwidth()  # in points, which is what cv2 windows use on macOS
root.destroy()

p = argparse.ArgumentParser()
p.add_argument("--session", default=datetime.now().strftime("%Y%m%d_%H%M"), help="folder name (default: date_time)")
p.add_argument("--interval", type=float, default=1.0, help="seconds between auto-saves")
p.add_argument("--auto", action="store_true", help="turn auto-save on after --delay seconds")
p.add_argument("--delay", type=float, default=30.0, help="countdown before --auto starts saving (time to get the drone up)")
p.add_argument("--pi", default=PI_IP)
args = p.parse_args()
OUT = DATA / args.session
OUT.mkdir(parents=True, exist_ok=True)

ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.RCVHWM, 4)
s.connect(f"tcp://{args.pi}:5555")
s.setsockopt(zmq.SUBSCRIBE, b"")

idx = re.compile(rf"{re.escape(args.session)}_(\d+)")
n = max((int(m.group(1)) + 1 for f in OUT.glob("*.png") if (m := idx.match(f.stem))), default=0)  # continue an existing session
start = time.time() + args.delay if args.auto else None  # wall-clock time when the --auto countdown ends
auto, last_save = False, 0  # sensor timestamp (ns), so auto-save spacing is exact even if frames arrive in bursts
log = None
cv2.namedWindow("drone capture")
cv2.moveWindow("drone capture", 0, 0)  # left edge, so a full-width view fits on screen

while True:
    msg = s.recv_multipart()
    while s.poll(0):  # drop stale frames so the view stays live
        msg = s.recv_multipart()
    ts = np.frombuffer(msg[0], np.int64)[0]
    if len(msg) == 4:  # sync_stream.py: [timestamp cam0, offset cam1-cam0, frame cam0, frame cam1]
        dt = np.frombuffer(msg[1], np.int64)[0]
        frames = [np.frombuffer(m, np.uint8).reshape(H, W) for m in msg[2:]]
        scale = SCREEN_W / (2 * W)  # pair fills the full screen width
        view = cv2.resize(np.hstack(frames), None, fx=scale, fy=scale)
    else:  # pi_stream.py: [timestamp, frame]
        dt = None
        frames = [np.frombuffer(msg[1], np.uint8).reshape(H, W)]
        view = frames[0]

    if start is not None and time.time() >= start:
        auto, start = True, None
        print("auto-save started")

    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    if start is not None:
        cv2.putText(view, f"saving starts in {int(start - time.time()) + 1} s", (40, view.shape[0] // 2), 0, 1.6, (0, 220, 255), 4)
    if auto:
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 0, 255), 12)
        cv2.circle(view, (view.shape[1] - 40, 40), 14, (0, 0, 255), -1)
    status = f"{args.session}  saved {n}  auto {'ON' if auto else 'off'}"
    if dt is not None:
        status += f"  offset {dt / 1000:.1f} us"
    cv2.putText(view, status, (10, 30), 0, 0.9, (0, 0, 255) if auto else (0, 255, 0), 2)
    cv2.imshow("drone capture", view)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord("a"), ord("A")):  # also skips a running countdown
        auto, start = start is not None or not auto, None
    if key == ord(" ") or (auto and ts - last_save >= args.interval * 1e9):
        if len(frames) == 1:
            names = [f"{args.session}_{n:03d}.png"]
        else:
            names = [f"{args.session}_{n:03d}_cam{i}.png" for i in range(2)]
            if log is None:
                new = not (OUT / "pairs.csv").exists()
                log = open(OUT / "pairs.csv", "a")
                if new:
                    log.write("pair,cam0_file,cam1_file,cam0_timestamp_ns,offset_us\n")
            log.write(f"{n},{names[0]},{names[1]},{ts},{dt / 1000:.1f}\n")
            log.flush()
        for name, img in zip(names, frames):
            cv2.imwrite(str(OUT / name), img)
        print("saved " + " + ".join(names))
        n, last_save = n + 1, ts
