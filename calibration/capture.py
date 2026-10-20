"""Live view of one camera from the Pi with board detection; saves frames for calibration.
On the Pi run pi_stream.py first.

Usage: python capture.py --label A [--interval 3]      (see board.py for the pattern args)
Keys:  SPACE save frame   A toggle auto-save (every --interval s while the board is detected)   ESC quit
Frames go to ./images/<label>/<label>_NNN.png. Green dots = corners already collected (aim to fill the frame).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from board import Board, add_args

PI_IP, W, H = "10.42.0.2", 1280, 800
IMAGES = Path(__file__).resolve().parent / "images"

p = argparse.ArgumentParser()
add_args(p)
p.add_argument("--label", required=True, help="name written on the camera module, e.g. A")
p.add_argument("--interval", type=float, default=3.0, help="seconds between auto-saves")
p.add_argument("--pi", default=PI_IP)
args = p.parse_args()
board = Board(args)
OUT = IMAGES / args.label
OUT.mkdir(parents=True, exist_ok=True)

ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.RCVHWM, 4)
s.connect(f"tcp://{args.pi}:5555")
s.setsockopt(zmq.SUBSCRIBE, b"")

n = max((int(f.stem.split("_")[-1]) + 1 for f in OUT.glob(f"{args.label}_*.png")), default=0)  # continue after gaps from review.py
coverage = np.zeros((H, W), np.uint8)
auto, last_save = False, 0.0

while True:
    msg = s.recv_multipart()
    while s.poll(0):  # drop stale frames so detection lag doesn't pile up
        msg = s.recv_multipart()
    img = np.frombuffer(msg[1], np.uint8).reshape(H, W)
    det = board.detect(img)

    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view[coverage > 0] = (0, 160, 0)
    if det is not None:
        for x, y in det[1]:
            cv2.circle(view, (int(x), int(y)), 4, (0, 0, 255), -1)
    status = f"{args.label}  saved {n}  auto {'ON' if auto else 'off'}  board {'ok' if det is not None else '--'}"
    cv2.putText(view, status, (10, 30), 0, 0.9, (0, 255, 0) if det is not None else (0, 0, 255), 2)
    cv2.imshow("calibration capture", view)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord("a"), ord("A")):
        auto = not auto
    if key == ord(" ") or (auto and det is not None and time.time() - last_save > args.interval):
        cv2.imwrite(str(OUT / f"{args.label}_{n:03d}.png"), img)
        if det is not None:
            for x, y in det[1]:
                cv2.circle(coverage, (int(x), int(y)), 6, 255, -1)
        print(f"saved {args.label}_{n:03d}.png")
        n, last_save = n + 1, time.time()
