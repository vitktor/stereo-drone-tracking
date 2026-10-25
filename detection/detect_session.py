"""Runs a trained drone detector on a captured stereo session and saves what it found, to check unseen footage.

Usage: python detect_session.py SESSION [--run full-n640] [--conf 0.25] [--imgsz 640]
Input: data/drone/<SESSION>/<SESSION>_NNN_cam0.png + _cam1.png (from capture.py)
Output: data/drone/detections/<run>/<SESSION>/: NNN.jpg (cam0 | cam1 with boxes), detections.csv
At most one drone per image; the summary counts pairs with a box in both cameras, since triangulation needs both.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA = Path(__file__).resolve().parent.parent / "data" / "drone"

p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--run", default="full-n640", help="training run under data/drone/runs/")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--imgsz", type=int, default=640, help="same as training")
p.add_argument("--device", default="mps")
args = p.parse_args()

model = YOLO(DATA / "runs" / args.run / "weights" / "best.pt")
OUT = DATA / "detections" / args.run / args.session
OUT.mkdir(parents=True, exist_ok=True)

pairs = sorted({f.stem.rsplit("_", 1)[0] for f in (DATA / args.session).glob("*_cam0.png")})
found = []  # per pair: [cam0 found, cam1 found]
with open(OUT / "detections.csv", "w") as log:
    log.write("pair,cam,conf,x1,y1,x2,y2\n")
    for pair in pairs:
        views, hits = [], []
        for cam in (0, 1):
            img = cv2.imread(str(DATA / args.session / f"{pair}_cam{cam}.png"))
            r = model.predict(img, imgsz=args.imgsz, conf=args.conf, max_det=1, device=args.device, verbose=False)[0]
            if len(r.boxes):
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())
                conf = float(r.boxes.conf[0])
                log.write(f"{pair},{cam},{conf:.3f},{x1},{y1},{x2},{y2}\n")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, y1 - 6), 0, 0.8, (0, 0, 255), 2)
            hits.append(len(r.boxes) > 0)
            views.append(img)
        cv2.putText(views[0], pair, (10, 30), 0, 0.9, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{pair.rsplit('_', 1)[1]}.jpg"), cv2.resize(np.hstack(views), None, fx=0.5, fy=0.5))
        found.append(hits)

f = np.array(found)
print(f"{args.session}: {len(f)} pairs | both cams {f.all(1).sum()} | cam0 only {(f[:, 0] & ~f[:, 1]).sum()}"
      f" | cam1 only {(~f[:, 0] & f[:, 1]).sum()} | none {(~f.any(1)).sum()} -> {OUT}")
