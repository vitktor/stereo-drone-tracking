"""Times a drone detector on a folder of frames and logs its boxes, to check speed and accuracy on the Pi.

Usage: python bench.py MODEL IMAGES [--imgsz 640] [--conf 0.25] [--out detections.csv]
MODEL: best.pt or an exported folder (e.g. best_ncnn_model from model.export(format="ncnn", imgsz=640))
IMAGES: folder of PNG frames (e.g. a capture.py session)
Output: per-stage timing summary, and one CSV row per image (same columns as detect_session.py) to diff
        the exported model against best.pt run on the same frames.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("images")
p.add_argument("--imgsz", type=int, default=640, help="same as training/export")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--out", default="detections.csv")
p.add_argument("--warmup", type=int, default=5, help="untimed frames first (model load, caches)")
args = p.parse_args()

model = YOLO(args.model, task="detect")
files = sorted(Path(args.images).glob("*.png"))
imgs = [cv2.imread(str(f)) for f in files]  # read up front, so disk time isn't counted

for img in imgs[:args.warmup]:
    model.predict(img, imgsz=args.imgsz, conf=args.conf, max_det=1, verbose=False)

stages, wall, hits = [], [], 0
with open(args.out, "w") as log:
    log.write("image,conf,x1,y1,x2,y2\n")
    for f, img in zip(files, imgs):
        t = time.perf_counter()
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, max_det=1, verbose=False)[0]
        wall.append((time.perf_counter() - t) * 1000)
        stages.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
        if len(r.boxes):
            hits += 1
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[0].tolist())
            log.write(f"{f.stem},{float(r.boxes.conf[0]):.3f},{x1},{y1},{x2},{y2}\n")
        else:
            log.write(f"{f.stem},,,,,\n")

s = np.array(stages)
w = np.array(wall)
print(f"{len(files)} images, {hits} with a box -> {args.out}")
print(f"ms per image: pre {s[:, 0].mean():.1f} | inference {s[:, 1].mean():.1f} | post {s[:, 2].mean():.1f}"
      f" | total {w.mean():.1f} (median {np.median(w):.1f}, max {w.max():.1f}) = {1000 / w.mean():.1f} fps,"
      f" {1000 / w.mean() / 2:.1f} stereo pairs/s")
