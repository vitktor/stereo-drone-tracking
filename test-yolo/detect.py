"""Run YOLO11 on an image and save the annotated result next to this script.
Usage: python detect.py [image] [model]   (defaults: latest capture_*.jpg, yolo11n.pt)
"""
import sys
from pathlib import Path

from ultralytics import YOLO

DIR = Path(__file__).resolve().parent

img = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else max((p for p in DIR.glob("capture_*.jpg") if "_det" not in p.stem), key=lambda p: p.stat().st_mtime)
weights = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"

model = YOLO(DIR / weights)  # downloads weights on first run
r = model(img)[0]

for b in r.boxes:
    print(f"{r.names[int(b.cls)]:>12}  {float(b.conf):.2f}  {[round(v) for v in b.xyxy[0].tolist()]}")

out = img.with_name(img.stem + "_det.jpg")
r.save(filename=str(out))
print(f"Saved {out}")
