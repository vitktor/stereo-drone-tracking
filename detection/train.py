"""Fine-tunes YOLO11 on the drone dataset built by build_dataset.py.

Usage: python train.py [--name bootstrap] [--model yolo11n.pt] [--imgsz 640] [--epochs 100] [--device mps]
Runs go to data/drone/runs/<name>/ (weights/best.pt, results.csv, plots); re-running a name overwrites it.
Frames are grayscale (OV9281), loaded as 3 identical channels, so hue/saturation augmentation is turned off.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "drone"

p = argparse.ArgumentParser()
p.add_argument("--name", default="bootstrap", help="run name under data/drone/runs/")
p.add_argument("--model", default=str(ROOT / "test-yolo" / "yolo11n.pt"), help="starting weights")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default="mps", help="mps (Apple GPU), cpu, or a CUDA index")
args = p.parse_args()

YOLO(args.model).train(
    data=str(DATA / "yolo" / "data.yaml"),
    imgsz=args.imgsz,
    epochs=args.epochs,
    patience=30,  # stop if val mAP hasn't improved for 30 epochs
    batch=args.batch,
    device=args.device,
    hsv_h=0, hsv_s=0,  # no colour in mono images
    project=str(DATA / "runs"),
    name=args.name,
    exist_ok=True,
    workers=4,
)
