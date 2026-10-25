"""Pre-labels CVAT tasks with a trained model, so the remaining frames only need correcting instead of drawing.
CVAT replaces a task's annotations on upload, so every frame you already boxed keeps your box; all other frames
get the model's single most confident box (at most one drone per frame), or none below --conf.

Usage: python prelabel.py [--model ../data/drone/runs/bootstrap/weights/best.pt] [--conf 0.25]
Input: data/drone/annotation/<task>/ (CVAT export: train.txt lists the task's images, labels/train/ your boxes)
Output: data/drone/prelabels/<task>.zip, in the same Ultralytics YOLO Detection layout as the export.
Upload in CVAT: task -> Actions -> Upload annotations -> Ultralytics YOLO Detection 1.0.
"""
import argparse
import zipfile
from pathlib import Path

from ultralytics import YOLO

DATA = Path(__file__).resolve().parent.parent / "data" / "drone"
OUT = DATA / "prelabels"

p = argparse.ArgumentParser()
p.add_argument("--model", default=str(DATA / "runs" / "bootstrap" / "weights" / "best.pt"))
p.add_argument("--conf", type=float, default=0.25, help="minimum confidence for a predicted box")
p.add_argument("--imgsz", type=int, default=640, help="same as training")
p.add_argument("--device", default="mps")
args = p.parse_args()

model = YOLO(args.model)
OUT.mkdir(exist_ok=True)

for task in sorted(d for d in (DATA / "annotation").iterdir() if (d / "train.txt").exists()):
    names = [Path(line).stem for line in (task / "train.txt").read_text().split()]
    labels, unsure = {}, []
    todo = []
    for name in names:
        manual = task / "labels" / "train" / f"{name}.txt"
        if manual.exists():
            labels[name] = manual.read_text()
        else:
            todo.append(name)
    for name in todo:
        r = model.predict(DATA / name.rsplit("_", 2)[0] / f"{name}.png", imgsz=args.imgsz, conf=args.conf,
                          max_det=1, device=args.device, verbose=False)[0]
        if len(r.boxes):
            x, y, w, h = r.boxes.xywhn[0].tolist()
            labels[name] = f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
            if r.boxes.conf[0] < 0.5:
                unsure.append(name)

    with zipfile.ZipFile(OUT / f"{task.name}.zip", "w") as z:
        z.write(task / "data.yaml", "data.yaml")
        z.write(task / "train.txt", "train.txt")
        for name, text in labels.items():
            z.writestr(f"labels/train/{name}.txt", text)
    predicted = sum(1 for n in todo if n in labels)
    print(f"{task.name}: {len(names) - len(todo)} yours, {predicted} predicted, {len(todo) - predicted} empty"
          f" -> {OUT / task.name}.zip")
    if unsure:
        print(f"  low confidence (<0.5), check these: {' '.join(unsure)}")
