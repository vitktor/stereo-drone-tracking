"""Builds a YOLO dataset from the CVAT exports (Ultralytics YOLO Detection format, without images).
Every image of a task is used: CVAT exports no label file for a frame without a box, which counts as "no drone",
so tasks must be fully reviewed. More empty frames come from the --negatives sessions.

Usage: python build_dataset.py [--val drone-train-3] [--negatives drone-train-6] [--zip]
Input: data/drone/annotation/<task>/ (train.txt lists the task's images, labels/train/ the boxes)
       + images in data/drone/<session>/
Output: data/drone/yolo/{images,labels}/{train,val} + data.yaml (rebuilt from scratch on every run);
        data.yaml has no absolute path, so with --zip the dataset also trains elsewhere (train_colab.ipynb)
Val is split by session; negatives are split by pair (every 5th to val, every other one of the rest to train),
so cam0/cam1 of a pair stay together.
"""
import argparse
import re
import shutil
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "drone"
OUT = DATA / "yolo"

p = argparse.ArgumentParser()
p.add_argument("--val", nargs="+", default=["drone-train-3"], help="sessions used for validation")
p.add_argument("--negatives", nargs="+", default=["drone-train-6"], help="sessions with no drone in any frame")
p.add_argument("--zip", action="store_true", help="also pack the dataset into data/drone/yolo.zip for upload")
args = p.parse_args()

name = re.compile(r"(.+)_(\d+)_cam\d$")
items = []  # (image, label text, split)
for task in sorted(DATA.glob("annotation/*/train.txt")):
    for stem in (Path(line).stem for line in task.read_text().split()):
        session = name.match(stem).group(1)
        label = task.parent / "labels" / "train" / f"{stem}.txt"
        items.append((DATA / session / f"{stem}.png", label.read_text() if label.exists() else "",
                      "val" if session in args.val else "train"))
for session in args.negatives:
    for img in sorted((DATA / session).glob("*.png")):
        pair = int(name.match(img.stem).group(2))
        if pair % 5 == 4:
            items.append((img, "", "val"))
        elif pair % 2 == 0:  # every other pair, so empty frames stay ~25% of train
            items.append((img, "", "train"))

shutil.rmtree(OUT, ignore_errors=True)
for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True)
    (OUT / "labels" / split).mkdir(parents=True)
for img, label, split in items:
    shutil.copy2(img, OUT / "images" / split / img.name)
    (OUT / "labels" / split / f"{img.stem}.txt").write_text(label)
(OUT / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: drone\n")  # relative to this file

for split in ("train", "val"):
    n = [i for i in items if i[2] == split]
    print(f"{split}: {len(n)} images, {sum(1 for i in n if i[1])} with drone, {sum(1 for i in n if not i[1])} empty")
print(f"wrote {OUT / 'data.yaml'}")
if args.zip:
    print(f"wrote {shutil.make_archive(str(OUT), 'zip', DATA, OUT.name)}")
