"""Step through captured stereo pairs (<name>_NNN_cam0.png + <name>_NNN_cam1.png) and pick the good ones.

Usage: python review_pairs.py --dir ../data/drone/checker-calib-2 --cols 10 --rows 9   (see board.py for the pattern args)
Keys:  K keep (copies both images to <dir>/kept/)   U undo last keep   RIGHT next   LEFT back   ESC quit
Originals are never moved; delete a pair from kept/ by hand to drop it. "KEPT" in the header = pair already in kept/.
Red dots = detected corners, green ring = corner 0: it must sit on the same board corner in both views.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from board import Board, add_args

p = argparse.ArgumentParser()
add_args(p)
p.add_argument("--dir", required=True, help="folder with the _cam0/_cam1 pairs")
args = p.parse_args()
board = Board(args)

DIR = Path(args.dir).resolve()
KEPT = DIR / "kept"
pairs = [(f, f.with_name(f.name.replace("_cam0", "_cam1"))) for f in sorted(DIR.glob("*_cam0.png"))]
pairs = [pr for pr in pairs if pr[1].exists()]
if not pairs:
    raise SystemExit(f"no pairs in {DIR}")


def draw(f):
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    det = board.detect(img)
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if det is not None:
        for x, y in det[1]:
            cv2.circle(view, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.circle(view, tuple(int(v) for v in det[1][0]), 18, (0, 255, 0), 4)
    cv2.putText(view, f"{f.stem.split('_')[-1]}  {'board ok' if det is not None else 'NO BOARD'}", (10, 120), 0, 1.6,
                (0, 255, 0) if det is not None else (0, 0, 255), 3)
    return view, det is not None


LEFT, RIGHT = {2, 81, 63234}, {3, 83, 63235}  # arrow key codes differ between OpenCV backends
history, i = [], 0  # (index, pair) for every keep made in this session, newest last
while 0 <= i < len(pairs):
    f0, f1 = pairs[i]
    (v0, ok0), (v1, ok1) = draw(f0), draw(f1)
    view = cv2.resize(np.hstack([v0, v1]), None, fx=0.5, fy=0.5)
    ok = ok0 and ok1
    is_kept = (KEPT / f0.name).exists()
    n_kept = len(list(KEPT.glob("*_cam0.png"))) if KEPT.exists() else 0
    cv2.putText(view, f"{f0.name.replace('_cam0', '')}  {i + 1}/{len(pairs)}  kept {n_kept}  "
                      f"{'board in both' if ok else 'NOT IN BOTH'}{'  KEPT' if is_kept else ''}",
                (10, 25), 0, 0.7, (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.putText(view, "K keep  U undo  -> next  <- back  ESC quit", (10, view.shape[0] - 12), 0, 0.6, (0, 255, 255), 2)
    cv2.imshow("review pairs", view)

    key = cv2.waitKeyEx(0)
    if key == 27:
        break
    if key in (ord("k"), ord("K")):
        if not is_kept:
            KEPT.mkdir(exist_ok=True)
            for f in (f0, f1):
                shutil.copy2(f, KEPT / f.name)
            history.append((i, pairs[i]))
            print(f"kept {f0.name} + {f1.name}")
        i += 1
    elif key in RIGHT:
        i += 1
    elif key in LEFT:
        i = max(i - 1, 0)
    elif key in (ord("u"), ord("U")) and history:
        i, pr = history.pop()
        for f in pr:
            (KEPT / f.name).unlink(missing_ok=True)
        print(f"un-kept {pr[0].name} + {pr[1].name}")

n_kept = len(list(KEPT.glob("*_cam0.png"))) if KEPT.exists() else 0
print(f"{n_kept} pairs in {KEPT}")
