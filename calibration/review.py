"""Step through captured images and keep or reject each one.

Usage: python review.py --label A   (see board.py for the pattern args)
Keys:  K / RIGHT keep   D / X reject   U undo last decision (one step back)   LEFT back   ESC quit
Rejected images are moved to ./images/<label>/rejected/ (calibrate.py ignores them; move back to restore).
Red dots = detected corners; "NO BOARD" frames are useless for calibration anyway.
"""
import argparse
import shutil
from pathlib import Path

import cv2

from board import Board, add_args

p = argparse.ArgumentParser()
add_args(p)
p.add_argument("--label", required=True, help="name written on the camera module, e.g. A")
args = p.parse_args()
board = Board(args)

DIR = Path(__file__).resolve().parent / "images" / args.label
REJECTED = DIR / "rejected"
files = sorted(DIR.glob(f"{args.label}_*.png"))
if not files:
    raise SystemExit(f"no images in {DIR}")

LEFT, RIGHT = {2, 81, 63234}, {3, 83, 63235}  # arrow key codes differ between OpenCV backends
history, i = [], 0  # (index, file, was_rejected) for every keep/reject, newest last
while 0 <= i < len(files):
    f = files[i]
    if not f.exists():  # rejected earlier in this session
        i += 1
        continue
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    det = board.detect(img)
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if det is not None:
        for x, y in det[1]:
            cv2.circle(view, (int(x), int(y)), 4, (0, 0, 255), -1)
    ok = det is not None
    cv2.putText(view, f"{f.name}  {i + 1}/{len(files)}  {'board ok' if ok else 'NO BOARD'}", (10, 30), 0, 0.9,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.putText(view, "K keep  D reject  U undo  <- back  ESC quit", (10, view.shape[0] - 15), 0, 0.7, (0, 255, 255), 2)
    cv2.imshow("review", view)

    key = cv2.waitKeyEx(0)
    if key == 27:
        break
    if key in RIGHT or key in (ord("k"), ord("K")):
        history.append((i, f, False))
        i += 1
    elif key in LEFT:
        i = max(i - 1, 0)
        while i > 0 and not files[i].exists():
            i -= 1
    elif key in (ord("d"), ord("D"), ord("x"), ord("X")):
        REJECTED.mkdir(exist_ok=True)
        shutil.move(f, REJECTED / f.name)
        history.append((i, f, True))
        print(f"rejected {f.name}")
        i += 1
    elif key in (ord("u"), ord("U")) and history:
        i, f, was_rejected = history.pop()
        if was_rejected:
            shutil.move(REJECTED / f.name, f)
            print(f"restored {f.name}")

kept = len(list(DIR.glob(f"{args.label}_*.png")))
print(f"{kept} images kept, {len(list(REJECTED.glob('*.png'))) if REJECTED.exists() else 0} in {REJECTED}")
