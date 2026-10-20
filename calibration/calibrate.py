"""Calibrate one camera's intrinsics from ./images/<label>/<label>_*.png.

Usage: python calibrate.py --label A [--rational]
Writes calib_<label>.yaml (OpenCV FileStorage, readable from C++/Python on the Pi) and undistorted_<label>.png.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from board import Board, add_args

DIR = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
add_args(p)
p.add_argument("--label", required=True, help="name written on the camera module, e.g. A")
p.add_argument("--rational", action="store_true", help="8-coefficient distortion model, for wide-angle lenses")
args = p.parse_args()
board = Board(args)

files = sorted((DIR / "images" / args.label).glob(f"{args.label}_*.png"))
if not files:
    raise SystemExit(f"no images/{args.label}/{args.label}_*.png - run capture.py first")
size = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE).shape[::-1]
views = [(f, d) for f in files if (d := board.detect(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))) is not None]
obj = [board.obj[ids] for _, (ids, _) in views]
img = [pts for _, (_, pts) in views]

flags = cv2.CALIB_RATIONAL_MODEL if args.rational else 0
# OpenCV's own initial guess ignores distortion and diverged on camera B (fx ~7000, cy off-image);
# starting from a generic ~90 deg lens with centred principal point converges reliably.
K0 = np.array([[0.6 * size[0], 0, (size[0] - 1) / 2], [0, 0.6 * size[0], (size[1] - 1) / 2], [0, 0, 1]])
rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, K0, np.zeros(14), flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)

errs = []
for o, im, r, t in zip(obj, img, rvecs, tvecs):
    proj, _ = cv2.projectPoints(o, r, t, K, D)
    errs.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im) ** 2, axis=1))))
print(f"{args.label}: {len(views)}/{len(files)} images used, RMS {rms:.3f} px")
print(f"  fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}")
def angle(px):  # degrees between the optical axis and the ray through pixel px, distortion included
    x, y = cv2.undistortPoints(np.array([[px]], np.float32), K, D)[0, 0]
    return np.degrees(np.arctan(np.hypot(x, y)))
w, h, cx, cy = size[0] - 1, size[1] - 1, K[0, 2], K[1, 2]
print(f"  FOV horizontal {angle((0, cy)) + angle((w, cy)):.1f}  vertical {angle((cx, 0)) + angle((cx, h)):.1f}"
      f"  diagonal {angle((0, 0)) + angle((w, h)):.1f} deg")
print(f"  dist {np.round(D.ravel()[:8 if args.rational else 5], 4).tolist()}")
worst = sorted(zip(errs, (f.name for f, _ in views)), reverse=True)[:5]
print("  worst views: " + ", ".join(f"{n} {e:.2f}" for e, n in worst))

fs = cv2.FileStorage(str(DIR / f"calib_{args.label}.yaml"), cv2.FILE_STORAGE_WRITE)
fs.write("image_width", size[0]); fs.write("image_height", size[1])
fs.write("rms", rms); fs.write("K", K); fs.write("D", D)
fs.release()

# sanity check: straight edges of the board should look straight after undistortion
src = cv2.imread(str(views[len(views) // 2][0]), cv2.IMREAD_GRAYSCALE)
newK, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=1)
cv2.imwrite(str(DIR / f"undistorted_{args.label}.png"), np.hstack([src, cv2.undistort(src, K, D, None, newK)]))
print(f"wrote {DIR / f'calib_{args.label}.yaml'} and {DIR / f'undistorted_{args.label}.png'}")
