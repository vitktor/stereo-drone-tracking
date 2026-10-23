"""Calibrate the extrinsics between cameras A and B from synced pairs ./images/ext/NNN_A.png + NNN_B.png.

Usage: python calibrate_stereo.py --cols 10 --rows 9 --square 40   (see board.py for the pattern args)
Intrinsics are taken from calib_A.yaml / calib_B.yaml and held fixed.
Writes calib_stereo.yaml: R, T map a point from A's camera frame to B's (X_B = R X_A + T, mm),
plus E, F and projection matrices P_A = [I|0], P_B = [R|T] for cv2.triangulatePoints on undistorted points
(cv2.undistortPoints(pts, K, D) without P gives normalised coordinates that fit these P matrices).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from board import Board, add_args

DIR = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
add_args(p)
args = p.parse_args()
board = Board(args)


def load(label):
    fs = cv2.FileStorage(str(DIR / f"calib_{label}.yaml"), cv2.FILE_STORAGE_READ)
    return fs.getNode("K").mat(), fs.getNode("D").mat()


(KA, DA), (KB, DB) = load("A"), load("B")
files = sorted((DIR / "images" / "ext").glob("*_A.png"))
if not files:
    raise SystemExit("no images/ext/NNN_A.png + NNN_B.png pairs")
size = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE).shape[::-1]

pairs = []
for fa in files:
    fb = fa.with_name(fa.name.replace("_A", "_B"))
    da = board.detect(cv2.imread(str(fa), cv2.IMREAD_GRAYSCALE))
    db = board.detect(cv2.imread(str(fb), cv2.IMREAD_GRAYSCALE)) if fb.exists() else None
    if da is None or db is None:
        print(f"  skip {fa.stem[:-2]}: board not found in both")
        continue
    ids = np.intersect1d(da[0], db[0])  # charuco may see different corners in each view
    if len(ids) < board.min_pts:
        continue
    pa, pb = (np.array([dict(zip(d[0], d[1]))[i] for i in ids]) for d in (da, db))
    pairs.append((fa.stem[:-2], board.obj[ids], pa, pb))

rms, *_, R, T, E, F, rvecs, tvecs, per_view = cv2.stereoCalibrateExtended(
    [o for _, o, _, _ in pairs], [a for _, _, a, _ in pairs], [b for _, _, _, b in pairs],
    KA, DA, KB, DB, size, None, None, flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)
per_view = per_view.reshape(-1, 2)

# sanity check: triangulate every board corner and compare neighbouring corners with the true square size
PA, PB = np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, T])
sq_err = []
for _, o, a, b in pairs:
    na = cv2.undistortPoints(a.reshape(-1, 1, 2), KA, DA).reshape(-1, 2).T
    nb = cv2.undistortPoints(b.reshape(-1, 1, 2), KB, DB).reshape(-1, 2).T
    X = cv2.triangulatePoints(PA, PB, na, nb)
    X = (X[:3] / X[3]).T
    d_obj = np.linalg.norm(np.diff(o, axis=0), axis=1)
    d_tri = np.linalg.norm(np.diff(X, axis=0), axis=1)
    near = np.isclose(d_obj, args.square)  # consecutive corners one square apart (skips row wraps)
    sq_err.extend(d_tri[near] - args.square)
sq_err = np.array(sq_err)

angle = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
dist = [np.linalg.norm(t) / 1000 for t in tvecs]
print(f"{len(pairs)}/{len(files)} pairs used, stereo RMS {rms:.3f} px")
print(f"  baseline {np.linalg.norm(T):.0f} mm  T {np.round(T.ravel(), 1).tolist()} mm  rotation {angle:.1f} deg")
print(f"  board distance from A {min(dist):.1f}-{max(dist):.1f} m")
print(f"  triangulated square size {args.square + sq_err.mean():.2f} mm (true {args.square}), "
      f"std {sq_err.std():.2f} mm")
worst = sorted(zip(per_view.max(axis=1), (n for n, *_ in pairs)), reverse=True)[:5]
print("  worst pairs: " + ", ".join(f"{n} {e:.2f}" for e, n in worst))

fs = cv2.FileStorage(str(DIR / "calib_stereo.yaml"), cv2.FILE_STORAGE_WRITE)
fs.write("image_width", size[0]); fs.write("image_height", size[1])
fs.write("cam1", "A"); fs.write("cam2", "B")
fs.write("rms", rms); fs.write("pairs", len(pairs))
fs.write("R", R); fs.write("T", T); fs.write("E", E); fs.write("F", F)
fs.write("P_A", PA); fs.write("P_B", PB)
fs.release()
print(f"wrote {DIR / 'calib_stereo.yaml'}")
