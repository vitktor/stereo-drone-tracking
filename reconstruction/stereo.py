"""Stereo geometry of the calibrated rig: undistortion, epipolar check, triangulation and projection.

Loads calibration/calib_A.yaml, calib_B.yaml (intrinsics, rational model) and calib_stereo.yaml (X_B = R X_A + T, mm).
3D points are in camera A's frame, in mm. Captured files are named by Pi camera index: cam0 is module B, cam1 is A.
"""
from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).resolve().parent.parent / "calibration"
LABEL = {0: "B", 1: "A"}  # Pi camera index -> module label
CRIT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)  # the default 5 iterations fall short near the edges of the wide lens


def _read(name, *keys):
    fs = cv2.FileStorage(str(CALIB / name), cv2.FILE_STORAGE_READ)
    return [fs.getNode(k).mat() for k in keys]


class Rig:
    def __init__(self):
        self.K, self.D = {}, {}
        for c in "AB":
            self.K[c], self.D[c] = _read(f"calib_{c}.yaml", "K", "D")
        self.R, self.T, self.E = _read("calib_stereo.yaml", "R", "T", "E")
        self.pose = {"A": (np.eye(3), np.zeros((3, 1))), "B": (self.R, self.T)}  # world (= A) -> camera
        self.P = {c: np.hstack(rt) for c, rt in self.pose.items()}

    def normalize(self, cam, pts):
        """Pixel points (N, 2) of camera 'A' or 'B' -> undistorted normalized coordinates (N, 2)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K[cam], self.D[cam], criteria=CRIT).reshape(-1, 2)

    def epipolar_px(self, pa, pb):
        """Distance in B's pixels from each point pb to the epipolar line of its pa; a few px for a true match."""
        xa = np.c_[self.normalize("A", pa), np.ones(len(pa))]
        xb = np.c_[self.normalize("B", pb), np.ones(len(pb))]
        lines = xa @ self.E.T  # line in B's normalized image for each xa
        d = np.abs((lines * xb).sum(1)) / np.hypot(lines[:, 0], lines[:, 1])
        return d * self.K["B"][0, 0]

    def triangulate(self, pa, pb):
        """Matching pixel points (N, 2) in A and B -> 3D points (N, 3) in A's frame, mm."""
        X = cv2.triangulatePoints(self.P["A"], self.P["B"], self.normalize("A", pa).T, self.normalize("B", pb).T)
        return (X[:3] / X[3]).T

    def project(self, cam, X):
        """3D points (N, 3) in A's frame -> distorted pixel points (N, 2) of camera 'A' or 'B'."""
        R, T = self.pose[cam]
        px, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), cv2.Rodrigues(R)[0], T, self.K[cam], self.D[cam])
        return px.reshape(-1, 2)

    def center(self, cam):
        """Camera centre in A's frame, mm."""
        R, T = self.pose[cam]
        return (-R.T @ T).ravel()
