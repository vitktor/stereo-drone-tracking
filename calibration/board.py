"""Calibration-board detection shared by capture.py and calibrate.py.

Checkerboard: --cols/--rows are INNER corners (a board of 10x7 squares has 9x6 inner corners).
ChArUco:      --cols/--rows are SQUARES, plus --marker size and --dict.
"""
import argparse

import cv2
import numpy as np


def add_args(p: argparse.ArgumentParser):
    p.add_argument("--pattern", choices=["checker", "charuco"], default="checker")
    p.add_argument("--cols", type=int, default=10, help="checker: inner corners across; charuco: squares across")
    p.add_argument("--rows", type=int, default=10, help="checker: inner corners down; charuco: squares down")
    p.add_argument("--square", type=float, default=40.0, help="square side in mm (measure the print!)")
    p.add_argument("--marker", type=float, help="charuco: marker side in mm")
    p.add_argument("--dict", default="DICT_5X5_100", help="charuco: aruco dictionary name")
    p.add_argument("--legacy", action="store_true", help="charuco: board generated with the pre-4.6 OpenCV layout")


class Board:
    def __init__(self, a):
        self.a = a
        if a.pattern == "checker":
            grid = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2)
            self.obj = np.hstack([grid * a.square, np.zeros((len(grid), 1))]).astype(np.float32)
            self.min_pts = len(self.obj)
        else:
            if a.marker is None:
                raise SystemExit("--marker is required for charuco")
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict))
            self.board = cv2.aruco.CharucoBoard((a.cols, a.rows), a.square, a.marker, d)
            self.board.setLegacyPattern(a.legacy)
            self.detector = cv2.aruco.CharucoDetector(self.board)
            self.obj = self.board.getChessboardCorners().astype(np.float32)
            self.min_pts = 8

    def detect(self, gray):
        """Return (ids, image_points Nx2) or None. ids index into self.obj."""
        if self.a.pattern == "checker":
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, (self.a.cols, self.a.rows), flags=flags)
            if not ok:
                return None
            return np.arange(len(self.obj)), corners.reshape(-1, 2).astype(np.float32)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < self.min_pts:
            return None
        return ids.flatten(), corners.reshape(-1, 2).astype(np.float32)
