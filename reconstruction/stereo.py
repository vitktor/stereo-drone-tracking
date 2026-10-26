"""Stereo geometry of the calibrated rig: undistortion, epipolar check, triangulation and projection.

Loads calibration/calib_A.yaml, calib_B.yaml (intrinsics, rational model) and calib_stereo.yaml (X_B = R X_A + T, mm).
3D points are in camera A's frame, in mm. Captured files are named by Pi camera index: cam0 is module B, cam1 is A.
World frame (Rig.world): z up from the floor, x along the baseline from camera A to B, y into the room, metres.
"""
from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).resolve().parent.parent / "calibration"
LABEL = {0: "B", 1: "A"}  # Pi camera index -> module label
CRIT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)  # the default 5 iterations fall short near the edges of the wide lens
W, H = 1280, 800
UP_A = np.array([-0.035, -0.931, 0.364])  # up in camera A's frame, from vertical room edges (A and B agree within 1 deg)
CAM_HEIGHT = 1.0  # m, cameras above the floor (tape-measured roughly)


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
        up = UP_A / np.linalg.norm(UP_A)
        ex = self.center("B") - self.center("A")
        ex -= (ex @ up) * up
        ex /= np.linalg.norm(ex)
        self.mid = (self.center("A") + self.center("B")) / 2
        self.to_w = np.array([ex, np.cross(up, ex), up])  # rows: world axes in A's frame

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

    def world(self, P):
        """Points (N, 3) in A's frame, mm -> world, m."""
        return (np.asarray(P) - self.mid) @ self.to_w.T / 1000 + [0, 0, CAM_HEIGHT]

    def from_world(self, Wp):
        """World points (N, 3), m -> A's frame, mm."""
        return (np.asarray(Wp) - [0, 0, CAM_HEIGHT]) * 1000 @ self.to_w + self.mid

    def locate(self, img, box, max_epi=5.0):
        """The drone in one synced pair: img and box are dicts 'A'/'B' of grayscale frames and YOLO boxes (x1, y1, x2, y2).
        The point in each view is the centroid of the dark drone blob inside the box, which the two views agree on
        better than the box centre. When the blob also takes in dark background (e.g. at a window edge) and the views
        no longer agree, the box centres are tried instead and the pair closer to the epipolar line is kept.
        Returns dict(pt={'A', 'B'} px, point='blob'|'box', epi px, edge, ok, X (3,) mm in A's frame); ok means within
        max_epi px of the epipolar line and neither box cut by the image edge (only part of the drone visible)."""
        pt = {c: blob_centroid(img[c], box[c]) for c in "AB"}
        epi, point = self.epipolar_px(pt["A"][None], pt["B"][None])[0], "blob"
        if epi > max_epi:
            mid = {c: np.array([(box[c][0] + box[c][2]) / 2, (box[c][1] + box[c][3]) / 2]) for c in "AB"}
            epi_mid = self.epipolar_px(mid["A"][None], mid["B"][None])[0]
            if epi_mid < epi:
                pt, epi, point = mid, epi_mid, "box"
        edge = at_edge(box["A"]) or at_edge(box["B"])
        return dict(pt=pt, point=point, epi=epi, edge=edge, ok=epi <= max_epi and not edge,
                    X=self.triangulate(pt["A"][None], pt["B"][None])[0])


def blob_centroid(img, box, pad=4):
    """Centroid of the largest dark region in the (slightly padded) box: the drone body and arms."""
    x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
    roi = cv2.GaussianBlur(img[y1:box[3] + pad, x1:box[2] + pad], (3, 3), 0)
    _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    return centroids[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] + (x1, y1)


def at_edge(box, margin=2):
    return box[0] <= margin or box[1] <= margin or box[2] >= W - margin or box[3] >= H - margin
