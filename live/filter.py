"""Forward-only constant-velocity Kalman filter for live tracking: the online counterpart of reconstruction/smooth.py.

It sees each point once, as it arrives, so it can't look ahead like the offline smoother. Same motion model and
defaults: points that jump away from the predicted path are rejected, and short misses (drone not found in one view)
are coasted through on the predicted position. A track whose last accepted point is older than max_gap is dropped,
and a run of rejected points restarts it (the track was locked onto a bad first point, or the drone really jumped).
"""
import numpy as np


class Track:
    def __init__(self, sigma=15.0, accel=8000.0, gate=4.0, max_gap=0.5, restart=5):
        """sigma: measurement noise (mm); accel: random acceleration (mm/s^2); gate: outlier limit in standard
        deviations; max_gap: longest time (s) coasted without an accepted point; restart: rejected points in a row
        that restart the track."""
        self.sigma, self.accel, self.gate, self.max_gap, self.restart = sigma, accel, gate, max_gap, restart
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.x = None  # state: position (mm), velocity (mm/s)
        self.t = self.t_seen = None
        self.rejected = 0

    def _start(self, t, X):
        self.x = np.r_[X, 0, 0, 0]
        self.P = np.diag([self.sigma ** 2] * 3 + [2000.0 ** 2] * 3)  # velocity unknown at the start
        self.t = self.t_seen = t
        self.rejected = 0

    def update(self, t, X=None):
        """t: time (s); X: measured point (3,) mm, or None when there is none for this frame.
        Returns (position (3,) mm or None when there is no track, whether X was accepted)."""
        if self.x is not None and t - self.t_seen > self.max_gap:
            self.x = None
        if self.x is None:
            if X is None:
                return None, False
            self._start(t, X)
            return self.x[:3].copy(), True

        dt = t - self.t
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        G = np.r_[np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]
        self.x, self.P = F @ self.x, F @ self.P @ F.T + G @ G.T * self.accel ** 2
        self.t = t
        used = False
        if X is not None:
            r = X - self.H @ self.x
            S = self.H @ self.P @ self.H.T + np.eye(3) * self.sigma ** 2
            if r @ np.linalg.solve(S, r) <= self.gate ** 2:  # 4 sigma ~ 99.9 % for 3 dof
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x, self.P = self.x + K @ r, (np.eye(6) - K @ self.H) @ self.P
                self.t_seen, self.rejected, used = t, 0, True
            else:
                self.rejected += 1
                if self.rejected >= self.restart:
                    self._start(t, X)
                    used = True
        return self.x[:3].copy(), used
