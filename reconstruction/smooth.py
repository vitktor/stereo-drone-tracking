"""Constant-velocity Kalman filter with Rauch-Tung-Striebel smoothing for a 3D trajectory sampled at known times.

Offline, so the backward pass also uses later frames: the path comes out smoother than a live filter's, short gaps
(drone missed in one view) are bridged, and points that jump away from the path are dropped as outliers.
"""
import numpy as np


def smooth(t, X, sigma=15.0, accel=8000.0, gate=4.0, max_gap=0.5):
    """t (N,) s, X (N, 3) mm with NaN rows where there is no measurement.
    sigma: measurement noise (mm); accel: random acceleration (mm/s^2); gate: outlier limit in standard deviations;
    max_gap: longest stretch (s) without measurements that is bridged; longer gaps start a new segment.
    Returns smoothed positions (N, 3) mm (NaN outside bridged segments) and a bool mask of measurements used."""
    N = len(t)
    have = ~np.isnan(X).any(1)
    used = np.zeros(N, bool)
    out = np.full((N, 3), np.nan)
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    Rm = np.eye(3) * sigma ** 2

    # segments: runs of frames whose measurements are at most max_gap apart
    idx = np.flatnonzero(have)
    if not len(idx):
        return out, used
    breaks = np.flatnonzero(np.diff(t[idx]) > max_gap)
    for seg in np.split(idx, breaks + 1):
        lo, hi = seg[0], seg[-1] + 1
        xs, Ps, xp, Pp, Fs = [], [], [], [], []
        x = np.r_[X[lo], 0, 0, 0]
        P = np.diag([sigma ** 2] * 3 + [2000.0 ** 2] * 3)  # velocity unknown at the start
        for k in range(lo, hi):
            dt = t[k] - t[k - 1] if k > lo else 0.0
            F = np.eye(6)
            F[:3, 3:] = np.eye(3) * dt
            G = np.r_[np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]
            x, P = F @ x, F @ P @ F.T + G @ G.T * accel ** 2
            xp.append(x), Pp.append(P), Fs.append(F)
            if have[k]:
                r = X[k] - H @ x
                S = H @ P @ H.T + Rm
                if r @ np.linalg.solve(S, r) <= gate ** 2 or k == lo:  # 4 sigma ~ 99.9 % for 3 dof
                    K = P @ H.T @ np.linalg.inv(S)
                    x, P = x + K @ r, (np.eye(6) - K @ H) @ P
                    used[k] = True
            xs.append(x), Ps.append(P)
        for i in range(len(xs) - 2, -1, -1):  # RTS backward pass
            C = Ps[i] @ Fs[i + 1].T @ np.linalg.inv(Pp[i + 1])
            xs[i] = xs[i] + C @ (xs[i + 1] - xp[i + 1])
            Ps[i] = Ps[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
        out[lo:hi] = np.array(xs)[:, :3]
    return out, used
