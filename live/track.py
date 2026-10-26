"""Tracks the drone live on the Pi: synced capture from both cameras, YOLO in each view, triangulation, Kalman filter.

Usage (on the Pi): python live/track.py [--model best_ncnn_model] [--session NAME] [--max-epi 5] [--pub]
Needs the cameras to itself, so stop sync_stream.py first. Ctrl+C stops it.
Output: data/live/<SESSION>.csv, one row per processed pair; a status line every second on the console;
with --pub, each pair's result (JSON) and both frames (half-size JPEG) over ZMQ on port 5556 for live/view.py.
A capture thread keeps syncing the cameras at 30 fps (with the same pairing and exposure mirroring as
test-sync/sync_stream.py) and holds only the newest pair. The main loop takes that pair whenever it is free,
so frames are skipped rather than queued when YOLO is slower than the cameras, and the result stays current.
Positions are in the world frame of stereo.Rig.world: z up from the floor, x from camera A to B, y into the room, m.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
from libcamera import controls
from picamera2 import Picamera2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reconstruction"))
from stereo import H, LABEL, W, Rig  # noqa: E402

from filter import Track  # noqa: E402

FPS = 30
EV = -1.0  # bias AE darker so the sky doesn't clip

p = argparse.ArgumentParser()
p.add_argument("--model", default=str(ROOT / "best_ncnn_model"), help="NCNN export (or best.pt)")
p.add_argument("--session", default=time.strftime("live-%Y%m%d-%H%M%S"))
p.add_argument("--imgsz", type=int, default=640, help="same as training/export")
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--max-epi", type=float, default=5.0, help="px from the epipolar line to accept a pair")
p.add_argument("--pub", action="store_true", help="publish results and frames for live/view.py")
p.add_argument("--port", type=int, default=5556)
args = p.parse_args()


def setup(idx, mode, ae):
    cam = Picamera2(idx)
    cfg = cam.create_video_configuration(main={"size": (W, H), "format": "YUV420"}, buffer_count=6)
    cam.configure(cfg)
    ctrl = {"SyncMode": mode, "FrameRate": FPS, "AeEnable": ae}
    if ae:
        ctrl |= {"AeExposureMode": controls.AeExposureModeEnum.Short, "ExposureValue": EV}
    else:
        ctrl |= {"ExposureTime": 1000, "AnalogueGain": 1.0}
    cam.set_controls(ctrl)
    return cam


def grab(cam):
    req = cam.capture_request()
    md = req.get_metadata()
    y = req.make_array("main")[:H, :W].copy()
    req.release()
    return md, y


latest, new_pair = None, threading.Condition()  # newest synced pair: (sensor timestamp ns, offset ns, {"A", "B"} frames)
captured = 0


def capture():
    global latest, captured
    # AE runs on the server only; its exposure/gain is mirrored to the client
    c0 = setup(0, controls.rpi.SyncModeEnum.Server, True)
    c1 = setup(1, controls.rpi.SyncModeEnum.Client, False)
    c1.start(), c0.start()
    half, t_ae = 0.5e9 / FPS, time.time()
    while True:
        m0, f0 = grab(c0)
        m1, f1 = grab(c1)
        if not (m0.get("SyncReady", True) and m1.get("SyncReady", True)):
            continue
        dt = m1["SensorTimestamp"] - m0["SensorTimestamp"]
        while dt > half:
            m0, f0 = grab(c0)
            dt = m1["SensorTimestamp"] - m0["SensorTimestamp"]
        while dt < -half:
            m1, f1 = grab(c1)
            dt = m1["SensorTimestamp"] - m0["SensorTimestamp"]
        with new_pair:
            latest = (m0["SensorTimestamp"], dt, {LABEL[0]: f0, LABEL[1]: f1})
            captured += 1
            new_pair.notify()
        if time.time() - t_ae > 1:
            c1.set_controls({"ExposureTime": m0["ExposureTime"], "AnalogueGain": m0["AnalogueGain"]})
            t_ae = time.time()


def detect(img):
    """Grayscale frame -> (box [x1, y1, x2, y2], conf) of the most confident drone, or None."""
    r = model.predict(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), imgsz=args.imgsz, conf=args.conf, max_det=1,
                      verbose=False)[0]
    if not len(r.boxes):
        return None
    return [int(v) for v in r.boxes.xyxy[0].tolist()], float(r.boxes.conf[0])


model = YOLO(args.model, task="detect")
model.predict(np.zeros((H, W, 3), np.uint8), imgsz=args.imgsz, verbose=False)  # load and warm up before timing
rig = Rig()
track = Track()
OUT = ROOT / "data" / "live"
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / f"{args.session}.csv", "w")
log.write("t_s,timestamp_ns,offset_us,conf_A,x1_A,y1_A,x2_A,y2_A,conf_B,x1_B,y1_B,x2_B,y2_B,point,u_A,v_A,u_B,v_B,"
          "epi_px,edge,ok,x_m,y_m,z_m,used,xf_m,yf_m,zf_m,latency_ms,detect_ms\n")
if args.pub:
    sock = zmq.Context().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://*:{args.port}")

threading.Thread(target=capture, daemon=True).start()
print(f"{args.session}: waiting for synced frames...")
fmt = lambda v, d: "" if v is None or np.isnan(v) else f"{v:.{d}f}"
t0, last_ts = None, 0
stat = dict(n=0, A=0, B=0, ok=0, used=0, lat=[], t=time.time(), captured=0)
try:
    while True:
        with new_pair:
            new_pair.wait_for(lambda: latest is not None and latest[0] != last_ts, timeout=5)
            if latest is None or latest[0] == last_ts:
                print("no frames from the cameras; is sync_stream.py still running?")
                continue
            ts, dt, img = latest
        last_ts, t0 = ts, t0 or ts
        t = (ts - t0) / 1e9

        t_det = time.perf_counter()
        found = {c: detect(img[c]) for c in "AB"}
        det_ms = (time.perf_counter() - t_det) * 1000
        res = None
        if found["A"] and found["B"]:
            res = rig.locate(img, {c: found[c][0] for c in "AB"}, args.max_epi)
        Xf, used = track.update(t, res["X"] if res and res["ok"] else None)
        Xw = rig.world(res["X"][None])[0] if res else None
        Xfw = rig.world(Xf[None])[0] if Xf is not None else None
        latency = (time.clock_gettime_ns(time.CLOCK_BOOTTIME) - ts) / 1e6  # sensor timestamps are boot time

        cols = [f"{t:.3f}", str(ts), f"{dt / 1000:.1f}"]
        for c in "AB":
            cols += [f"{found[c][1]:.3f}", *map(str, found[c][0])] if found[c] else [""] * 5
        if res:
            cols += [res["point"], *(f"{v:.2f}" for c in "AB" for v in res["pt"][c]), f"{res['epi']:.2f}",
                     str(int(res["edge"])), str(int(res["ok"])), *(f"{v:.3f}" for v in Xw)]
        else:
            cols += [""] * 5 + ["", "0", "0"] + [""] * 3
        cols += [str(int(used)), *(fmt(v, 3) for v in (Xfw if Xfw is not None else [None] * 3)),
                 f"{latency:.0f}", f"{det_ms:.0f}"]
        log.write(",".join(cols) + "\n")

        if args.pub:
            view = cv2.resize(np.hstack([img["A"], img["B"]]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            msg = dict(t=t, box={c: found[c] for c in "AB"},
                       pt={c: res["pt"][c].tolist() for c in "AB"} if res else None,
                       epi=float(res["epi"]) if res else None, ok=bool(res and res["ok"]),
                       X=Xw.tolist() if res else None, Xf=Xfw.tolist() if Xfw is not None else None,
                       used=used, latency_ms=latency, detect_ms=det_ms)
            sock.send_multipart([json.dumps(msg).encode(), cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]])

        stat["n"] += 1
        stat["A"] += found["A"] is not None
        stat["B"] += found["B"] is not None
        stat["ok"] += bool(res and res["ok"])
        stat["used"] += used
        stat["lat"].append(latency)
        if time.time() - stat["t"] >= 1:
            secs, n = time.time() - stat["t"], stat["n"]
            where = "no track" if Xfw is None else "drone at x {:+.2f} y {:.2f} z {:.2f} m".format(*Xfw)
            print(f"{n / secs:4.1f} pairs/s ({(captured - stat['captured']) / secs:.0f} captured) | "
                  f"found A {stat['A']}/{n} B {stat['B']}/{n} | 3D {stat['ok']}/{n} used {stat['used']} | "
                  f"latency {np.median(stat['lat']):.0f} ms | {where}", flush=True)
            stat = dict(n=0, A=0, B=0, ok=0, used=0, lat=[], t=time.time(), captured=captured)
except KeyboardInterrupt:
    pass
log.close()
print(f"-> {OUT / args.session}.csv")
