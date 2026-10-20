"""Runs on the Pi: streams ONE camera for calibration (capture.py on the Mac receives it).

Usage: python3 pi_stream.py [--exposure US] [--gain G]   (default auto exposure; set a short exposure if frames are blurry)
Message: [sensor timestamp int64, 1280x800 uint8 frame]
"""
import argparse
import time

import numpy as np
import zmq
from picamera2 import Picamera2

FPS, W, H = 30, 1280, 800

p = argparse.ArgumentParser()
p.add_argument("--exposure", type=int, help="fixed exposure in microseconds (default: auto)")
p.add_argument("--gain", type=float, default=1.0, help="analogue gain used with --exposure (brightens, adds noise)")
args = p.parse_args()

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 4)
sock.bind("tcp://*:5555")

cam = Picamera2(0)
cam.configure(cam.create_video_configuration(main={"size": (W, H), "format": "YUV420"}, buffer_count=6))
ctrl = {"FrameRate": FPS}
if args.exposure:
    ctrl.update(AeEnable=False, ExposureTime=args.exposure, AnalogueGain=args.gain)
cam.set_controls(ctrl)
cam.start()

n, t_last = 0, time.time()
while True:
    req = cam.capture_request()
    md = req.get_metadata()
    y = req.make_array("main")[:H, :W].copy()
    req.release()
    sock.send_multipart([np.int64(md["SensorTimestamp"]).tobytes(), y.tobytes()])
    n += 1
    if time.time() - t_last > 1:
        print(f"{n} frames/s, exposure {md['ExposureTime']} us, gain {md['AnalogueGain']:.2f}")
        n, t_last = 0, time.time()
