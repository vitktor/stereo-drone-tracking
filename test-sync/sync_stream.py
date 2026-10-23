from picamera2 import Picamera2
from libcamera import controls
import zmq, numpy as np, time

FPS, W, H = 30, 1280, 800
EV = -1.0   # bias AE darker so the sky doesn't clip

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 4)
sock.bind("tcp://*:5555")

def setup(idx, mode, ae):
    cam = Picamera2(idx)
    cfg = cam.create_video_configuration(main={"size": (W, H), "format": "YUV420"},
                                         buffer_count=6)
    cam.configure(cfg)
    ctrl = {"SyncMode": mode, "FrameRate": FPS, "AeEnable": ae}
    if ae:
        ctrl |= {"AeExposureMode": controls.AeExposureModeEnum.Short, "ExposureValue": EV}
    else:
        ctrl |= {"ExposureTime": 1000, "AnalogueGain": 1.0}
    cam.set_controls(ctrl)
    return cam

# AE runs on the server only; its exposure/gain is mirrored to the client
c0 = setup(0, controls.rpi.SyncModeEnum.Server, True)
c1 = setup(1, controls.rpi.SyncModeEnum.Client, False)
c1.start(); c0.start()

def grab(cam):
    req = cam.capture_request()
    md = req.get_metadata()
    y = req.make_array("main")[:H, :W].copy()
    req.release()
    return md, y

half = 0.5e9 / FPS
n, t_last = 0, time.time()
print("Waiting for sync...")
while True:
    m0, f0 = grab(c0); m1, f1 = grab(c1)
    if not (m0.get("SyncReady", True) and m1.get("SyncReady", True)):
        continue
    dt = m1["SensorTimestamp"] - m0["SensorTimestamp"]
    while dt > half:
        m0, f0 = grab(c0); dt = m1["SensorTimestamp"] - m0["SensorTimestamp"]
    while dt < -half:
        m1, f1 = grab(c1); dt = m1["SensorTimestamp"] - m0["SensorTimestamp"]
    sock.send_multipart([np.int64(m0["SensorTimestamp"]).tobytes(),
                         np.int64(dt).tobytes(), f0.tobytes(), f1.tobytes()])
    n += 1
    if time.time() - t_last > 1:
        c1.set_controls({"ExposureTime": m0["ExposureTime"], "AnalogueGain": m0["AnalogueGain"]})
        print(f"{n} pairs/s, offset = {dt/1000:.1f} us, "
              f"exp A/B = {m0['ExposureTime']}/{m1['ExposureTime']} us, "
              f"gain A/B = {m0['AnalogueGain']:.2f}/{m1['AnalogueGain']:.2f}")
        n, t_last = 0, time.time()
