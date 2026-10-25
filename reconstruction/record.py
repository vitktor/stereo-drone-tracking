"""Records every synced stereo pair from the Pi for a flight video (e.g. the README demo), with no dropped frames.
On the Pi run test-sync/sync_stream.py over the Ethernet link (Wi-Fi only carries about 5 pairs/s).

Usage: python record.py SESSION [--delay 10] [--duration 60] [--writers 4]
Keys:  SPACE start/stop recording   ESC quit (stops recording first)
Output: data/drone/<SESSION>/<SESSION>_NNNN_cam0.png + _cam1.png (cam0 = module B, cam1 = module A) and pairs.csv,
same layout as detection/capture.py, so detection/detect_session.py runs on it unchanged.
Unlike capture.py it keeps every pair: frames go to a queue that writer threads save as fast PNGs.
Missing pairs show up as gaps in the sensor timestamps, which are counted live and in the summary.
"""
import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

PI_IP, W, H, FPS = "10.42.0.2", 1280, 800, 30
DATA = Path(__file__).resolve().parent.parent / "data" / "drone"
VIEW_EVERY = 1 / 15  # s between live view refreshes, so drawing never holds up receiving

p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--delay", type=float, help="start recording automatically after this many seconds (time to take off)")
p.add_argument("--duration", type=float, help="stop recording and quit after this many seconds of recording")
p.add_argument("--writers", type=int, default=4, help="PNG writer threads")
p.add_argument("--pi", default=PI_IP)
args = p.parse_args()
OUT = DATA / args.session
if OUT.exists() and any(OUT.glob("*.png")):
    raise SystemExit(f"{OUT} already has frames; pick a new session name")
OUT.mkdir(parents=True, exist_ok=True)

todo = queue.Queue()  # (file name, frame); unbounded, so a slow disk costs RAM instead of frames


def writer():
    while (job := todo.get()) is not None:
        cv2.imwrite(str(OUT / job[0]), job[1], [cv2.IMWRITE_PNG_COMPRESSION, 1])


writers = [threading.Thread(target=writer) for _ in range(args.writers)]
for t in writers:
    t.start()

ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.RCVHWM, 1000)
s.connect(f"tcp://{args.pi}:5555")
s.setsockopt(zmq.SUBSCRIBE, b"")

period = 1e9 / FPS
start = time.time() + args.delay if args.delay is not None else None  # wall-clock time when the countdown ends
recording, n, gaps, missing = False, 0, 0, 0
t_first = t_last = None  # sensor timestamps (ns) of the first and latest recorded pair
rate, rate_n, rate_t = 0.0, 0, time.time()
log = open(OUT / "pairs.csv", "w")
log.write("pair,cam0_file,cam1_file,cam0_timestamp_ns,offset_us\n")
cv2.namedWindow("record")
cv2.moveWindow("record", 0, 0)
shown = 0.0

while True:
    if not s.poll(1000):
        print("no frames from the Pi; is sync_stream.py running?")
        continue
    msg = s.recv_multipart()
    if len(msg) != 4:
        raise SystemExit("not a stereo stream; run test-sync/sync_stream.py on the Pi")
    ts = np.frombuffer(msg[0], np.int64)[0]
    dt = np.frombuffer(msg[1], np.int64)[0]
    frames = [np.frombuffer(m, np.uint8).reshape(H, W) for m in msg[2:]]

    rate_n += 1
    if time.time() - rate_t >= 1:
        rate, rate_n, rate_t = rate_n / (time.time() - rate_t), 0, time.time()

    if start is not None and time.time() >= start:
        recording, start = True, None
        print("recording started")
    if recording:
        if t_last is not None and ts - t_last > 1.5 * period:
            gaps += 1
            missing += round((ts - t_last) / period) - 1
        t_first = t_first if t_first is not None else ts
        t_last = ts
        names = [f"{args.session}_{n:04d}_cam{i}.png" for i in range(2)]
        for name, img in zip(names, frames):
            todo.put((name, img))
        log.write(f"{n},{names[0]},{names[1]},{ts},{dt / 1000:.1f}\n")
        n += 1

    stop = args.duration is not None and t_first is not None and (ts - t_first) / 1e9 >= args.duration
    if time.time() - shown >= VIEW_EVERY or stop:
        shown = time.time()
        view = cv2.cvtColor(cv2.resize(np.hstack(frames), None, fx=0.5, fy=0.5), cv2.COLOR_GRAY2BGR)
        if start is not None:
            cv2.putText(view, f"recording starts in {int(start - time.time()) + 1} s", (40, view.shape[0] // 2), 0, 1.6, (0, 220, 255), 4)
        if recording:
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 0, 255), 12)
            cv2.circle(view, (view.shape[1] - 40, 40), 14, (0, 0, 255), -1)
        status = f"{args.session}  {rate:.0f} pairs/s  recorded {n}  missing {missing}  write queue {todo.qsize()}"
        cv2.putText(view, status, (10, 30), 0, 0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.imshow("record", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if recording:
                stop = True
            else:
                recording, start = True, None
                print("recording started")
        if key == 27:
            stop = True
    if stop:
        break

cv2.destroyAllWindows()
log.close()
print(f"writing {todo.qsize()} queued frames...")
for _ in writers:
    todo.put(None)
for t in writers:
    t.join()
if n:
    secs = (t_last - t_first) / 1e9
    print(f"{n} pairs over {secs:.1f} s ({n / secs if secs else 0:.1f} pairs/s), "
          f"{missing} missing in {gaps} gaps -> {OUT}")
else:
    OUT.joinpath("pairs.csv").unlink()
    print("nothing recorded")
