import zmq, numpy as np, cv2

PI_IP, W, H = "10.42.0.2", 1280, 800
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.RCVHWM, 4)
s.connect(f"tcp://{PI_IP}:5555")
s.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    ts, dt, a, b = s.recv_multipart()
    print(f"got pair, offset {np.frombuffer(dt, np.int64)[0]/1000:.1f} us")
    ts = np.frombuffer(ts, np.int64)[0]; dt = np.frombuffer(dt, np.int64)[0]
    left  = np.frombuffer(a, np.uint8).reshape(H, W)
    right = np.frombuffer(b, np.uint8).reshape(H, W)
    view = cv2.resize(np.hstack([left, right]), None, fx=0.5, fy=0.5)
    cv2.putText(view, f"offset {dt/1000:.1f} us", (10, 30), 0, 1, 255, 2)
    cv2.imshow("stereo", view)
    if cv2.waitKey(1) == 27:   # Esc to quit
        break
