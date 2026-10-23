# Real-time 3D Drone Tracking

A stereo camera rig on a Raspberry Pi 5 that detects drones in the sky in real time and tracks them in 3D.
Two global-shutter OV9281-120 cameras sit on a balcony railing and send synchronized frames to the Pi.
YOLO detects the drone in each view, and the calibrated camera pair triangulates its position in 3D.

## Hardware setup

Each camera's CSI ribbon goes into a CSI-to-HDMI adapter, so a standard HDMI cable can carry the signal along the railing to the Pi. All mounts are 3D-printed and clamp onto the railing; the models are in [3d-models/](3d-models/).

![Stereo drone tracking rig: two OV9281-120 cameras and a Raspberry Pi 5 with CSI-to-HDMI adapters mounted on a balcony railing](stereo-rig-overview.jpg)

## Calibrating intrinsics

Each camera is calibrated on its own from 60–70 photos of a printed checkerboard, using OpenCV's rational distortion model to handle the wide-angle lens. The results are in [calibration/](calibration/): camera A reaches 0.17 px RMS reprojection error and camera B 0.26 px.

![16 of the checkerboard calibration frames from camera A](calibration/checkerboards_A.jpg)

## Calibrating extrinsics

### Using a checkerboard

A checkerboard on a rigid backing is moved through the area both cameras see, and 48 sharp synchronized pairs with the board fully visible in both views are kept. One row of squares is cut off the board (10×9 inner corners, 40 mm squares) so each corner gets the same number in both views. A square board can be numbered from a different corner in each camera. With each camera's intrinsics held fixed, OpenCV's stereo calibration reaches 0.43 px RMS and puts the cameras 1.96 m apart, matching the tape-measured distance.

![A synchronized stereo pair with the detected checkerboard corners; matching colored rings mark the same four board corners in both views](calibration/stereo_pair.jpg)

### Using the drone's LEDs
