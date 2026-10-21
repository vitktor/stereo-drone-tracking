# Real-time 3D Drone Tracking

A stereo camera rig on a Raspberry Pi 5 that detects drones in the sky in real time and tracks them in 3D.
Two global-shutter OV9281-120 cameras sit on a balcony railing and send synchronized frames to the Pi.
YOLO detects the drone in each view, and the calibrated camera pair triangulates its position in 3D.

## Hardware setup

Each camera's CSI ribbon goes into a CSI-to-HDMI adapter, so a standard HDMI cable can carry the signal along the railing to the Pi. All mounts are 3D-printed and clamp onto the railing; the models are in [3d-models/](3d-models/).

![Stereo drone tracking rig: two OV9281-120 cameras and a Raspberry Pi 5 with CSI-to-HDMI adapters mounted on a balcony railing](stereo-rig-overview.jpg)

## Calibrating intrinsics

## Calibrating extrinsics
