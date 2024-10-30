# NormalFlow: Fast and Accurate Tactile-based Object Pose Tracking

<p align="center">
  <img src="assets/demo.gif" alt="NormalFlow Demo">
</p>

This repository is the official implementation of our paper, *NormalFlow: Fast, Robust, and Accurate Contact-based Object 6DoF Pose Tracking with Vision-based Tactile Sensors* ([Paper Link](TODO: Link)). NormalFlow is significantly more accurate and robust compared to other tactile-based tracking methods and performs well even on low-textured objects like an egg or a flat table. It operates at 70 Hz on a standard CPU. If you use this package, please cite our paper:
[TODO: Citation block]

Additionally, two other repositories comes with the paper:
* The object tracking dataset is available [here](TODO: link).
* Baseline implementations and experiments for comparing NormalFlow with other methods on the dataset are available [here](TODO: link).


## Support System
* Tested on Ubuntu 22.04
* Tested on GelSight Mini and Digit
* Python >= 3.9
* For the demo and example, install [gs_sdk](TODO: link).

## Installation
Clone and install normalflow from source:
```bash
git clone [TODO:link]
cd normalflow
pip install -e .
```

## Real-time Demo
Connect a GelSight Mini sensor (without markers) to your machine and run the command below to start a real-time object tracking demo.
```bash
realtime_object_tracking
```

After starting, wait a few seconds for a window to appear. Tracking will begin once an object contacts the sensor. Press any key to exit.

* Note: For other GelSight sensors, please use the GelSight SDK Calibration tool to calibrate. Supply the configuration file and calibrated model path as arguments to run the demo with other GelSight sensors.
* Note: This demo also serves as an implementation of the long-horizon tracking algorithm presented in the paper.

## Examples
This example demonstrates basic usage of NormalFlow. Run the command below to test the tracking algorithm.
```bash
test_tracking
```
The command reads the tactile video `examples/data/tactile_video.avi`, tracks the touched object, and saves the result in `examples/data/tracked_tactile_video.avi`.