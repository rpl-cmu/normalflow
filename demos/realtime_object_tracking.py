import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import numpy as np
import yaml

from gs_sdk.gs_device import Camera, FastCamera
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import normalflow, LoseTrackError
from normalflow.utils import Frame
from normalflow.viz_utils import annotate_coordinate_system

"""
This script demonstrates real-time using normalflow to track objects in contact.
The implementation is using the long-horizon tracking algorithm discussed in our GelSLAM paper.

Prerequisite:
    - Calibrate your GelSight sensor if you are not using the GelSight Mini sensor.
    - Connect the sensor to your computer.
Instruction:
    - Run this script, wait two seconds for background image collection.
    - Press whatever object to the sensor for tracking. You may switch objects in between.
    - Press any key to quit the streaming session.
Important Note:
    - If you are using the GelSight Mini, setting args.streamer to 'opencv' will result in a lower frame rate 
      (~10 Hz instead of the expected 25 Hz). Setting args.streamer to 'ffmpeg' resolves this frame rate issue;
      however, on some systems, it may introduce significant delays and duplicate frames.
    - If you are not using GelSight Mini, please calibrate your sensor using the gs_sdk package.
    - The whole calibration process only requires a metal ball with a known diameter and less than half hour.
    - Provide the trained calibration model and the sensor configuration file as arguments when running.

Usage:
    python realtime_object_tracking.py [--calib_model_path CALIB_MODEL_PATH] [--config_path CONFIG_PATH] [--device {cpu, cuda}]

Arguments:
    --calib_model_path: (Optional) The directory where the calibration model are stored.
            The calibration model can be easily trained using the gs_sdk package.
            The default is our trained calibration model for GelSight Mini.
    --config_path: (Optional) The path of the configuration file for the GelSight sensor.
            The configuration file specifies the specifications of the sensor.
            The default is the configuration file for GelSight Mini.
    --streamer: (Optional) The sensor streamer.
            Can be either 'opencv' for 'gs_device.Camera' or 'ffmpeg' for 'gs_device.FastCamera'.
    --device: (Optional) The device to run the neural network model that predicts the normal map.
            Can be either 'cpu' or 'cuda'. The default is 'cpu'.

Press any key to quit the streaming session.
"""


calib_model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")


def resize_show(image, frame_name="frame", scale=2.5):
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(frame_name, image)


def realtime_object_tracking():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Real-time tracking the object using tactile sensors."
    )
    parser.add_argument(
        "-b",
        "--calib_model_path",
        type=str,
        help="place where the calibration data and model is stored",
        default=calib_model_path,
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the configuration file for the GelSight sensor",
        default=config_path,
    )
    parser.add_argument(
        "-s",
        "--streamer",
        type=str,
        choices=["opencv", "ffmpeg"],
        help="The sensor streamer. 'opencv' for 'gs_device.Camera' and 'ffmpeg' for 'gs_device.FastCamera'.",
        default="opencv",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="the device to run the neural network model that predicts the normal map",
        default="cpu",
    )
    args = parser.parse_args()

    # Read the configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        ppmm = config["ppmm"]
        imgh = config["imgh"]
        imgw = config["imgw"]
        raw_imgh = config["raw_imgh"]
        raw_imgw = config["raw_imgw"]
        framerate = config["framerate"]

    # Connect to the sensor and the reconstructor
    if args.streamer == "opencv":
        device = Camera(device_name, imgh, imgw)
    elif args.streamer == "ffmpeg":
        device = FastCamera(device_name, imgh, imgw, raw_imgh, raw_imgw, framerate)
    device.connect()
    recon = Reconstructor(args.calib_model_path, device="cpu")

    # Collect background images
    print("Collecting 10 background images, please wait ...")
    bg_images = []
    for _ in range(10):
        image = device.get_image()
        bg_images.append(image)
    bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
    recon.load_bg(bg_image)
    print("Done with background collection.")

    # Real-time object tracking
    print("\nStart object tracking, Press any key to quit.\n")
    is_running = True
    while is_running:
        image = device.get_image()
        G, H, C = recon.get_surface_info(image, ppmm)
        frame = Frame(G, H, C)
        if not frame.is_contacted:
            resize_show(image)
            key = cv2.waitKey(1)
            if key != -1:
                is_running = False
            continue
        else:
            # Tracking a new object, wait 2 frames for the contact to stabilize
            for _ in range(2):
                image = device.get_image()
                resize_show(image)
                key = cv2.waitKey(1)

            # Get the surface information of the reference frame (key frame)
            image_start = device.get_image()
            G_start, H_start, C_start = recon.get_surface_info(image_start, ppmm)
            frame_start = Frame(G_start, H_start, C_start)
            # For display purpose, get the largest contour and its center
            contours_start, _ = cv2.findContours(
                (C_start * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            M_start = cv2.moments(max(contours_start, key=cv2.contourArea))
            cx_start, cy_start = int(M_start["m10"] / M_start["m00"]), int(
                M_start["m01"] / M_start["m00"]
            )

            # Start tracking this object relative to the reference frame (key frame)
            frame_ref = frame_start
            frame_prev = frame_start
            prev_T_ref = np.eye(4, dtype=np.float32)
            start_T_ref = np.eye(4, dtype=np.float32)
            is_tracking = True
            while is_tracking:
                # Get the surface information of the current frame
                image_curr = device.get_image()
                G_curr, H_curr, C_curr = recon.get_surface_info(image_curr, ppmm)
                frame_curr = Frame(G_curr, H_curr, C_curr)
                if not frame_curr.is_contacted:
                    is_tracking = False
                    break

                # Use NormalFlow to estimate the transformation
                try:
                    curr_T_ref = normalflow(
                        frame_ref.N,
                        frame_ref.C,
                        frame_ref.H,
                        frame_ref.L,
                        frame_curr.N,
                        frame_curr.C,
                        frame_curr.H,
                        frame_curr.L,
                        prev_T_ref,
                        ppmm,
                    )
                    frame_prev = frame_curr
                    prev_T_ref = curr_T_ref
                except LoseTrackError:
                    # Reset reference frame as the previous frame
                    frame_ref = frame_prev
                    start_T_ref = start_T_ref @ np.linalg.inv(prev_T_ref)
                    prev_T_ref = np.eye(4, dtype=np.float32)
                    # Use NormalFlow to estimate the transformation to the new reference frame
                    try:
                        # We disable the threshold for consecutive frame tracking
                        curr_T_ref = normalflow(
                            frame_ref.N,
                            frame_ref.C,
                            frame_ref.H,
                            frame_ref.L,
                            frame_curr.N,
                            frame_curr.C,
                            frame_curr.H,
                            frame_curr.L,
                            prev_T_ref,
                            ppmm,
                            scr_threshold=0.0,
                            ccs_threshold=0.0,
                        )
                        frame_prev = frame_curr
                        prev_T_ref = curr_T_ref
                    except LoseTrackError:
                        # Lose track, set current frame as new start frame
                        print("Lose Track!")
                        is_tracking = False
                        break

                # Display the object tracking result
                image_l = image_start.copy()
                cv2.putText(
                    image_l,
                    "Initial Frame",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                center_start = np.array([cx_start, cy_start]).astype(np.int32)
                unit_vectors_start = np.eye(3)[:, :2]
                annotate_coordinate_system(image_l, center_start, unit_vectors_start)
                # Annotate the transformation on the target frame
                image_r = image_curr.copy()
                cv2.putText(
                    image_r,
                    "Current Frame",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                center_3d_start = (
                    np.array(
                        [(cx_start - imgw / 2 + 0.5), (cy_start - imgh / 2 + 0.5), 0]
                    )
                    * ppmm
                    / 1000.0
                )
                unit_vectors_3d_start = np.eye(3) * ppmm / 1000.0
                curr_T_start = curr_T_ref @ np.linalg.inv(start_T_ref)
                remapped_center_3d_start = (
                    np.dot(curr_T_start[:3, :3], center_3d_start) + curr_T_start[:3, 3]
                )
                remapped_cx_start = (
                    remapped_center_3d_start[0] * 1000 / ppmm + imgw / 2 - 0.5
                )
                remapped_cy_start = (
                    remapped_center_3d_start[1] * 1000 / ppmm + imgh / 2 - 0.5
                )
                remapped_center_start = np.array(
                    [remapped_cx_start, remapped_cy_start]
                ).astype(np.int32)
                remapped_unit_vectors_start = (
                    np.dot(curr_T_start[:3, :3], unit_vectors_3d_start.T).T
                    * 1000
                    / ppmm
                )[:, :2]
                annotate_coordinate_system(
                    image_r, remapped_center_start, remapped_unit_vectors_start
                )

                # Display
                resize_show(cv2.hconcat([image_l, image_r]))
                key = cv2.waitKey(1)
                if key != -1:
                    is_tracking = False
                    is_running = False

    device.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_object_tracking()
