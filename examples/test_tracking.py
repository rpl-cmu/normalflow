import argparse
import os

import cv2
import numpy as np
import yaml

from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import normalflow
from normalflow.utils import erode_contact_mask, gxy2normal
from normalflow.viz_utils import annotate_coordinate_system

"""
This script demonstrates tracking objects using NormalFlow by tracking the object in a tactile video.

It loads the tactile video, extract the normal maps using the calibration model, and track the object using NormalFlow. 
The tracked object is then saved back as a tracked video in the data/ directory.

Usage:
    python test_tracking.py --device {cuda, cpu}

Arguments:
    --device: The device to load the neural network model. Options are 'cuda' or 'cpu'.
"""

model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
data_dir = os.path.join(os.path.dirname(__file__), "data")


def test_tracking():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Track the object in the tactile video."
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="The device to load and run the neural network model.",
    )
    args = parser.parse_args()

    # Load the device configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]
        imgh = config["imgh"]
        imgw = config["imgw"]

    # Create reconstructor
    recon = Reconstructor(model_path, device=args.device)
    bg_image = cv2.imread(os.path.join(data_dir, "background.png"))
    recon.load_bg(bg_image)

    # Load the frames from the video
    video_path = os.path.join(data_dir, "tactile_video.avi")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    tactile_images = []
    while True:
        ret, tactile_image = video.read()
        if not ret:
            break
        tactile_images.append(tactile_image)
    video.release()

    # Get the reference frame surface informations
    G_ref, H_ref, C_ref = recon.get_surface_info(tactile_images[0], ppmm)
    C_ref = erode_contact_mask(C_ref)
    N_ref = gxy2normal(G_ref)
    # For display purpose, get the largest contour and its center
    contours_ref, _ = cv2.findContours(
        (C_ref * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    M_ref = cv2.moments(max(contours_ref, key=cv2.contourArea))
    cx_ref, cy_ref = int(M_ref["m10"] / M_ref["m00"]), int(M_ref["m01"] / M_ref["m00"])

    # Track the target frames with regards to the reference frame
    tracked_tactile_images = []
    curr_T_ref_init = np.eye(4)
    for tactile_image in tactile_images[1:]:
        G_curr, H_curr, C_curr = recon.get_surface_info(tactile_image, ppmm)
        C_curr = erode_contact_mask(C_curr)
        N_curr = gxy2normal(G_curr)
        curr_T_ref = normalflow(
            N_ref,
            C_ref,
            H_ref,
            N_curr,
            C_curr,
            H_curr,
            curr_T_ref_init,
            ppmm,
        )
        curr_T_ref_init = curr_T_ref

        # Compose the tracked frames
        image_l = tactile_images[0].copy()
        cv2.putText(
            image_l,
            "Initial Frame",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        center_ref = np.array([cx_ref, cy_ref]).astype(np.int32)
        unit_vectors_ref = np.eye(3)[:, :2]
        annotate_coordinate_system(image_l, center_ref, unit_vectors_ref)
        # Annotate the transformation on the target frame
        image_r = tactile_image.copy()
        cv2.putText(
            image_r,
            "Current Frame",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        center_3d_ref = (
            np.array([(cx_ref - imgw / 2 + 0.5), (cy_ref - imgh / 2 + 0.5), 0])
            * ppmm
            / 1000.0
        )
        unit_vectors_3d_ref = np.eye(3) * ppmm / 1000.0
        remapped_center_3d_ref = (
            np.dot(curr_T_ref[:3, :3], center_3d_ref) + curr_T_ref[:3, 3]
        )
        remapped_cx_ref = remapped_center_3d_ref[0] * 1000 / ppmm + imgw / 2 - 0.5
        remapped_cy_ref = remapped_center_3d_ref[1] * 1000 / ppmm + imgh / 2 - 0.5
        remapped_center_ref = np.array([remapped_cx_ref, remapped_cy_ref]).astype(
            np.int32
        )
        remapped_unit_vectors_ref = (
            np.dot(curr_T_ref[:3, :3], unit_vectors_3d_ref.T).T * 1000 / ppmm
        )[:, :2]
        annotate_coordinate_system(
            image_r, remapped_center_ref, remapped_unit_vectors_ref
        )
        tracked_tactile_images.append(cv2.hconcat([image_l, image_r]))

    # Save the tracked frames into a tracking video
    save_path = os.path.join(data_dir, "tracked_tactile_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    video = cv2.VideoWriter(
        save_path,
        fourcc,
        fps,
        (tracked_tactile_images[0].shape[1], tracked_tactile_images[0].shape[0]),
    )
    for tracked_tactile_image in tracked_tactile_images:
        video.write(tracked_tactile_image)
    video.release()


if __name__ == "__main__":
    test_tracking()
