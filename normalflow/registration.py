import cv2
import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation as R

from normalflow.utils import height2pointcloud, get_J


def normalflow(
    N_ref,
    C_ref,
    H_ref,
    N_tar,
    C_tar,
    H_tar,
    tar_T_ref_init=np.eye(4),
    ppmm=0.0634,
    n_samples=5000,
    verbose=False,
):
    """
    The NormalFlow algorithm to estimate the homogeneous transformation of the sensor between two frames.
    Given the normal map, contact map, and height map of two frames, return the sensor transformation.

    :param N_ref: np.ndarray (H, W, 3); the normal map of the reference frame.
    :param C_ref: np.ndarray (H, W); the contact map of the reference frame.
    :param H_ref: np.ndarray (H, W); the height map of the reference frame. (unit: pixel)
    :param N_tar: np.ndarray (H, W, 3); the normal map of the target frame.
    :param C_tar: np.ndarray (H, W); the contact map of the target frame.
    :param H_tar: np.ndarray (H, W); the height map of the target frame. (unit: pixel)
    :param tar_T_ref_init: np.2darray (4, 4); the initial guess homogeneous transformation matrix.
    :param ppmm: float; pixel per millimeter.
    :param n_samples: int; the number of samples to use for the optimization. If None, use all the pixels in contact.
    :param verbose: bool; whether to print the information of the algorithm
    :return: np.ndarray (4, 4); the homogeneous transformation matrix from frame t to frame t+1.
    """
    tar_T_ref_init = tar_T_ref_init.astype(np.float32)
    # Apply mask to pointcloud and normals on the reference
    pointcloud_ref = height2pointcloud(H_ref, ppmm).astype(np.float32)
    masked_pointcloud_ref = pointcloud_ref[C_ref.reshape(-1)]
    masked_N_ref = N_ref.reshape(-1, 3)[C_ref.reshape(-1)]
    # Randomly sample the points to speed up
    if n_samples is not None and n_samples < masked_N_ref.shape[0]:
        sample_mask = np.random.choice(masked_N_ref.shape[0], n_samples, replace=False)
    else:
        sample_mask = np.arange(masked_N_ref.shape[0])
    masked_pointcloud_ref = masked_pointcloud_ref[sample_mask]
    masked_N_ref = masked_N_ref[sample_mask]
    J = get_J(N_ref, C_ref, masked_pointcloud_ref, sample_mask, ppmm)

    # Apply Gauss-Newton optimization
    tar_T_ref = tar_T_ref_init.copy()
    max_iters = 50
    for i in range(max_iters):
        # Remap the pointcloud
        remapped_pointcloud_ref = (
            np.dot(tar_T_ref[:3, :3], masked_pointcloud_ref.T).T + tar_T_ref[:3, 3]
        )
        remapped_xx_ref = (
            remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + N_ref.shape[1] / 2 - 0.5
        )
        remapped_yy_ref = (
            remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + N_ref.shape[0] / 2 - 0.5
        )
        # Get the shared contact map
        remapped_C_tar = (
            cv2.remap(
                C_tar.astype(np.float32),
                remapped_xx_ref,
                remapped_yy_ref,
                cv2.INTER_LINEAR,
            )[:, 0]
            > 0.5
        )
        xx_region = np.logical_and(remapped_xx_ref >= 0, remapped_xx_ref < C_ref.shape[1])
        yy_region = np.logical_and(remapped_yy_ref >= 0, remapped_yy_ref < C_ref.shape[0])
        xy_region = np.logical_and(xx_region, yy_region)
        shared_C = np.logical_and(remapped_C_tar, xy_region)
        if np.sum(shared_C) < 10:
            if verbose:
                print("lose track")
            return tar_T_ref_init

        # Least square estimation
        remapped_N_tar = cv2.remap(N_tar, remapped_xx_ref, remapped_yy_ref, cv2.INTER_LINEAR)[
            :, 0, :
        ]
        b = (remapped_N_tar @ np.linalg.inv(tar_T_ref[:3, :3]).T - masked_N_ref)[
            shared_C
        ].reshape(-1)
        A = np.transpose(J, (2, 0, 1))[shared_C].reshape(-1, 5)
        dp = lstsq(A, b, lapack_driver="gelsy")[0]

        # Update matrix T by transformation composition
        dR = R.from_euler("xyz", dp[:3], degrees=False).as_matrix()
        dT = np.identity(4, dtype=np.float32)
        dT[:3, :3] = dR
        dT[:2, 3] = dp[3:]
        tar_T_ref = np.dot(tar_T_ref, np.linalg.inv(dT))
        tar_T_ref[2, 3] = 0.0

        # Convergence check or reaching maximum iterations
        if np.linalg.norm(dp[:3]) < 1e-4 and np.linalg.norm(dp[3:]) < 1e-5 and i > 5:
            if verbose:
                print("Total number of iterations: %i" % i)
            break
        if i == max_iters - 1:
            if verbose:
                print("Total number of iterations: %i" % i)

    # Calculate z translation by height difference
    remapped_pointcloud_ref = (
        np.dot(tar_T_ref[:3, :3], masked_pointcloud_ref.T).T + tar_T_ref[:3, 3]
    )
    remapped_xx_ref = (
        remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + N_ref.shape[1] / 2 - 0.5
    )
    remapped_yy_ref = (
        remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + N_ref.shape[0] / 2 - 0.5
    )
    remapped_C_tar = (
        cv2.remap(
            C_tar.astype(np.float32), remapped_xx_ref, remapped_yy_ref, cv2.INTER_LINEAR
        )[:, 0]
        > 0.5
    )
    xx_region = np.logical_and(remapped_xx_ref >= 0, remapped_xx_ref < C_ref.shape[1])
    yy_region = np.logical_and(remapped_yy_ref >= 0, remapped_yy_ref < C_ref.shape[0])
    xy_region = np.logical_and(xx_region, yy_region)
    remapped_C_tar = np.logical_and(remapped_C_tar, xy_region)
    remapped_H_tar = cv2.remap(H_tar, remapped_xx_ref, remapped_yy_ref, cv2.INTER_LINEAR)[
        :, 0
    ]
    tar_T_ref[2, 3] = np.mean(
        remapped_H_tar[remapped_C_tar] * ppmm / 1000.0
        - remapped_pointcloud_ref[:, 2][remapped_C_tar],
        axis=0,
    )
    return tar_T_ref