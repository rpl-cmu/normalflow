import cv2
import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation as R

from normalflow.utils import (
    height2pointcloud,
    get_J,
    get_backproj_laplacian,
    wide_remap,
)


class LoseTrackError(Exception):
    """Exception raised when NormalFlow failed to estimate the sensor pose."""

    def __init__(
        self,
        message="NormalFlow registration failed.",
    ):
        super().__init__(message)


def normalflow(
    N_ref,
    C_ref,
    H_ref,
    L_ref,
    N_tar,
    C_tar,
    H_tar,
    L_tar,
    tar_T_ref_init=np.eye(4),
    ppmm=0.0634,
    n_samples=3000,
    scr_threshold=0.3,
    ccs_threshold=0.85,
    return_quality=False,
    verbose=False,
):
    """
    The NormalFlow algorithm to estimate the homogeneous transformation of the sensor between two frames.
    Given the normal map, contact map, height map, and laplacian map of two frames, return the sensor transformation.

    :param N_ref: np.ndarray (H, W, 3); the normal map of the reference frame.
    :param C_ref: np.ndarray (H, W); the contact map of the reference frame.
    :param H_ref: np.ndarray (H, W); the height map of the reference frame. (unit: pixel)
    :param L_ref: np.ndarray (H, W); the laplacian map of the reference frame.
    :param N_tar: np.ndarray (H, W, 3); the normal map of the target frame.
    :param C_tar: np.ndarray (H, W); the contact map of the target frame.
    :param H_tar: np.ndarray (H, W); the height map of the target frame. (unit: pixel)
    :param L_tar: np.ndarray (H, W); the laplacian map of the target frame.
    :param tar_T_ref_init: np.2darray (4, 4); the initial guess homogeneous transformation matrix.
    :param ppmm: float; pixel per millimeter.
    :param n_samples: int; the number of samples to use for the optimization. If None, use all the pixels in contact.
    :param scr_threshold: float; the threshold of the shared curvature ratio (the ratio of the shared laplacian).
    :param ccs_threshold: float; the threshold of the curvature cosine similarity (cosine distance between the laplacians).
    :param return_qaulity: bool; whether to retunr the registration quality.
    :param verbose: bool; whether to print the information of the algorithm
    :return:
        np.ndarray (4, 4); the homogeneous transformation matrix from frame t to frame t+1.
        (optional) dict:
            scr: float; the shared curvature ratio.
            ccs: float; the curvature cosine similarity.
    :raises: LoseTrackError; NormalFlow registration failed
    """
    # Pick the pixels with the most curvature as the mask M
    L_threshold = np.partition(np.abs(L_ref).flatten(), -n_samples)[-n_samples] + 1e-6
    M_ref = np.abs(L_ref) >= L_threshold
    # Apply mask to pointcloud and normals on the reference
    masked_pointcloud_ref = height2pointcloud(H_ref, M_ref, ppmm)
    masked_N_ref = N_ref[M_ref]
    masked_C_ref = C_ref[M_ref]
    masked_L_ref = L_ref[M_ref]
    J = get_J(N_ref, M_ref, masked_pointcloud_ref, ppmm)

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
            wide_remap(
                C_tar.astype(np.float32),
                remapped_xx_ref,
                remapped_yy_ref,
                mode=cv2.INTER_NEAREST,
            )[:, 0]
            > 0.5
        )
        xx_region = np.logical_and(
            remapped_xx_ref >= 0, remapped_xx_ref < C_ref.shape[1]
        )
        yy_region = np.logical_and(
            remapped_yy_ref >= 0, remapped_yy_ref < C_ref.shape[0]
        )
        xy_region = np.logical_and(xx_region, yy_region)
        shared_C = np.logical_and(remapped_C_tar, xy_region)
        if np.sum(shared_C) < 10:
            raise LoseTrackError()

        # Least square estimation
        remapped_N_tar = wide_remap(N_tar, remapped_xx_ref, remapped_yy_ref)[:, 0, :]
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
        wide_remap(
            C_tar.astype(np.float32),
            remapped_xx_ref,
            remapped_yy_ref,
            mode=cv2.INTER_NEAREST,
        )[:, 0]
        > 0.5
    )
    xx_region = np.logical_and(remapped_xx_ref >= 0, remapped_xx_ref < C_ref.shape[1])
    yy_region = np.logical_and(remapped_yy_ref >= 0, remapped_yy_ref < C_ref.shape[0])
    xy_region = np.logical_and(xx_region, yy_region)
    remapped_C_tar = np.logical_and(remapped_C_tar, xy_region)
    remapped_H_tar = wide_remap(H_tar, remapped_xx_ref, remapped_yy_ref)[:, 0]
    z_diff = (
        remapped_H_tar[remapped_C_tar] * ppmm / 1000.0
        - remapped_pointcloud_ref[:, 2][remapped_C_tar]
    )
    tar_T_ref[2, 3] = np.mean(z_diff, axis=0)
    # Examine the tracking result with enough overlap and curvature matches well
    masked_L_tar_backproj, masked_C_tar_backproj = get_backproj_laplacian(
        L_tar, C_tar, masked_pointcloud_ref, tar_T_ref, ppmm
    )
    masked_C_shared = np.logical_and(masked_C_ref, masked_C_tar_backproj)
    scr = np.sum(np.abs(masked_L_ref[masked_C_shared])) / np.sum(
        np.abs(masked_L_ref)
    )
    ccs = (
        masked_L_ref[masked_C_shared]
        @ masked_L_tar_backproj[masked_C_shared]
        / np.linalg.norm(masked_L_ref[masked_C_shared])
        / np.linalg.norm(masked_L_tar_backproj[masked_C_shared])
    )
    if scr < scr_threshold or ccs < ccs_threshold:
        raise LoseTrackError()
    # Return the estimated transformation and the registration quality if requested
    if return_quality:
        registration_quality = {}
        registration_quality["scr"] = scr
        registration_quality["ccs"] = ccs
        return tar_T_ref, registration_quality
    else:
        return tar_T_ref
