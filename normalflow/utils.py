import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation as R

class Frame:
    """
    The frame data structure.
    """
    def __init__(self, G, H, C, contact_threshold=500):
        """
        Initialize the frame with the gradient map, height map, and contact mask.
        """
        self.G = G
        self.H = H
        self.C = erode_contact_mask(C)
        self.N = gxy2normal(self.G)
        self.L = gxy2laplacian(self.G, self.C)
        # Is the frame in contact
        self.is_contacted = np.sum(self.C) >= contact_threshold


def height2pointcloud(H, M, ppmm):
    """
    Convert the height map to the pointcloud.

    :param H: np.ndarray (H, W); the height map (unit: pixel).
    :param M: np.ndarray (H, W); the subsample mask.
    :param ppmm: float; the pixel per mm.
    :return pointcloud: np.ndarray (N, 3); the pointcloud (unit: m).
    """
    masked_indices = np.nonzero(M)
    xx = masked_indices[1] - H.shape[1] / 2 + 0.5
    yy = masked_indices[0] - H.shape[0] / 2 + 0.5
    pointcloud = np.vstack((xx, yy, H[masked_indices])).T * (ppmm / 1000.0)
    pointcloud = pointcloud.astype(np.float32)
    return pointcloud


def get_J(N, M, masked_pointcloud, ppmm):
    """
    Implement the Jacobian Matrix calculation for NormalFlow.
    Please refer to the mathematical expression of the Jacobian matrix in Appendix I of the paper.

    :param N: np.ndarray (H, W, 3); the normal map.
    :param M: np.ndarray (H, W); the subsample mask.
    :param masked_pointcloud: np.ndarray (N, 3); the masked pointcloud. (unit: m)
    :param ppmm: float; the pixel per mm.
    :return J: np.ndarray (3, 2, N); the Jacobian matrix.
    """
    # Calculate Jacobian matrix of Nref
    dNdx = cv2.Sobel(N, cv2.CV_32F, 1, 0, ksize=5, scale=2 ** (-7))
    dNdy = cv2.Sobel(N, cv2.CV_32F, 0, 1, ksize=5, scale=2 ** (-7))
    yy, xx = np.nonzero(M)
    Jxx = dNdx[:, :, 0][yy, xx] / ppmm * 1000.0
    Jyx = dNdx[:, :, 1][yy, xx] / ppmm * 1000.0
    Jzx = dNdx[:, :, 2][yy, xx] / ppmm * 1000.0
    Jxy = dNdy[:, :, 0][yy, xx] / ppmm * 1000.0
    Jyy = dNdy[:, :, 1][yy, xx] / ppmm * 1000.0
    Jzy = dNdy[:, :, 2][yy, xx] / ppmm * 1000.0
    JN = (
        np.stack([Jxx, Jxy, Jyx, Jyy, Jzx, Jzy], axis=-1)
        .T.reshape(3, 2, -1)
        .astype(np.float32)
    )
    # Calculate Jacobian of remapping
    Jx = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    Jy = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
    )
    Jz = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    Jw = np.concatenate(
        [
            (P @ Jx @ masked_pointcloud.T).reshape(2, 1, -1),
            (P @ Jy @ masked_pointcloud.T).reshape(2, 1, -1),
            (P @ Jz @ masked_pointcloud.T).reshape(2, 1, -1),
            np.tile(P[:, :-1, np.newaxis], (1, 1, masked_pointcloud.shape[0])),
        ],
        axis=1,
    )
    JNw = np.matmul(JN.transpose(2, 0, 1), Jw.transpose(2, 0, 1)).transpose(1, 2, 0)
    # Calculate Jacobian of rotation
    Jr = np.zeros((3, 5, masked_pointcloud.shape[0]), dtype=np.float32)
    Jr[0, 1] = N[yy, xx, 2]
    Jr[0, 2] = -N[yy, xx, 1]
    Jr[1, 0] = -N[yy, xx, 2]
    Jr[1, 2] = N[yy, xx, 0]
    Jr[2, 0] = N[yy, xx, 1]
    Jr[2, 1] = -N[yy, xx, 0]

    J = JNw - Jr
    return J


def gxy2normal(G):
    """
    Get the normal map from the gradient map.

    :param G: np.ndarray (H, W, 2); the gradient map.
    :return N: np.ndarray (H, W, 3); the normal map.
    """
    ones = np.ones_like(G[:, :, :1], dtype=np.float32)
    N = np.dstack([-G, ones])
    N = N / np.linalg.norm(N, axis=-1, keepdims=True)
    return N


def gxy2laplacian(G, C):
    """
    Convert the gradient map into the laplacian map.

    :param G: np.3darray (H, W, 2); the gradient map.
    :param C: np.2darray (H, W); the contact mask.
    :return L: np.2darray (H, W); the laplacian map.
    """
    L = np.gradient(G[:, :, 0], axis=1) + np.gradient(G[:, :, 1], axis=0)
    L = cv2.GaussianBlur(L, (5, 5), 0)
    L[np.logical_not(C)] = 0
    L = np.clip(L, -0.7, 0.7)
    return L


def erode_contact_mask(C):
    """
    Erode the contact mask to obtain a robust contact mask.

    :param C: np.ndarray (H, W); the contact mask.
    :return eroded_C: np.ndarray (H, W); the eroded contact mask.
    """
    erode_size = max(C.shape[0] // 48, 1)
    eroded_C = binary_erosion(C, structure=np.ones((erode_size, erode_size)))
    return eroded_C


def transform2pose(T):
    """
    Transform the transformation matrix to the 6D pose. Pose is in (mm, degrees) unit.

    :param T: np.ndarray (4, 4); the transformation matrix.
    :return pose: np.ndarray (6,); the 6D pose.
    """
    zxy = np.degrees(R.from_matrix(T[:3, :3]).as_euler("zxy"))
    pose = np.array(
        [
            T[0, 3] * 1000.0,
            T[1, 3] * 1000.0,
            T[2, 3] * 1000.0,
            zxy[1],
            zxy[2],
            zxy[0],
        ]
    )
    return pose


def wide_remap(f, xx, yy, mode=cv2.INTER_LINEAR):
    """
    cv2.remap only deals with length less than 32767. This function deals with longer length.
    """
    remapped = []
    for i in range(0, len(xx), 32000):
        j = min(len(xx), i + 32000)
        remapped.append(cv2.remap(f, xx[i:j], yy[i:j], mode))
    return np.concatenate(remapped, axis=0)


def get_backproj_laplacian(L_tar, C_tar, masked_pointcloud_ref, tar_T_ref, ppmm=0.0634):
    """
    Given the laplacian and contact map of the target frame, return the backprojected laplacian
    map to the reference frame.

    :param L_tar: np.ndarray (H, W); the laplacian map of the target frame.
    :param C_tar: np.ndarray (H, W); the contact map of the target frame.
    :param masked_pointcloud_ref: np.ndarray (N, 3); the pointcloud of the reference frame.
    :param tar_T_ref: np.ndarray (4, 4); the homogeneous transformation matrix from the reference frame to the target frame.
    :param ppmm: float; pixel per millimeter.
    :return:
        masked_L_tar_backproj: np.ndarray (N,); the backprojected laplacian map.
        masked_C_tar_backproj: np.ndarray (N,); the backprojected contact mask.
    """
    # Get the remapped pixels
    remapped_masked_pointcloud_ref = (
        np.dot(tar_T_ref[:3, :3], masked_pointcloud_ref.T).T + tar_T_ref[:3, 3]
    )
    remapped_masked_xx_ref = (
        remapped_masked_pointcloud_ref[:, 0] * 1000.0 / ppmm + C_tar.shape[1] / 2 - 0.5
    )
    remapped_masked_yy_ref = (
        remapped_masked_pointcloud_ref[:, 1] * 1000.0 / ppmm + C_tar.shape[0] / 2 - 0.5
    )
    # Get the backprojected laplacians and contact mask
    masked_L_tar_backproj = wide_remap(
        L_tar, remapped_masked_xx_ref, remapped_masked_yy_ref
    )[:, 0]
    masked_C_tar_backproj = (
        wide_remap(
            C_tar.astype(np.float32), remapped_masked_xx_ref, remapped_masked_yy_ref
        )[:, 0]
        > 0.5
    )
    xx_region = np.logical_and(
        remapped_masked_xx_ref >= 0, remapped_masked_xx_ref < C_tar.shape[1]
    )
    yy_region = np.logical_and(
        remapped_masked_yy_ref >= 0, remapped_masked_yy_ref < C_tar.shape[0]
    )
    xy_region = np.logical_and(xx_region, yy_region)
    masked_C_tar_backproj = np.logical_and(masked_C_tar_backproj, xy_region)
    return masked_L_tar_backproj, masked_C_tar_backproj
