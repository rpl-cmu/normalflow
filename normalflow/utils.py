import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation as R


def height2pointcloud(H, ppmm):
    """
    Convert the height map to the pointcloud.

    :param H: np.ndarray (H, W); the height map (unit: pixel).
    :param ppmm: float; the pixel per mm.
    :return pointcloud: np.ndarray (N, 3); the pointcloud (unit: m).
    """
    xx, yy = np.meshgrid(np.arange(H.shape[1]), np.arange(H.shape[0]), indexing="xy")
    xx = xx - H.shape[1] / 2 + 0.5
    yy = yy - H.shape[0] / 2 + 0.5
    pointcloud = np.stack((xx, yy, H), axis=-1) * ppmm / 1000.0
    pointcloud = np.reshape(pointcloud, (-1, 3))
    return pointcloud


def get_J(N, C, masked_pointcloud, sample_mask, ppmm):
    """
    Implement the Jacobian Matrix calculation for NormalFlow.
    Please refer to the mathematical expression of the Jacobian matrix in Appendix I of the paper.

    :param N: np.ndarray (H, W, 3); the normal map.
    :param C: np.ndarray (H, W); the contact mask.
    :param masked_pointcloud: np.ndarray (N, 3); the masked pointcloud. (unit: m)
    :param sample_mask: np.ndarray (N,); the sample mask.
    :param ppmm: float; the pixel per mm.
    :return J: np.ndarray (3, 2, N); the Jacobian matrix.
    """
    # Calculate Jacobian matrix of Nref
    dNdx = cv2.Sobel(N, cv2.CV_32F, 1, 0, ksize=5, scale=2 ** (-7))
    dNdy = cv2.Sobel(N, cv2.CV_32F, 0, 1, ksize=5, scale=2 ** (-7))
    xx, yy = np.meshgrid(np.arange(N.shape[1]), np.arange(N.shape[0]), indexing="xy")
    xx = xx.reshape(-1)[C.reshape(-1)][sample_mask]
    yy = yy.reshape(-1)[C.reshape(-1)][sample_mask]
    Jxx = dNdx[:, :, 0][yy, xx] / ppmm * 1000.0
    Jyx = dNdx[:, :, 1][yy, xx] / ppmm * 1000.0
    Jzx = dNdx[:, :, 2][yy, xx] / ppmm * 1000.0
    Jxy = dNdy[:, :, 0][yy, xx] / ppmm * 1000.0
    Jyy = dNdy[:, :, 1][yy, xx] / ppmm * 1000.0
    Jzy = dNdy[:, :, 2][yy, xx] / ppmm * 1000.0
    JN = np.stack([Jxx, Jxy, Jyx, Jyy, Jzx, Jzy], axis=-1).T.reshape(3, 2, -1)
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
