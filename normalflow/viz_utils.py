import cv2
import numpy as np


def annotate_coordinate_system(image, center, unit_vectors, vector_scale=50, alpha=1.0):
    """
    Annotate the coordinate system on the image.

    :param image: np.ndarray (H, W, 3); the image to annotate.
    :param center: np.ndarray (2,); the center of the coordinate system in pixel
    :param unit_vectors: np.ndarray (3, 2); the unit vectors of the coordinate system in 2D.
        It is the three 3D unit vectors (ex, ey, ez) projected to the image plane.
    :param vector_scale: float; the scale to display the unit vectors.
    :param alpha: float; the transparency of the annotation.
    """
    overlay = image.copy()
    # Annotate the origin
    cv2.circle(
        overlay,
        (center[0], center[1]),
        radius=5,
        color=(0, 0, 0),
        thickness=-1,
    )
    # Annotate the three axes
    end_points = (center + unit_vectors * vector_scale).astype(np.int32)
    cv2.arrowedLine(
        overlay,
        (center[0], center[1]),
        (end_points[0, 0], end_points[0, 1]),
        color=(0, 0, 255),
        thickness=2,
        tipLength=0.15,
    )
    cv2.arrowedLine(
        overlay,
        (center[0], center[1]),
        (end_points[1, 0], end_points[1, 1]),
        color=(0, 255, 0),
        thickness=2,
        tipLength=0.15,
    )
    cv2.arrowedLine(
        overlay,
        (center[0], center[1]),
        (end_points[2, 0], end_points[2, 1]),
        color=(255, 0, 0),
        thickness=2,
        tipLength=0.15,
    )
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
