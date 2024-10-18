import numpy as np
import trimesh
from typing import List


def calculate_rotation_z(matrix: np.ndarray) -> float:
    """
    Calculate the rotation angle around the Z-axis from a 4x4 transformation matrix.

    Parameters:
    - matrix: A 4x4 numpy array representing the transformation matrix.

    Returns:
    - Rotation angle around the Z-axis in radians. The angle is normalized to the range [0, 2π].
    """
    # Validate input matrix
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 transformation matrix.")

    # Extract relevant elements for Z-axis rotation calculation
    r11 = matrix[0, 0]
    r21 = matrix[1, 0]

    # Compute rotation angle in radians
    theta_z = np.arctan2(r21, r11)

    # Normalize the angle to the range [0, 2π] radians
    if theta_z < 0:
        theta_z += 2 * np.pi

    return theta_z


def accumulate_transformations(transformations: List[np.ndarray]) -> np.ndarray:
    """
    Accumulate a list of transformation matrices.

    Parameters:
    - transformations: A list of 4x4 transformation matrices.

    Returns:
    - final_transformation: The accumulated transformation matrix.
    """
    final_transformation = np.eye(4)
    for transform in transformations:
        final_transformation = transform @ final_transformation

    return final_transformation
