import numpy as np
import trimesh
from typing import List

def compute_z_offset(
    combined_mesh: trimesh.Trimesh, glb_mesh: trimesh.Trimesh
) -> float:
    """
    Compute the Z offset needed to align the floor of the GLB mesh with the combined mesh.

    Parameters:
    - combined_mesh: The target mesh (e.g., the mesh representing the terrain/building).
    - glb_mesh: The source mesh (e.g., the GLB mesh).

    Returns:
    - z_offset: The calculated Z offset needed to align the two meshes.
    """
    # Get the axis-aligned bounding boxes (AABB) of the two meshes
    combined_bbox = combined_mesh.bounding_box.bounds
    glb_bbox = glb_mesh.bounding_box.bounds

    # Extract the lowest Z values (the minimum Z coordinate) from the bounding boxes
    lowest_z_combined = combined_bbox[0, 2]  # Min Z value of combined mesh
    lowest_z_glb = glb_bbox[0, 2]  # Min Z value of GLB mesh

    # Calculate the Z offset required to align the two meshes
    z_offset = lowest_z_combined - lowest_z_glb

    return z_offset


def apply_z_offset(mesh: trimesh.Trimesh, z_offset: float) -> trimesh.Trimesh:
    """
    Apply a Z offset to the given mesh, translating it along the Z-axis.

    Parameters:
    - mesh: The mesh to which the Z offset should be applied (trimesh.Trimesh).
    - z_offset: The Z offset value to apply.

    Returns:
    - mesh: The transformed mesh after applying the Z offset.
    """
    # Create a translation matrix for the Z offset
    translation_matrix = np.eye(4)
    translation_matrix[2, 3] = z_offset  # Apply Z-axis translation

    # Apply the transformation to the mesh
    mesh.apply_transform(translation_matrix)

    return mesh


def calculate_rotation_z(matrix: np.ndarray) -> float:
    """
    Calculate the rotation around the Z-axis from a 4x4 transformation matrix.

    Parameters:
    - matrix: A 4x4 numpy array representing the transformation matrix.

    Returns:
    - Rotation angle around the Z-axis in degrees.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 transformation matrix.")

    # Extract the elements needed for calculating the rotation around the Z-axis
    r11 = matrix[0, 0]
    r21 = matrix[1, 0]

    # Calculate the rotation angle in radians
    theta_z = np.arctan2(r21, r11)

    # Normalize the angle to be within [0, 2*pi] radians
    theta_z = (theta_z + 2 * np.pi) % (2 * np.pi)

    # Convert the angle from radians to degrees
    theta_z_degrees = np.degrees(theta_z)

    # Normalize the angle to be within [0, 360] degrees
    theta_z_degrees = (theta_z_degrees + 360) % 360

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
