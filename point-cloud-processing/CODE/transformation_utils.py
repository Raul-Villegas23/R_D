import numpy as np
import open3d as o3d
from typing import List, Union

def compute_z_offset(combined_mesh: o3d.geometry.TriangleMesh, glb_mesh: o3d.geometry.TriangleMesh) -> float:
    """
    Compute the Z offset needed to align the floor of the GLB mesh with the combined mesh.

    Parameters:
    - combined_mesh: The combined reference mesh.
    - glb_mesh: The GLB mesh to align.

    Returns:
    - z_offset: The calculated Z offset.
    """
    combined_bbox = combined_mesh.get_axis_aligned_bounding_box()
    glb_bbox = glb_mesh.get_axis_aligned_bounding_box()
    
    lowest_z_combined = combined_bbox.min_bound[2]
    lowest_z_glb = glb_bbox.min_bound[2]

    z_offset = lowest_z_combined - lowest_z_glb
    
    return z_offset

def apply_z_offset(mesh: o3d.geometry.TriangleMesh, z_offset: float) -> None:
    """
    Apply the Z offset to the mesh.

    Parameters:
    - mesh: The mesh to which the Z offset will be applied.
    - z_offset: The Z offset value.
    """
    mesh.translate((0, 0, z_offset))

def calculate_transformation_matrix(
    initial_transformation: Union[np.ndarray, List[List[float]]], 
    angle: float, 
    translation: Union[np.ndarray, List[float]], 
    center_translation: Union[np.ndarray, List[float]], 
    z_offset: float
) -> np.ndarray:
    """
    Calculate the transformation matrix based on the given parameters.

    Parameters:
    - initial_transformation: The initial 3x3 transformation matrix.
    - angle: The rotation angle in degrees.
    - translation: The translation vector [tx, ty, tz].
    - center_translation: The translation vector to account for center alignment.
    - z_offset: The Z offset value.

    Returns:
    - combined_transformation: The calculated 4x4 transformation matrix.
    """
    cos_theta = np.cos(np.radians(angle))
    sin_theta = np.sin(np.radians(angle))
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    initial_transformation_matrix = np.eye(4)
    initial_transformation_matrix[:3, :3] = initial_transformation

    combined_transformation = np.eye(4)
    combined_transformation[:3, :3] = rotation_matrix @ initial_transformation_matrix[:3, :3]
    combined_transformation[:3, 3] = translation_matrix[:3, 3] + center_translation
    
    combined_transformation[2, 3] += z_offset
    
    return combined_transformation

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

def create_center_based_transformation_matrix(
    mesh: o3d.geometry.TriangleMesh, 
    optimal_angle: float, 
    optimal_tx: float, 
    optimal_ty: float
) -> np.ndarray:
    """
    Create a 4x4 transformation matrix with center-based rotation and translation.

    Parameters:
    - mesh: The mesh to be transformed.
    - optimal_angle: The optimal rotation angle around the Z-axis.
    - optimal_tx: The optimal translation in the X direction.
    - optimal_ty: The optimal translation in the Y direction.

    Returns:
    - combined_transformation_matrix: The calculated 4x4 transformation matrix.
    """
    center = mesh.get_center()
    
    angle_rad = np.radians(optimal_angle)
    
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                  0,                 1]
    ])
    
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center
    
    translation_back = np.eye(4)
    translation_back[:3, 3] = center

    optimal_translation_matrix = np.eye(4)
    optimal_translation_matrix[0, 3] = optimal_tx
    optimal_translation_matrix[1, 3] = optimal_ty

    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_mat
    
    combined_transformation_matrix = (
        optimal_translation_matrix @ translation_back @ rotation_matrix_4x4 @ translation_to_origin
    )
    
    return combined_transformation_matrix
