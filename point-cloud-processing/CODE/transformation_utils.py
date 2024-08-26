import numpy as np

def compute_z_offset(combined_mesh, glb_mesh):
    """
    Compute the Z offset needed to align the floor of the GLB mesh with the combined mesh.
    """
    combined_bbox = combined_mesh.get_axis_aligned_bounding_box()
    glb_bbox = glb_mesh.get_axis_aligned_bounding_box()
    
    lowest_z_combined = combined_bbox.min_bound[2]
    lowest_z_glb = glb_bbox.min_bound[2]

    z_offset = lowest_z_combined - lowest_z_glb
    
    return z_offset

def apply_z_offset(mesh, z_offset):
    """
    Apply the Z offset to the mesh.
    """
    mesh.translate((0, 0, z_offset))

def calculate_transformation_matrix(initial_transformation, angle, translation, center_translation, z_offset):
    """ Calculate the transformation matrix based on the given parameters. """
    cos_theta = np.cos(np.radians(angle))
    sin_theta = np.sin(np.radians(angle))
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # Create the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    # Create the initial transformation matrix
    initial_transformation_matrix = np.eye(4)
    initial_transformation_matrix[:3, :3] = initial_transformation

    # Combine the transformations
    combined_transformation = np.eye(4)
    combined_transformation[:3, :3] = rotation_matrix @ initial_transformation_matrix[:3, :3]
    combined_transformation[:3, 3] = translation_matrix[:3, 3] + center_translation
    
    # Add Z offset
    combined_transformation[2, 3] += z_offset
    
    return combined_transformation

def accumulate_transformations(transformations):
    """Accumulate a list of transformation matrices."""
    final_transformation = np.eye(4)
    for transform in transformations:
        final_transformation = transform @ final_transformation
    return final_transformation

def create_center_based_transformation_matrix(mesh, optimal_angle, optimal_tx, optimal_ty):
    """Create a 4x4 transformation matrix with center-based rotation and translation."""
    # Compute the center of the mesh
    center = mesh.get_center()
    
    # Convert angle to radians
    angle_rad = np.radians(optimal_angle)
    
    # Create a 3x3 rotation matrix for the Z-axis
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                  0,                 1]
    ])
    
    # Translation to move the center to the origin
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center
    
    # Translation to move the center back
    translation_back = np.eye(4)
    translation_back[:3, 3] = center

    # Optimal translation matrix
    optimal_translation_matrix = np.eye(4)
    optimal_translation_matrix[0, 3] = optimal_tx
    optimal_translation_matrix[1, 3] = optimal_ty

    # Create a 4x4 homogeneous rotation matrix
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_mat
    
    # Combine the transformations: Translate to origin -> Rotate -> Translate back -> Optimal translation
    combined_transformation_matrix = (
        optimal_translation_matrix @ translation_back @ rotation_matrix_4x4 @ translation_to_origin
    )
    
    return combined_transformation_matrix

