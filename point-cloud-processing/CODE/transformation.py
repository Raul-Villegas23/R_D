import numpy as np
from scipy.optimize import minimize
from geometry_utils import calculate_intersection_error

def optimize_rotation_and_translation(perimeter1, perimeter2):
    """Optimize rotation angle and translation to align two perimeters."""
    initial_guesses = [[-45.0, 0.0, 0.0], [45.0, 0.0, 0.0], [90.0, 0.0, 0.0], [-90.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    bounds = [(-180, 180), (-np.inf, np.inf), (-np.inf, np.inf)]
    best_result, lowest_error = None, float('inf')

    for initial_guess in initial_guesses:
        result = minimize(calculate_intersection_error, initial_guess, args=(perimeter1, perimeter2), method='L-BFGS-B', bounds=bounds)
        if result.success and result.fun < lowest_error:
            best_result, lowest_error = result, result.fun

    return best_result.x if best_result else None

def compute_z_offset(combined_mesh, glb_mesh):
    """
    Compute the Z offset needed to align the floor of the GLB mesh with the combined mesh.
    """
    combined_bbox = combined_mesh.get_axis_aligned_bounding_box()
    glb_bbox = glb_mesh.get_axis_aligned_bounding_box()
    
    lowest_z_combined = combined_bbox.min_bound[2]
    lowest_z_glb = glb_bbox.min_bound[2]
    
    print(f"Lowest Z in Combined Mesh Bounding Box: {lowest_z_combined}")
    print(f"Lowest Z in GLB Mesh Bounding Box: {lowest_z_glb}")
    
    z_offset = lowest_z_combined - lowest_z_glb
    
    return z_offset

def apply_z_offset(mesh, z_offset):
    """
    Apply the Z offset to the mesh.
    """
    mesh.translate((0, 0, z_offset))

def calculate_transformation_matrix(initial_transformation, angle, translation, center_translation, z_offset):
    """
    Calculate the transformation matrix for initial transformation, rotation angle, translation, centering, and Z offset.
    
    :param initial_transformation: The initial transformation matrix (3x3) to apply.
    :param angle: The rotation angle in degrees.
    :param translation: The translation vector (x, y, z).
    :param center_translation: Additional center translation (x, y, z).
    :param z_offset: The Z offset to apply.
    :return: The final transformation matrix (4x4).
    """
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
