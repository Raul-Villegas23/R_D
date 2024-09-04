import numpy as np
import trimesh

def compute_z_offset(combined_mesh: trimesh.Trimesh, glb_mesh: trimesh.Trimesh) -> float:
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
