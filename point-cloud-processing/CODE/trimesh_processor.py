import numpy as np
import trimesh
from typing import Optional, Tuple, Dict, Any
import logging


def create_trimesh_from_feature(
    feature: Dict[str, Any]
) -> Tuple[Optional[trimesh.Trimesh], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a Trimesh object using only the highest LoD from feature data.

    Parameters:
    - feature: A dictionary containing the feature data with vertices and geometry.

    Returns:
    - mesh: The created Trimesh object.
    - scale: The scale applied to the vertices.
    - translate: The translation applied to the vertices.
    """

    if 'vertices' in feature['feature']:
        vertices = np.array(feature['feature']['vertices'])
        transform = feature['metadata'].get('transform', {})
        scale = np.array(transform.get('scale', [1, 1, 1]))
        translate = np.array(transform.get('translate', [0, 0, 0]))
        vertices = vertices * scale + translate

        city_objects = feature['feature'].get('CityObjects', {})
        max_lod = None
        max_lod_geom = None

        for obj in city_objects.values():
            for geom in obj.get('geometry', []):
                lod = geom.get('lod', None)
                if max_lod is None or (lod is not None and float(lod) > float(max_lod)):
                    max_lod = lod
                    max_lod_geom = geom

        if max_lod_geom:
            print(f"Using highest LoD: {max_lod}")
            faces = []

            for boundary_group in max_lod_geom.get('boundaries', []):
                for boundary in boundary_group:
                    if isinstance(boundary[0], list):
                        for sub_boundary in boundary:
                            if len(sub_boundary) >= 3:
                                for i in range(1, len(sub_boundary) - 1):
                                    faces.append([sub_boundary[0], sub_boundary[i], sub_boundary[i + 1]])
                    else:
                        if len(boundary) >= 3:
                            for i in range(1, len(boundary) - 1):
                                faces.append([boundary[0], boundary[i], boundary[i + 1]])

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh, scale, translate
        else:
            logging.error("No geometry data found for the highest LoD.")
            return None, None, None
    else:
        logging.error("No vertices found in the feature data.")
        return None, None, None
    
def find_closest_vertex(vertices: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find the closest vertex to the origin in a set of vertices.

    Parameters:
    - vertices: A numpy array of vertices.

    Returns:
    - closest_vertex: The vertex closest to the origin.
    - index: The index of the closest vertex.
    """
    distances = np.linalg.norm(vertices, axis=1) # Calculate the Euclidean distance to the origin (0, 0, 0)
    index = np.argmin(distances) # Find the index of the closest vertex based on distance
    closest_vertex = vertices[index] # Get the closest vertex
    return closest_vertex, index

def load_and_transform_glb_model_trimesh(
    file_path: str, 
    translate: np.ndarray, 
    enable_post_processing: bool = False
) -> Tuple[Optional[trimesh.Trimesh], Optional[np.ndarray]]:
    """
    Load and transform a GLB model using trimesh.

    Parameters:
    - file_path: The file path to the GLB model.
    - translate: The translation vector to apply to the model.
    - enable_post_processing: Whether to enable post-processing (e.g., centering the mesh).

    Returns:
    - mesh: The transformed GLB model mesh (trimesh.Trimesh).
    - combined_transformation: The transformation matrix applied to the mesh.
    """

    # Load the mesh or scene using trimesh
    scene_or_mesh = trimesh.load_mesh(file_path)
    selected_meshes = []

    # Check if the loaded object is a scene (contains multiple meshes) or a single mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Prioritize meshes if they contain "opaque" in the name
        for name, geometry in scene_or_mesh.geometry.items():
            # Check for "opaque" in the name
            if "opaque" in name.lower():
                selected_meshes.append(geometry)

        # If no opaque meshes were found, but other geometries are available
        if not selected_meshes:
            logging.warning("No opaque meshes found, will include other geometries.")
            selected_meshes = list(scene_or_mesh.geometry.values())

        if selected_meshes:
            mesh = trimesh.util.concatenate(selected_meshes)
        else:
            logging.error("No suitable meshes found in the GLB model.")
            raise ValueError(
                "No suitable meshes found in the GLB model."
            )  # Raise an exception

    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        logging.error("Loaded object is neither a Trimesh.Scene nor a Trimesh.")
        raise ValueError("Loaded object is neither a Trimesh.Scene nor a Trimesh.")

    # Ensure the mesh has vertices and faces
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        logging.error("The GLB model has no vertices or faces.")
        raise ValueError(
            "The GLB model has no vertices or faces."
        )  # Raise an exception

    # This transformation matrix is used to convert the GLB model from a left-handed coordinate system to a right-handed one
    transformation_matrix = np.array(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )

    # Define a translation matrix to move the model to the 3d bag origin location
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translate

    # Combine transformations (translation + rotation)
    combined_transformation = translation_matrix @ transformation_matrix

    # Apply the transformation to the mesh
    mesh.apply_transform(combined_transformation)

    return mesh, combined_transformation

def align_trimesh_centers(
    mesh1: trimesh.Trimesh, 
    mesh2: trimesh.Trimesh
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Align the center of mesh2 to mesh1 using trimesh.

    Parameters:
    - mesh1: The reference mesh to align to.
    - mesh2: The mesh to be aligned.

    Returns:
    - mesh2: The aligned mesh.
    - transformation_matrix: The 4x4 transformation matrix used for alignment.
    """
    
    center1 = mesh1.centroid
    center2 = mesh2.centroid
    translation = center1 - center2
    mesh2.apply_translation(translation)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = translation
    return mesh2, transformation_matrix

def apply_optimal_params_trimesh(
    mesh: trimesh.Trimesh, 
    optimal_angle: float, 
    optimal_tx: float, 
    optimal_ty: float
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Apply the optimal rotation and translation to the mesh, based on the mesh center.

    Parameters:
    - mesh: The Trimesh object to be transformed.
    - optimal_angle: The optimal rotation angle around the Z-axis (in degrees).
    - optimal_tx: The optimal translation in the X direction.
    - optimal_ty: The optimal translation in the Y direction.

    Returns:
    - mesh: The transformed mesh with the applied rotation and translation.
    - transformation_matrix: The 4x4 transformation matrix applied to the mesh.
    """

    # Convert angle from degrees to radians
    angle_rad = np.radians(optimal_angle)

    # Create a 4x4 identity transformation matrix
    transformation_matrix = np.eye(4)

    # Rotation matrix around the Z-axis
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                  0,                 1]
    ])

    # Translate to the origin, rotate, and translate back
    mesh_center = mesh.centroid

    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -mesh_center

    translation_back = np.eye(4)
    translation_back[:3, 3] = mesh_center

    # Create a 4x4 rotation matrix from the 3x3 rotation matrix
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix

    # Combine the transformations: T_back * R * T_origin
    combined_transform = translation_back @ rotation_matrix_4x4 @ translation_to_origin

    # Apply translation in the X and Y directions
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = [optimal_tx, optimal_ty, 0]

    # Combine the rotation and translation transformations
    transformation_matrix = translation_matrix @ combined_transform

    # Apply the final transformation to the mesh
    mesh.apply_transform(transformation_matrix)

    return mesh, transformation_matrix