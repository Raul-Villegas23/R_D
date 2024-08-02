import numpy as np
import open3d as o3d
import logging

def create_mesh_from_feature(feature):
    """Create a mesh object from feature data."""
    if 'vertices' in feature['feature']:
        vertices = np.array(feature['feature']['vertices'])
        transform = feature['metadata'].get('transform', {})
        scale = np.array(transform.get('scale', [1, 1, 1]))
        translate = np.array(transform.get('translate', [0, 0, 0]))
        vertices = vertices * scale + translate

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        city_objects = feature['feature'].get('CityObjects', {})
        for obj in city_objects.values():
            for geom in obj.get('geometry', []):
                for boundary in geom.get('boundaries', []):
                    for i in range(1, len(boundary[0]) - 1):
                        mesh.triangles.append([boundary[0][0], boundary[0][i], boundary[0][i + 1]])

        mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
        return mesh, scale, translate
    else:
        logging.error("No vertices found in the feature data.")
        return None, None, None

def load_and_transform_glb_model(file_path, translate):
    """Load and transform a GLB model."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        logging.error("The GLB model has no vertices or triangles.")
        return None, None

    vertices = np.asarray(mesh.vertices)
    # Transformation to convert the GLB model to the right-handed coordinate system
    transformation_matrix = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    vertices = np.dot(vertices, transformation_matrix.T) + translate
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    
    # Create the initial transformation matrix as a homogeneous transformation
    initial_transformation = np.eye(4)
    initial_transformation[:3, :3] = transformation_matrix
    initial_transformation[:3, 3] = translate
    
    return mesh, initial_transformation


def align_mesh_centers(mesh1, mesh2):
    """Align the center of mesh2 to mesh1."""
    center1 = mesh1.get_center()
    center2 = mesh2.get_center()
    translation = center1 - center2
    vertices = np.asarray(mesh2.vertices) + translation
    mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh2, translation  # Return the translation used for alignment

def apply_optimal_params(mesh, optimal_angle, optimal_tx, optimal_ty):
    """Apply the optimal rotation and translation to the mesh."""
    # Apply optimal rotation
    rotation_matrix = mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(optimal_angle)))
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    print(f"Rotation Matrix: {rotation_matrix}")

    # Apply optimal translation
    vertices = np.asarray(mesh.vertices)
    vertices[:, :2] += [optimal_tx, optimal_ty]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Transformation matrix for the optimal parameters
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [optimal_tx, optimal_ty, 0]
    print(f"Transformation Matrix from Optimal parameters: {transformation_matrix}")

    return mesh

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