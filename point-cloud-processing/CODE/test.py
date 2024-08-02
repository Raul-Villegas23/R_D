import logging
import time
import numpy as np
import open3d as o3d
from fetcher import fetch_json
from mesh_processor import create_mesh_from_feature, load_and_transform_glb_model, align_mesh_centers, apply_optimal_params
from geometry_utils import extract_2d_perimeter, extract_latlon_orientation_from_mesh, calculate_intersection_error
from transformation import optimize_rotation_and_translation, compute_z_offset, apply_z_offset, accumulate_transformations
from visualization import visualize_glb_and_combined_meshes, visualize_2d_perimeters, visualize_meshes_with_height_coloring
from icp_alignment import refine_alignment_with_icp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_feature_list(collections_url, collection_id, feature_ids):
    """Process a list of feature IDs and combine their meshes."""
    meshes, scale, translate, reference_system = [], None, None, None

    for feature_id in feature_ids:
        feature_url = f"{collections_url}/{collection_id}/items/{feature_id}"
        feature = fetch_json(feature_url)
        if feature:
            mesh, scale, translate = create_mesh_from_feature(feature)
            if mesh:
                meshes.append(mesh)
            reference_system = feature['metadata'].get('metadata', {}).get('referenceSystem')

    if meshes:
        combined_mesh = sum(meshes, o3d.geometry.TriangleMesh())
        return combined_mesh, scale, translate, reference_system
    else:
        logging.error("No meshes to visualize.")
        return None, None, None, None

def load_transformation_matrix(file_path):
    """Load transformation matrix from a text file."""
    return np.loadtxt(file_path)

def apply_transformation(mesh, transformation_matrix):
    """Apply transformation matrix to the mesh."""
    vertices = np.asarray(mesh.vertices)
    transformed_vertices = (transformation_matrix[:3, :3] @ vertices.T).T + transformation_matrix[:3, 3]
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    mesh.compute_vertex_normals()
    return mesh


def load_optimal_params(file_path):
    """Load optimal parameters from a text file."""
    return np.loadtxt(file_path)

def rotation_matrix(axis, angle, is_degree=True):
    """Generate a rotation matrix given an axis and an angle."""
    if is_degree:
        angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
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


def main():
    start_time = time.time()

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    # feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  # Pijlkruidstraat 11, 13 and 15
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb
    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if combined_mesh:
        # Load GLB model and apply transformation
        data_folder = "DATA/"
        # glb_dataset = "pijlkruidstraat11-13-15.glb"
        # glb_dataset = "pijlkruid37-37.glb"
        glb_dataset = "rietstraat31-33.glb"
        
        # Load transformation matrix
        transformation_matrix_path = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
        transformation_matrix = load_transformation_matrix(transformation_matrix_path)
        logging.info(f"Loaded transformation matrix:\n{transformation_matrix}")
        
        # Load optimal parameters
        optimal_params_path = f"RESULTS/{glb_dataset.split('.')[0]}_optimal_params.txt"
        optimal_params = load_optimal_params(optimal_params_path)
        optimal_angle, optimal_tx, optimal_ty = optimal_params
        logging.info(f"Loaded optimal parameters: angle={optimal_angle}, tx={optimal_tx}, ty={optimal_ty}")

        glb_model_path = data_folder + glb_dataset
        
        transformed_glb_mesh = o3d.io.read_triangle_mesh(glb_model_path)
        if not transformed_glb_mesh.has_vertices() or not transformed_glb_mesh.has_triangles():
            logging.error("The GLB model has no vertices or triangles.")
            return

        # Apply the loaded transformation matrix
        transformed_glb_mesh = apply_transformation(transformed_glb_mesh, transformation_matrix)
        transformation_matrix = create_center_based_transformation_matrix(transformed_glb_mesh, optimal_angle, optimal_tx, optimal_ty)
        print("Transformation Matrix 4x4:\n", transformation_matrix)
        # Apply this transformation to the mesh
        transformed_glb_mesh = apply_transformation(transformed_glb_mesh, transformation_matrix)

        # Apply the optimal rotation
        # rotation_mat = rotation_matrix('z', optimal_angle)
        # print("Rotation Matrix 3x3:\n", rotation_mat)

        # # Apply the rotation to the mesh
        # transformed_glb_mesh.rotate(rotation_mat, center=transformed_glb_mesh.get_center())
        
        # Visualize the results
        visualize_meshes_with_height_coloring(combined_mesh, transformed_glb_mesh)

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
