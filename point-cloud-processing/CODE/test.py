import logging
import time
import numpy as np
import open3d as o3d
from fetcher import fetch_json, process_feature_list
from mesh_processor import create_mesh_from_feature, apply_transformation
from transformation import optimize_rotation_and_translation, compute_z_offset, apply_z_offset, accumulate_transformations, create_center_based_transformation_matrix
from geometry_utils import extract_2d_perimeter, extract_latlon_orientation_from_mesh, calculate_intersection_error
from visualization import visualize_glb_and_combined_meshes, visualize_2d_perimeters, visualize_meshes_with_height_coloring
from icp_alignment import refine_alignment_with_icp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_transformation_matrix(file_path):
    """Load transformation matrix from a text file."""
    return np.loadtxt(file_path)


def load_optimal_params(file_path):
    """Load optimal parameters from a text file."""
    return np.loadtxt(file_path)


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

        # Initialize transformation matrix
        transformations = []
        
        # Load transformation matrix
        transformation_matrix_path = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
        transformation_matrix = load_transformation_matrix(transformation_matrix_path)
        logging.info(f"Loaded transformation matrix:\n{transformation_matrix}")
        transformations.append(transformation_matrix)
        
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
        transformations.append(transformation_matrix)
        final_transformation_matrix = accumulate_transformations(transformations)
        print("Transformations:\n", final_transformation_matrix)

        
        # Visualize the results
        visualize_meshes_with_height_coloring(combined_mesh, transformed_glb_mesh)

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
