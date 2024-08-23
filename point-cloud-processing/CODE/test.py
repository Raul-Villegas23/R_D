# Libraries for API requests and data processing
import logging
import time

# Libraries for point cloud processing
import numpy as np
import open3d as o3d

 # Import the necessary functions from the custom modules
from fetcher import process_feature_list
from geolocation import extract_latlon_orientation_from_mesh
from mesh_processor import load_and_transform_glb_model, align_mesh_centers, apply_optimal_params
from geometry_utils import extract_2d_perimeter, optimize_rotation_and_translation
from transformation_utils import compute_z_offset, apply_z_offset, accumulate_transformations, create_center_based_transformation_matrix
from visualization_utils import visualize_meshes_with_height_coloring
from icp_alignment import refine_alignment_with_icp


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_transformation_matrix(file_path):
    """Load transformation matrix from a text file."""
    return np.loadtxt(file_path)


def load_optimal_params(file_path):
    """Load optimal parameters from a text file."""
    return np.loadtxt(file_path)

def apply_transformation(mesh, transformation_matrix):
    """Apply transformation matrix to the mesh."""
    vertices = np.asarray(mesh.vertices)
    transformed_vertices = (transformation_matrix[:3, :3] @ vertices.T).T + transformation_matrix[:3, 3]
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    mesh.compute_vertex_normals()
    return mesh


def main():
    start_time = time.time()

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    # feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  # Pijlkruidstraat 11, 13 and 15
    feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb
    bag_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if bag_mesh:
        # Load GLB model and apply transformation
        data_folder = "DATA/"
        # glb_dataset = "pijlkruidstraat11-13-15.glb"
        glb_dataset = "pijlkruid37-37.glb"
        # glb_dataset = "rietstraat31-33.glb"

        # Initialize transformation matrix
        transformations = []
        
        # Load transformation matrix
        transformation_matrix_path = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
        transformation_matrix_1 = load_transformation_matrix(transformation_matrix_path) # Load transformation matrix from a text file
        logging.info(f"Loaded transformation matrix:\n{transformation_matrix_1}")
        transformations.append(transformation_matrix_1)
        
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
        transformed_glb_mesh = apply_transformation(transformed_glb_mesh, transformation_matrix_1)
        transformation_matrix_2 = create_center_based_transformation_matrix(transformed_glb_mesh, optimal_angle, optimal_tx, optimal_ty)
        print("Transformation Matrix :\n", transformation_matrix_2)

        # Apply this transformation to the mesh
        transformed_glb_mesh = apply_transformation(transformed_glb_mesh, transformation_matrix_2)
        transformations.append(transformation_matrix_2)
        final_transformation_matrix = accumulate_transformations(transformations)
        print(" Final Transformations:\n", final_transformation_matrix)

        transformed_glb_mesh = o3d.io.read_triangle_mesh(glb_model_path)
        transformed_glb_mesh = apply_transformation(transformed_glb_mesh, transformation_matrix_1)
        

        # Visualize the results
        visualize_meshes_with_height_coloring(bag_mesh, transformed_glb_mesh, colormap_1='YlGnBu', colormap_2='YlOrRd')
        
    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
