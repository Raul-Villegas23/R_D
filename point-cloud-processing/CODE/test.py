# Libraries for API requests and data processing
import logging
import time

# Libraries for point cloud processing
import numpy as np
import open3d as o3d

 # Import the necessary functions from the custom modules
from fetcher import process_feature_list
from geometry_utils import extract_2d_perimeter
from visualization_utils import visualize_meshes_with_height_coloring, visualize_2d_perimeters


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        glb_model_path = data_folder + glb_dataset
        
        transformed_glb_mesh = o3d.io.read_triangle_mesh(glb_model_path, True)
        if not transformed_glb_mesh.has_vertices() or not transformed_glb_mesh.has_triangles():
            logging.error("The GLB model has no vertices or triangles.")
            return

        # Define the GLB model path
        transformed_glb_mesh = apply_transformation(transformed_glb_mesh, transformation_matrix_1)
        visualize_meshes_with_height_coloring(bag_mesh, transformed_glb_mesh, colormap_1='YlOrRd', colormap_2='YlGnBu')
        
        # Visualize the 3DBAG mesh and the aligned GLB mesh perimeters
        bag_perimeter = extract_2d_perimeter(bag_mesh)
        glb_perimeter = extract_2d_perimeter(transformed_glb_mesh)
        # visualize_2d_perimeters(bag_perimeter, glb_perimeter)
    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
