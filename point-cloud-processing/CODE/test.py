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

def process_glb_and_bag(feature_ids, glb_dataset, collections_url, collection_id):
    """Process a single GLB and BAG feature IDs."""
    bag_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)
    bag_mesh.compute_vertex_normals()

    if bag_mesh:
        data_folder = "DATA/"
        transformation_matrix_path = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
        transformation_matrix = load_transformation_matrix(transformation_matrix_path)
        logging.info(f"Loaded transformation matrix:\n{transformation_matrix}")

        glb_model_path = data_folder + glb_dataset
        transformed_glb_mesh = o3d.io.read_triangle_mesh(glb_model_path, enable_post_processing=True)

        if not transformed_glb_mesh.has_vertices() or not transformed_glb_mesh.has_triangles():
            logging.error("The GLB model has no vertices or triangles.")
            return

        transformed_glb_mesh = apply_transformation(transformed_glb_mesh, transformation_matrix)

        visualize_meshes_with_height_coloring(bag_mesh, transformed_glb_mesh, colormap_1='YlOrRd', colormap_2='YlGnBu')
        
        # Save combined mesh to file with the name of the GLB dataset as a .ply file
        # o3d.io.write_triangle_mesh(f"RESULTS/{glb_dataset.split('.')[0]}.ply", bag_mesh + transformed_glb_mesh, print_progress=True)


        bag_perimeter = extract_2d_perimeter(bag_mesh)
        glb_perimeter = extract_2d_perimeter(transformed_glb_mesh)
        # visualize_2d_perimeters(bag_perimeter, glb_perimeter)

def main():
    start_time = time.time()

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'

    # List of feature IDs and corresponding GLB datasets
    tasks = [
        {
            "feature_ids": ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"],
            "glb_dataset": "pijlkruidstraat11-13-15.glb"
        },
        {
            "feature_ids": ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"],
            "glb_dataset": "pijlkruid37-37.glb"
        },
        {
            "feature_ids": ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"],
            "glb_dataset": "rietstraat31-33.glb"
        }
    ]

    # Process each GLB dataset and its corresponding BAG feature IDs
    for task in tasks:
        feature_ids = task["feature_ids"]
        glb_dataset = task["glb_dataset"]
        logging.info(f"Processing GLB dataset {glb_dataset} with feature IDs {feature_ids}")
        process_glb_and_bag(feature_ids, glb_dataset, collections_url, collection_id)

    logging.info(f"Total elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
