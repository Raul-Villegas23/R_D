import logging
import time
import numpy as np
import open3d as o3d
from fetcher import fetch_json
from mesh_processor import create_mesh_from_feature, load_and_transform_glb_model
from visualization import visualize_glb_and_combined_meshes

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

def load_and_transform_glb_model(file_path, transformation_matrix):
    """Load and transform a GLB model using the given transformation matrix."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        logging.error("The GLB model has no vertices or triangles.")
        return None

    mesh = apply_transformation(mesh, transformation_matrix)
    return mesh

def main():
    start_time = time.time()

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  #  Pijlkruidstraat 11, 13 and 15
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb
    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if combined_mesh:
        # Load GLB model and apply transformation
        data_folder = "DATA/"
        glb_dataset = "pijlkruidstraat11-13-15.glb"
        # glb_dataset = "pijlkruid37-37.glb"
        # glb_dataset = "rietstraat31-33.glb"
        
        # Load transformation matrix
        transformation_matrix_path = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
        transformation_matrix = load_transformation_matrix(transformation_matrix_path)
        logging.info(f"Loaded transformation matrix:\n{transformation_matrix}")

        glb_model_path = data_folder + glb_dataset
        
        transformed_glb_mesh = load_and_transform_glb_model(glb_model_path, transformation_matrix)
        if transformed_glb_mesh:
            # Visualize the results
            visualize_glb_and_combined_meshes(combined_mesh, transformed_glb_mesh)

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
