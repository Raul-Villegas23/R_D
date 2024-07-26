import requests
import numpy as np
import open3d as o3d
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_json(url):
    """Fetch JSON data from a specified URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to retrieve data: {response.status_code}")
        return None

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

def visualize_meshes(combined_mesh, transformed_glb_mesh):
    """Visualize the combined mesh and the transformed GLB mesh."""
    transformed_glb_mesh.paint_uniform_color([0.1, 0.5, 0.9])
    combined_mesh.paint_uniform_color([0.9, 0.5, 0.1])

    o3d.visualization.draw_geometries([combined_mesh, transformed_glb_mesh],
                                      window_name="Combined Mesh vs Transformed GLB Mesh",
                                      width=800, height=600, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True)

def main():
    start_time = time.time()

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  #  Pijlkruidstraat 11, 13 and 15
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb
    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if combined_mesh:
        # Load transformation matrix
        transformation_matrix_path = "RESULTS/transformation_matrix.txt"
        transformation_matrix = load_transformation_matrix(transformation_matrix_path)
        logging.info(f"Loaded transformation matrix:\n{transformation_matrix}")

        # Load GLB model and apply transformation
        data_folder = "DATA/"
        glb_dataset = "pijlkruidstraat11-13-15.glb"
        # glb_dataset = "pijlkruid37-37.glb"
        # glb_dataset = "rietstraat31-33.glb"
        
        glb_model_path = data_folder + glb_dataset
        
        transformed_glb_mesh = load_and_transform_glb_model(glb_model_path, transformation_matrix)
        if transformed_glb_mesh:
            # Visualize the results
            visualize_meshes(combined_mesh, transformed_glb_mesh)

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
