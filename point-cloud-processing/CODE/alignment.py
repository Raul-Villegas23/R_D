import requests
import numpy as np
import open3d as o3d
import logging
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.affinity import rotate
from pyproj import Transformer
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

def load_and_transform_glb_model(file_path, translate):
    """Load and transform a GLB model."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        logging.error("The GLB model has no vertices or triangles.")
        return None

    vertices = np.asarray(mesh.vertices)
    transformation_matrix = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    vertices = np.dot(vertices, transformation_matrix.T) + translate
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh

def compute_z_offset(combined_mesh, glb_mesh):
    """
    Compute the Z offset needed to align the floor of the GLB mesh with the combined mesh.
    """
    combined_bbox = combined_mesh.get_axis_aligned_bounding_box()
    glb_bbox = glb_mesh.get_axis_aligned_bounding_box()
    
    lowest_z_combined = combined_bbox.min_bound[2]
    lowest_z_glb = glb_bbox.min_bound[2]
    
    print(f"Lowest Z in Combined Mesh Bounding Box: {lowest_z_combined}")
    print(f"Lowest Z in GLB Mesh Bounding Box: {lowest_z_glb}")
    
    z_offset = lowest_z_combined - lowest_z_glb
    
    return z_offset

def apply_z_offset(mesh, z_offset):
    """
    Apply the Z offset to the mesh.
    """
    mesh.translate((0, 0, z_offset))

def extract_2d_perimeter(mesh):
    """Extract the 2D perimeter of the mesh by projecting onto the xy-plane and computing the convex hull."""
    vertices = np.asarray(mesh.vertices)[:, :2]
    hull = ConvexHull(vertices)
    perimeter_points = vertices[hull.vertices]
    return np.vstack([perimeter_points, perimeter_points[0]])

def calculate_intersection_error(params, perimeter1, perimeter2):
    """Calculate the error between intersections of two perimeters after rotating and translating one."""
    angle, tx, ty = params
    rotated_perimeter2 = rotate(Polygon(perimeter2), angle, origin='centroid')
    translated_perimeter2 = np.array(rotated_perimeter2.exterior.coords) + [tx, ty]
    poly1, poly2 = Polygon(perimeter1), Polygon(translated_perimeter2)
    intersection = poly1.intersection(poly2)
    union = poly1.union(poly2)
    return 1 - (intersection.area / union.area) if union.area != 0 else 0

def optimize_rotation_and_translation(perimeter1, perimeter2):
    """Optimize rotation angle and translation to align two perimeters."""
    initial_guesses = [[-45.0, 0.0, 0.0], [45.0, 0.0, 0.0], [90.0, 0.0, 0.0], [-90.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    bounds = [(-180, 180), (-np.inf, np.inf), (-np.inf, np.inf)]
    best_result, lowest_error = None, float('inf')

    for initial_guess in initial_guesses:
        result = minimize(calculate_intersection_error, initial_guess, args=(perimeter1, perimeter2), method='L-BFGS-B', bounds=bounds)
        if result.success and result.fun < lowest_error:
            best_result, lowest_error = result, result.fun

    return best_result.x if best_result else None

def extract_latlon_orientation_from_mesh(mesh, reference_system):
    """Extract longitude, latitude, and orientation from mesh vertices."""
    vertices = np.asarray(mesh.vertices)
    epsg_code = reference_system.split('/')[-1]
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    latlon_vertices = np.array([transformer.transform(x, y) for x, y, z in vertices])
    centroid = np.mean(latlon_vertices, axis=0)
    hull = ConvexHull(latlon_vertices)
    hull_vertices = latlon_vertices[hull.vertices]
    longest_edge = max(((hull_vertices[i], hull_vertices[j]) for i in range(len(hull_vertices)) for j in range(i+1, len(hull_vertices))), key=lambda edge: np.linalg.norm(edge[1] - edge[0]))
    orientation_angle = (np.degrees(np.arctan2(longest_edge[1][1] - longest_edge[0][1], longest_edge[1][0] - longest_edge[0][0])) + 360) % 360
    return centroid[1], centroid[0], orientation_angle

def align_mesh_centers(mesh1, mesh2):
    """Align the center of mesh2 to mesh1."""
    center1 = mesh1.get_center()
    center2 = mesh2.get_center()
    translation = center1 - center2
    vertices = np.asarray(mesh2.vertices) + translation
    mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh2, translation  # Return the translation used for alignment

def calculate_transformation_matrix(initial_transformation, angle, translation, center_translation, z_offset):
    """
    Calculate the transformation matrix for initial transformation, rotation angle, translation, centering, and Z offset.
    
    :param initial_transformation: The initial transformation matrix (3x3) to apply.
    :param angle: The rotation angle in degrees.
    :param translation: The translation vector (x, y, z).
    :param center_translation: Additional center translation (x, y, z).
    :param z_offset: The Z offset to apply.
    :return: The final transformation matrix (4x4).
    """
    cos_theta = np.cos(np.radians(angle))
    sin_theta = np.sin(np.radians(angle))
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # Create the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    # Create the initial transformation matrix
    initial_transformation_matrix = np.eye(4)
    initial_transformation_matrix[:3, :3] = initial_transformation

    # Combine the transformations
    combined_transformation = np.eye(4)
    combined_transformation[:3, :3] = rotation_matrix @ initial_transformation_matrix[:3, :3]
    combined_transformation[:3, 3] = translation_matrix[:3, 3] + center_translation
    
    # Add Z offset
    combined_transformation[2, 3] += z_offset
    
    return combined_transformation

def main():
    start_time = time.time()

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"] # pijlkruidstraat11-13-15.glb Pijlkruidstraat 11, 13 and 15
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb

    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)
    
    if combined_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/" 
        glb_dataset = "pijlkruidstraat11-13-15.glb"
        # glb_dataset = "pijlkruid37-37.glb"
        # glb_dataset = "rietstraat31-33.glb"

        glb_model_path = data_folder + glb_dataset
        glb_mesh = load_and_transform_glb_model(glb_model_path, translate)
        if glb_mesh:
            glb_mesh, center_translation = align_mesh_centers(combined_mesh, glb_mesh)
            perimeter1, perimeter2 = extract_2d_perimeter(combined_mesh), extract_2d_perimeter(glb_mesh)
            optimal_params = optimize_rotation_and_translation(perimeter1, perimeter2)
            if optimal_params is not None:
                optimal_angle, optimal_tx, optimal_ty = optimal_params
                glb_mesh.rotate(glb_mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(optimal_angle))), center=glb_mesh.get_center())
                vertices = np.asarray(glb_mesh.vertices)
                vertices[:, :2] += [optimal_tx, optimal_ty]
                glb_mesh.vertices = o3d.utility.Vector3dVector(vertices)

                # Apply optimal rotation and translation
                glb_mesh.rotate(glb_mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(optimal_angle))), center=glb_mesh.get_center())
                vertices = np.asarray(glb_mesh.vertices)
                vertices[:, :2] += [optimal_tx, optimal_ty]
                glb_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                try:
                    z_offset = compute_z_offset(combined_mesh, glb_mesh)
                    print(f"Calculated Z offset: {z_offset}")  # Print the Z offset
                    apply_z_offset(glb_mesh, z_offset)
                except ValueError as e:
                    print(f"Error computing Z-offset: {e}")
                    return
                
                lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
                logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")

                initial_transformation = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
                transformation_matrix = calculate_transformation_matrix(initial_transformation, optimal_angle, translate, center_translation, z_offset)
                print(f"Transformation Matrix:\n{transformation_matrix}")
                np.savetxt("RESULTS/transformation_matrix_1.txt", transformation_matrix)
                with open("RESULTS/lat_lon_orientation.txt", "w") as file:
                    file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
