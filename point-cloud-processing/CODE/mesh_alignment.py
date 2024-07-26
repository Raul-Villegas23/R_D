import requests
import json
import numpy as np
import open3d as o3d
import logging
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import rotate
from scipy.optimize import minimize

from pyproj import Transformer
from geopy.geocoders import Nominatim
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_json(url):
    """Fetch JSON data from a specified URL."""
    with requests.Session() as session:
        response = session.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to retrieve data: {response.status_code}")
            return None

def create_mesh_from_feature(feature, feature_id):
    """Create and save a mesh object from feature data."""
    if 'vertices' in feature['feature']:
        vertices = np.array(feature['feature']['vertices'])
        if 'transform' in feature['metadata']:
            transform = feature['metadata']['transform']
            scale = np.array(transform['scale'])
            translate = np.array(transform['translate'])
            vertices = vertices * scale + translate  # Apply transformation
        else:
            logging.error("Transformation data missing in the feature.")
            return None, None, None

        # Create a mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # If 'boundaries' data exists and is properly formatted
        if 'CityObjects' in feature['feature']:
            city_object = feature['feature']['CityObjects'][next(iter(feature['feature']['CityObjects']))]
            if 'geometry' in city_object:
                for geom in city_object['geometry']:
                    for boundary in geom['boundaries']:
                        # Assuming all boundaries are triangulated or simple polygons
                        for i in range(1, len(boundary[0]) - 1):  # Use boundary[0] if boundaries are nested lists
                            mesh.triangles.append([boundary[0][0], boundary[0][i], boundary[0][i + 1]])

        mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
        # logging.info(f"Created mesh with {len(vertices)} vertices and {len(mesh.triangles)} triangles.")
        return mesh, scale, translate
    else:
        logging.error("No vertices found in the feature data.")
        return None, None, None

def process_feature_list(collections_url, collection_id, feature_ids):
    """Process a list of feature IDs, visualize and save their meshes."""
    meshes = []
    scale = None
    translate = None
    reference_system = None

    for feature_id in feature_ids:
        feature_url = f"{collections_url}/{collection_id}/items/{feature_id}"
        logging.info(f"Processing feature: {feature_id}")
        feature = fetch_json(feature_url)
        if feature:
            mesh, scale, translate = create_mesh_from_feature(feature, feature_id)
            if mesh:
                meshes.append(mesh)

            # Extract reference system
            if 'metadata' in feature and 'metadata' in feature['metadata'] and 'referenceSystem' in feature['metadata']['metadata']:
                reference_system = feature['metadata']['metadata']['referenceSystem']
        else:
            logging.error(f"Failed to fetch feature: {feature_id}")

    if meshes:
        combined_mesh = o3d.geometry.TriangleMesh()
        for m in meshes:
            combined_mesh += m

        return combined_mesh, scale, translate, reference_system
    else:
        logging.error("No meshes to visualize.")
        return None, None, None, None



def visualize_glb_and_combined_meshes(mesh1, mesh2):
    """Visualize the GLB and combined meshes using Matplotlib."""
    vertices1 = np.asarray(mesh1.vertices)
    triangles1 = np.asarray(mesh1.triangles)
    
    # Simplify the second mesh for visualization purposes
    if mesh2.has_triangle_uvs():
        mesh2.triangle_uvs = o3d.utility.Vector2dVector([])
    
    mesh2 = mesh2.simplify_quadric_decimation(1000) #1000 is the number of vertices after simplification
    vertices2 = np.asarray(mesh2.vertices)
    triangles2 = np.asarray(mesh2.triangles)

    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], c='k', marker='o', s=5, label='3D BAG Mesh Vertices')

    # Create a 3D surface plot using plot_trisurf with a colormap
    ax.plot_trisurf(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], triangles=triangles1, cmap='plasma', edgecolor='k', alpha=0.5)
    ax.plot_trisurf(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], triangles=triangles2, cmap='plasma', edgecolor='k', alpha=0.5)  
    
    # Auto scale to the mesh size
    scale = np.concatenate((vertices1, vertices2)).flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    # Figure name
    ax.set_title('3D BAG and GLB Meshes')
    
    # Set axis limits based on the range of vertices
    xlim = (min(vertices1[:, 0].min(), vertices2[:, 0].min()), max(vertices1[:, 0].max(), vertices2[:, 0].max()))
    ylim = (min(vertices1[:, 1].min(), vertices2[:, 1].min()), max(vertices1[:, 1].max(), vertices2[:, 1].max()))
    zlim = (min(vertices1[:, 2].min(), vertices2[:, 2].min()), max(vertices1[:, 2].max(), vertices2[:, 2].max()))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Show the plot
    plt.show()

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

def load_and_transform_glb_model(file_path, translate):
    """Load a GLB model, remap y to z, apply translation, reflect the x-axis, and apply Z offset."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        logging.error("The GLB model has no vertices or triangles.")
        return None

    vertices = np.asarray(mesh.vertices)
    # Define the transformation matrix
    # Reflect the x-axis and remap y to z
    transformation_matrix = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    # Apply the transformation
    vertices = np.dot(vertices, transformation_matrix.T)
    # Apply translation
    vertices = vertices + translate
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()

    return mesh


def extract_2d_perimeter(mesh):
    """Extract the 2D perimeter of the mesh by projecting onto the xy-plane and computing the convex hull."""
    vertices = np.asarray(mesh.vertices)[:, :2]  # Ignore z-values by taking only x and y
    hull = ConvexHull(vertices)
    perimeter_points = vertices[hull.vertices]

    # Close the loop by appending the first point to the end
    perimeter_points = np.vstack([perimeter_points, perimeter_points[0]])

    return perimeter_points

def calculate_centroid(perimeter):
    """Calculate the centroid of a given perimeter using Shapely."""
    polygon = Polygon(perimeter)
    centroid = polygon.centroid

    return np.array([centroid.x, centroid.y])


def visualize_2d_perimeters(perimeter1, perimeter2, perimeter3):
    """Visualize three 2D perimeters, their centroids, orientations, and longest edges using Matplotlib."""
    fig, ax = plt.subplots()
    ax.plot(perimeter1[:, 0], perimeter1[:, 1], 'r-', label='3D BAG Mesh Perimeter')
    ax.plot(perimeter2[:, 0], perimeter2[:, 1], 'b-', label='GLB Mesh Perimeter')
    ax.plot(perimeter3[:, 0], perimeter3[:, 1], 'g--', label='Non-aligned Perimeter')  # Dashed lines for the third perimeter

    # Calculate and plot centroids
    centroid1 = calculate_centroid(perimeter1)
    centroid2 = calculate_centroid(perimeter2)
    centroid3 = calculate_centroid(perimeter3)
    
    ax.plot(centroid1[0], centroid1[1], 'ro', label='Centroid 3D BAG Mesh')
    ax.plot(centroid2[0], centroid2[1], 'bo', label='Centroid GLB Mesh')
    ax.plot(centroid3[0], centroid3[1], 'go', label='Centroid Non-aligned')

    # Compute and display orientations
    orientation1, longest_edge1 = compute_orientation(perimeter1)
    orientation2, longest_edge2 = compute_orientation(perimeter2)
    orientation3, longest_edge3 = compute_orientation(perimeter3)

    # ax.text(centroid1[0], centroid1[1], f'{orientation1:.1f}°', color='red', fontsize=12, ha='right')
    ax.text(centroid2[0], centroid2[1], f'{orientation2:.1f}°', color='blue', fontsize=12, ha='right')
    # ax.text(centroid3[0], centroid3[1], f'{orientation3:.1f}°', color='green', fontsize=12, ha='right')

    # Plot north and east direction arrow (adjust the coordinates as needed)
    ax.plot([centroid2[0], centroid2[0]], [centroid2[1], centroid2[1] + 6], 'k--', linewidth= 0.5)
    ax.plot([centroid2[0], centroid2[0] + 6], [centroid2[1], centroid2[1]], 'k--', linewidth= 0.5)
 
    # Plot the longest edges
    # if longest_edge1[0] is not None and longest_edge1[1] is not None:
    #     ax.plot([longest_edge1[0][0], longest_edge1[1][0]], [longest_edge1[0][1], longest_edge1[1][1]], 'r--', linewidth=2, label='Longest Edge 3D BAG Mesh')
    if longest_edge2[0] is not None and longest_edge2[1] is not None:
        ax.plot([longest_edge2[0][0], longest_edge2[1][0]], [longest_edge2[0][1], longest_edge2[1][1]], 'b--', linewidth=1, label='Longest Edge GLB Mesh')
    # if longest_edge3[0] is not None and longest_edge3[1] is not None:
    #     ax.plot([longest_edge3[0][0], longest_edge3[1][0]], [longest_edge3[0][1], longest_edge3[1][1]], 'g--', linewidth=2, label='Longest Edge Non-optimized')

    # Plot orientation lines from centroid to the direction given by orientation angle
    def plot_orientation_line(centroid, orientation, color):
        length = 2.0  # Length of the orientation line
        end_x = centroid[0] + length * np.cos(np.radians(orientation))
        end_y = centroid[1] + length * np.sin(np.radians(orientation))
        ax.plot([centroid[0], end_x], [centroid[1], end_y], color=color, linestyle='--')

    # plot_orientation_line(centroid1, orientation1, 'red')
    plot_orientation_line(centroid2, orientation2, 'blue')
    # plot_orientation_line(centroid3, orientation3, 'green')
    # Adjust legend position to be outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_title('2D Perimeters')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make room for the legend
    plt.show()

def align_mesh_centers(mesh1, mesh2):
    """Align the center of mesh2 to mesh1."""
    center1 = mesh1.get_center()
    center2 = mesh2.get_center()
    translation = center1 - center2
    vertices = np.asarray(mesh2.vertices) + translation
    mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh2, translation  # Return the translation used for alignment

def calculate_intersection_error(params, perimeter1, perimeter2):
    """Calculate the error between the intersections of two perimeters after rotating and translating one by given parameters."""
    angle, tx, ty = params
    rotated_perimeter2 = rotate(Polygon(perimeter2), angle, origin='centroid')
    translated_perimeter2 = np.array(rotated_perimeter2.exterior.coords) + [tx, ty]

    poly1 = Polygon(perimeter1)
    poly2 = Polygon(translated_perimeter2)
    intersection = poly1.intersection(poly2)
    union = poly1.union(poly2)
    if union.area == 0:
        return 0
    error = 1 - intersection.area / union.area
    logging.debug(f"Angle: {angle}, Translation: ({tx}, {ty}), Intersection Error: {error}")
    return error


def optimize_rotation_and_translation(perimeter1, perimeter2, num_attempts=5):
    """Find the optimal rotation angle and translation to align two perimeters by minimizing the intersection error."""
    best_result = None
    lowest_error = float('inf')
    initial_guesses = [
        [-45.0, 0.0, 0.0],
        [45.0, 0.0, 0.0],
        [90.0, 0.0, 0.0],
        [-90.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
    
    method = 'L-BFGS-B'  # Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
    bounds = [(-180, 180), (-np.inf, np.inf), (-np.inf, np.inf)]  # Bounds for [angle, tx, ty]
    
    for attempt in range(min(num_attempts, len(initial_guesses))):
        initial_guess = initial_guesses[attempt]
        logging.info(f"Attempt {attempt + 1}: Initial guess = {initial_guess}")
        
        try:
            result = minimize(calculate_intersection_error, initial_guess, args=(perimeter1, perimeter2), method=method, bounds=bounds)
            if result.success and result.fun < lowest_error:
                best_result = result
                lowest_error = result.fun
                logging.info(f"New best result found: {result.x} with error {result.fun}")
        except Exception as e:
            logging.error(f"Optimization attempt {attempt + 1} failed: {e}")
    
    if best_result is not None:
        return best_result.x
    else:
        logging.error("All optimization attempts failed.")
        return None


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


def compute_orientation(vertices):
    """Compute the orientation of the building based on the azimuth angle of the longest edge relative to the north."""
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    
    max_length = 0
    orientation_angle = 0
    longest_edge = (None, None)
    
    for i in range(len(hull_vertices)):
        for j in range(i + 1, len(hull_vertices)):
            vec = hull_vertices[j] - hull_vertices[i]
            length = np.linalg.norm(vec)
            if length > max_length:
                max_length = length
                # Calculate the azimuth angle relative to the north (y-axis)
                orientation_angle = (np.degrees(np.arctan2(vec[1], vec[0])) + 360) % 360
                longest_edge = (hull_vertices[i], hull_vertices[j])
    
    return orientation_angle, longest_edge

def extract_latlon_orientation_from_mesh(mesh, reference_system):
    """Extract longitude, latitude, and orientation from mesh vertices."""
    vertices = np.asarray(mesh.vertices)
    
    # Extract reference system EPSG code
    epsg_code = reference_system.split('/')[-1]
    
    # Convert to latitude and longitude using pyproj Transformer
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    latlon_vertices = np.array([transformer.transform(x, y) for x, y, z in vertices])

    # Compute orientation based on the convex hull's longest edge
    orientation_angle, longest_edge = compute_orientation(latlon_vertices) # (latlon_vertices)

    # Extract the latitude and longitude from the center of the mesh
    lon, lat = np.mean(latlon_vertices, axis=0)

    # Log the results
    logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation_angle:.5f} degrees")
    return lon, lat, orientation_angle

def transform_coordinates(lat, lon, reference_system):
    """Transform coordinates to EPSG:7415 if they are not already in that reference system."""
    epsg_code = reference_system.split('/')[-1]
    if epsg_code != '7415':
        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:7415", always_xy=True)
        lon, lat = transformer.transform(lon, lat)
    return lat, lon

def get_geo_location(lat, lon, reference_system):
    """Given latitude and longitude, return the geo location using Nominatim."""
    # Ensure coordinates are in EPSG:7415
    lat, lon = transform_coordinates(lat, lon, reference_system)
    
    # Use Nominatim geolocator
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.reverse((lat, lon), exactly_one=True)
    
    if location:
        address = location.address
        logging.info(f"Address: {address}")
        return address
    else:
        logging.error("Unable to retrieve location information.")
        return None
# Function to apply height-based coloring to a mesh
def color_mesh_by_height(mesh):
    vertices = np.asarray(mesh.vertices)
    heights = vertices[:, 2]
    min_height = np.min(heights)
    max_height = np.max(heights)
    normalized_heights = (heights - min_height) / (max_height - min_height)
    colors = plt.get_cmap('plasma')(normalized_heights)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def main():
    start_time = time.time()

    # Define API and dataset parameters
    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  # model.glb Pijlkruidstraat 11, 13 and 15
    # Uncomment for different datasets:
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"]  # pijlkruid37-37.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"]  # rietstraat31-33.glb

    # Process the feature list
    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)
    
    if combined_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        glb_dataset = "pijlkruidstraat11-13-15.glb"
        # Uncomment for different GLB models:
        # glb_dataset = "pijlkruid37-37.glb"
        # glb_dataset = "rietstraat31-33.glb"
        glb_model_path = data_folder + glb_dataset

        # Load the GLB model
        glb_mesh = o3d.io.read_triangle_mesh(glb_model_path)
        if not glb_mesh.has_vertices() or not glb_mesh.has_triangles():
            logging.error("The GLB model has no vertices or triangles.")
            return
        # Transform the GLB model with Z offset
        glb_mesh = load_and_transform_glb_model(glb_model_path, translate)

        
        if glb_mesh:
            # Align the GLB mesh with the combined mesh
            glb_mesh, center_translation = align_mesh_centers(combined_mesh, glb_mesh)
            
            # Extract 2D perimeters for optimization
            perimeter1 = extract_2d_perimeter(combined_mesh)
            perimeter2 = extract_2d_perimeter(glb_mesh)
            
            # Optimize rotation and translation
            optimal_params = optimize_rotation_and_translation(perimeter1, perimeter2)
            if optimal_params is not None:
                optimal_angle, optimal_tx, optimal_ty = optimal_params
                
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
                
                # Save the optimized GLB mesh as a .ply file with the name of the GLB dataset + .ply
                # ply_filename = glb_dataset.replace('.glb', '.ply')
                # o3d.io.write_triangle_mesh(f"RESULTS/{ply_filename}", glb_mesh)

                # Extract and log geographic data
                lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
                get_geo_location(lat, lon, reference_system)

                # Calculate and save the transformation matrix
                initial_transformation = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
                transformation_matrix = calculate_transformation_matrix(initial_transformation, optimal_angle, translate, center_translation, z_offset)
                print(transformation_matrix)
                np.savetxt("RESULTS/transformation_matrix.txt", transformation_matrix)
                
                # Save latitude, longitude, and orientation
                with open("RESULTS/lat_lon_orientation.txt", "w") as file:
                    file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")

                # Visualize meshes and perimeters with Matplotlib before and after optimization
                visualize_glb_and_combined_meshes(combined_mesh, glb_mesh)

                # Apply height-based coloring
                glb_mesh = color_mesh_by_height(glb_mesh)
                combined_mesh = color_mesh_by_height(combined_mesh)
                o3d.visualization.draw_geometries([combined_mesh, glb_mesh], window_name="3D BAG and GLB Meshes", width=800, height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=True)

                # Visualize 2D perimeters
                rotated_perimeter2 = rotate(Polygon(perimeter2), optimal_angle, origin='centroid')
                translated_rotated_perimeter2 = np.array(rotated_perimeter2.exterior.coords) + [optimal_tx, optimal_ty]
                visualize_2d_perimeters(perimeter1, translated_rotated_perimeter2, perimeter2)

                # Print error after optimization
                error = calculate_intersection_error(optimal_params, perimeter1, perimeter2)
                logging.info(f"Intersection Error after optimization: {error:.5f}")

                # # Save the optimized GLB mesh as a .ply file with the name of the GLB dataset + .ply
                # ply_filename = glb_dataset.replace('.glb', '.ply')
                # o3d.io.write_triangle_mesh(f"RESULTS/{ply_filename}", glb_mesh)

                

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()



