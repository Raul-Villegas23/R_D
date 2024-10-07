import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the point cloud from a Pandas DataFrame (assume the DataFrame is already loaded)
def load_point_cloud_from_dataframe(pcd_df):
    """
    Load a point cloud from a Pandas DataFrame and convert it to an Open3D point cloud.
    
    Args:
    - pcd_df (pd.DataFrame): DataFrame with columns 'X', 'Y', 'Z' for coordinates, and optionally 'R', 'G', 'B' for color.
    
    Returns:
    - o3d.geometry.PointCloud: The loaded point cloud in Open3D format.
    """
    # Extract XYZ coordinates from DataFrame
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcd_df[['X', 'Y', 'Z']])))
    
    # Check if color information exists in the DataFrame and add it if available
    if all(col in pcd_df.columns for col in ['R', 'G', 'B']):
        pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(pcd_df[['R', 'G', 'B']]) / 255.0)  # Normalize colors
    else:
        print("No RGB color information found in DataFrame.")
    
    return pcd_o3d

# Step 2: Load the PLY file (Reference Mesh)
def load_ply_mesh(ply_file):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    mesh.compute_vertex_normals()  # Ensure normals are computed
    return mesh

# Step 3: Convert the mesh to a point cloud by sampling points
def sample_mesh_to_point_cloud(mesh, num_points=500):  # Increase sample points for better proximity detection
    sampled_pcd = mesh.sample_points_poisson_disk(num_points)
    return sampled_pcd

# Step 4: Crop the point cloud by selecting points near the mesh (preserving original RGB values)
def crop_point_cloud_by_mesh(point_cloud, mesh_sampled_pcd, threshold=1.0):  # Increase threshold to capture more points
    distances = point_cloud.compute_point_cloud_distance(mesh_sampled_pcd)
    
    # Get points that are within the threshold distance to the mesh
    indices = np.where(np.asarray(distances) < threshold)[0]
    cropped_points = point_cloud.select_by_index(indices)

    # Preserve the colors for the cropped point cloud (if color exists)
    if point_cloud.has_colors():
        cropped_points.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[indices])
    
    return cropped_points

# Step 5A: Create a mesh using Ball Pivoting algorithm (optimized for building structure)
def create_ball_pivoting_mesh(cropped_point_cloud, radii=[1.0, 2.0, 5.0]):  # Larger radii for building structures
    # Estimate normals for the cropped point cloud
    cropped_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=50))
    
    # Perform Ball Pivoting to create the mesh
    pcd_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cropped_point_cloud,
        o3d.utility.DoubleVector(radii),
    )
    
    return pcd_mesh

# Step 5B: Create a mesh using Poisson reconstruction (optimized for building structure)

def create_poisson_mesh(cropped_point_cloud, depth=12, point_weight=4.0, density_threshold=0.05, smoothing_iterations=5):
    """
    Create a less puffy mesh using Poisson surface reconstruction by increasing the depth, 
    adjusting point weight, and applying post-processing steps for refinement.

    Args:
    - cropped_point_cloud (o3d.geometry.PointCloud): The point cloud to be meshed.
    - depth (int): The depth of the Poisson reconstruction (higher values give more detail).
    - point_weight (float): The point weight controls how much the original points influence the surface.
    - density_threshold (float): The quantile threshold to remove low-density vertices (default: 5%).
    - smoothing_iterations (int): Number of iterations for Laplacian smoothing (default: 5).

    Returns:
    - poisson_mesh (o3d.geometry.TriangleMesh): The refined Poisson surface mesh.
    """
    
    # Step 1: Estimate normals for the point cloud (required for Poisson reconstruction)
    print("Estimating normals for Poisson reconstruction...")
    cropped_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=50))
    cropped_point_cloud.orient_normals_consistent_tangent_plane(100)

    # Step 2: Perform Poisson reconstruction with increased depth and point weight
    print(f"Performing Poisson surface reconstruction with depth={depth}, point_weight={point_weight}...")
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cropped_point_cloud, depth=depth, linear_fit=False)

    # Step 3: Visualize the density of the mesh using pseudo-color
    print("Visualizing the density of the mesh...")
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))[:, :3]
    
    # Apply the density colors to the vertices of the mesh
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = poisson_mesh.vertices
    density_mesh.triangles = poisson_mesh.triangles
    density_mesh.triangle_normals = poisson_mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    
    o3d.visualization.draw_geometries([density_mesh], mesh_show_back_face=True)

    # Step 4: Remove low-density vertices
    print(f"Removing low-density vertices below the {density_threshold} quantile...")
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    # Step 5: Apply Laplacian smoothing to make the mesh prettier and less puffy
    print(f"Applying Laplacian smoothing with {smoothing_iterations} iterations...")
    poisson_mesh = poisson_mesh.filter_smooth_laplacian(number_of_iterations=smoothing_iterations)

    # Step 6: Clean up the mesh by removing degenerate triangles and unreferenced vertices
    print("Removing degenerate triangles and unreferenced vertices...")
    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_unreferenced_vertices()

    # Step 7: Recompute normals for the final mesh
    poisson_mesh.compute_vertex_normals()

    print(f"Poisson mesh reconstruction completed with {len(poisson_mesh.vertices)} vertices remaining after post-processing.")

    return poisson_mesh


# Step 6: Map RGB values from point cloud to the mesh vertices
def map_colors_to_mesh(mesh, cropped_point_cloud):
    # Create an empty list to store the vertex colors
    vertex_colors = []

    # For each vertex in the mesh, find the nearest point in the cropped point cloud
    for vertex in mesh.vertices:
        distances = np.linalg.norm(np.asarray(cropped_point_cloud.points) - vertex, axis=1)
        nearest_point_idx = np.argmin(distances)
        vertex_colors.append(cropped_point_cloud.colors[nearest_point_idx])

    # Assign the colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))
    return mesh

# Main workflow
if __name__ == "__main__":
    # Load the point cloud DataFrame
    pcd_df = pd.read_csv("DATA/28GN1_08_sampled.xyz", delimiter=";")

    # Step 1: Load the point cloud from the DataFrame
    point_cloud = load_point_cloud_from_dataframe(pcd_df)
    
    # Step 2: Load the PLY reference mesh
    reference_mesh = load_ply_mesh("DATA/pijlkruid_aligned.ply")
    
    # Step 3: Sample more points from the mesh surface to create a point cloud
    mesh_sampled_pcd = sample_mesh_to_point_cloud(reference_mesh, num_points=5000)
    
    # Step 4: Crop the point cloud based on proximity to the mesh (preserving original colors)
    cropped_point_cloud = crop_point_cloud_by_mesh(point_cloud, mesh_sampled_pcd, threshold=0.5)
    
    
    # Step 5A: Create a mesh from the cropped point cloud using Ball Pivoting (optimized for buildings)
    ball_pivoting_mesh = create_ball_pivoting_mesh(mesh_sampled_pcd , radii=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    
    # Step 5B: Create a mesh from the cropped point cloud using Poisson reconstruction (optimized for buildings)
    poisson_mesh = create_poisson_mesh(mesh_sampled_pcd , depth=9, density_threshold=0.1)
    
    # Step 6: Map RGB colors from the point cloud to both the Ball Pivoting and Poisson meshes
    # ball_pivoting_mesh = map_colors_to_mesh(ball_pivoting_mesh, cropped_point_cloud)
    # poisson_mesh = map_colors_to_mesh(poisson_mesh, cropped_point_cloud)
    
    # Step 7: Visualize the original point cloud, reference mesh, cropped point cloud, and both meshes
    o3d.visualization.draw_geometries([reference_mesh,cropped_point_cloud, poisson_mesh])
    
    # Save the cropped point cloud and both meshes with colors (optional)
    o3d.io.write_point_cloud("RESULTS/cropped_point_cloud_with_rgb.ply", cropped_point_cloud)
    o3d.io.write_triangle_mesh("RESULTS/cropped_ball_pivoting_mesh_with_rgb.ply", ball_pivoting_mesh)
    o3d.io.write_triangle_mesh("RESULTS/cropped_poisson_mesh_with_rgb.ply", poisson_mesh)
