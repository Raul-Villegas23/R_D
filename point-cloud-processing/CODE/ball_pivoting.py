# Import necessary libraries
import numpy as np
import open3d as o3d
import laspy
import os
import pandas as pd

# Function to load .laz point cloud with RGB values
def load_laz_point_cloud(laz_file_path):
    # Read the .laz file using laspy
    with laspy.open(laz_file_path) as las_file:
        las_data = las_file.read()

    # Extract X, Y, Z coordinates
    xyz = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    # Check if RGB values exist in the .laz file
    if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue'):
        # Normalize RGB values (usually stored as integers from 0 to 65535, depending on the file format)
        red = las_data.red / 65535.0
        green = las_data.green / 65535.0
        blue = las_data.blue / 65535.0

        # Stack the RGB values and set them in the point cloud
        rgb = np.vstack((red, green, blue)).transpose()
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
    else:
        print("No RGB color information found in .laz file.")

    return point_cloud

# Function to load point cloud from a Pandas DataFrame (for .xyz files)
def load_point_cloud_from_dataframe(pcd_df):
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcd_df[['X', 'Y', 'Z']])))

    # Check if color information exists in the DataFrame and add it if available
    if all(col in pcd_df.columns for col in ['R', 'G', 'B']):
        pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(pcd_df[['R', 'G', 'B']]) / 255.0)
    else:
        print("No RGB color information found in DataFrame.")
    
    return pcd_o3d

# Function to load .xyz point cloud with RGB values
def load_xyz_point_cloud(xyz_file_path):
    # Load the .xyz file as a Pandas DataFrame
    pcd_df = pd.read_csv(xyz_file_path, sep=" ", header=None, names=['X', 'Y', 'Z', 'R', 'G', 'B'])

    # Convert the DataFrame to an Open3D point cloud
    return load_point_cloud_from_dataframe(pcd_df)

# Main point cloud loading function based on file extension
def load_point_cloud(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    
    if file_extension == ".laz":
        print("Loading .laz point cloud...")
        return load_laz_point_cloud(file_path)
    elif file_extension == ".xyz":
        print("Loading .xyz point cloud...")
        return load_xyz_point_cloud(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# The rest of the processing workflow follows
if __name__ == "__main__":
    # Path to point cloud file (either .laz or .xyz)
    point_cloud_file = "DATA/S1030801 - van hoornestr eo - blok_2.copc (1).laz"  # Change to your .laz file

    # Step 1: Load the point cloud based on file type
    point_cloud = load_point_cloud(point_cloud_file)
    print(f"Loaded point cloud has {len(point_cloud.points)} points.")
    
    # Step 2: Voxel-based downsampling for efficiency
    voxel_size = 0.5  # Adjust based on the level of detail
    point_cloud = point_cloud.voxel_down_sample(voxel_size)
    print(f"Voxel downsampled point cloud has {len(point_cloud.points)} points.")
    
    # Step 3: Center the point cloud (translate to the origin)
    point_cloud = point_cloud.translate(-point_cloud.get_center())
    
    # Step 4: Estimate normals using KD-Tree for efficiency
    if not point_cloud.has_normals():
        print("Estimating normals using KD-Tree...")
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Orient the normals consistently with the tangent plane to avoid issues in meshing
    o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(point_cloud, 10)
    
    # Step 5: Estimate nearest neighbor distance and calculate radius for ball-pivoting
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3.0 * avg_dist  # Increase radius for faster mesh generation
    print(f"Average neighbor distance = {avg_dist:.6f}")
    
    # Step 6: Create a mesh using the Ball-Pivoting algorithm with a larger radius
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    
    # Step 7: Visualize the mesh
    print(f"Mesh Details: {mesh}")
    # mesh.paint_uniform_color(np.array([0.5, 1.0, 0.5]))  # Color the mesh uniformly
    o3d.visualization.draw_geometries([mesh, point_cloud], window_name='Mesh with Estimated Normals', width=1200, height=800)
    
    # Step 8: Save the mesh to a .ply file
    output_mesh_folder = "RESULTS/"
    output_mesh_file = f"{output_mesh_folder}mesh_from_point_cloud.ply"
    o3d.io.write_triangle_mesh(output_mesh_file, mesh)
