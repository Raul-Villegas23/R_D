# Import libraries
import numpy as np
import open3d as o3d
import pandas as pd

# Step 1: Load the point cloud data using pandas (with semicolon delimiter)
file_path = "RESULTS/appartment_cloud.xyz"
point_cloud_df = pd.read_csv(file_path, delimiter=";", header=None)

# Assuming the point cloud has 4 columns: X, Y, Z, and labels
# Extract X, Y, Z coordinates
xyz = point_cloud_df[[0, 1, 2]].values  # Only use X, Y, Z columns (ignore the label column)

# Step 2: Create an Open3D PointCloud object and populate it with the XYZ coordinates
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(xyz)

# Step 3: Center the point cloud (translate to the origin)
point_cloud = point_cloud.translate(-point_cloud.get_center())

# Step 4: Estimate normals if they don't exist
if not point_cloud.has_normals():
    print("Estimating normals...")
    point_cloud.estimate_normals()
else:
    print("PCD already has normals. Skipping normal estimation.")

# Orient the normals consistently with the tangent plane to avoid issues in meshing
o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(point_cloud, 10)

# Step 5: Estimate nearest neighbor distance and calculate radius for ball-pivoting
distances = point_cloud.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist
print("Minimum neighbor distance = {:.6f}".format(np.min(distances)))
print("Maximum neighbor distance = {:.6f}".format(np.max(distances)))
print("Average neighbor distance = {:.6f}".format(avg_dist))

# Step 6: Create a mesh using the Ball-Pivoting algorithm
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    point_cloud,
    o3d.utility.DoubleVector([radius, radius * 2])
)

# Step 7: Visualize the mesh
print("PCD Details: {}".format(point_cloud))
print("Mesh Details: {}".format(mesh))
mesh.paint_uniform_color(np.array([0.5, 1.0, 0.5]))  # Color the mesh uniformly
o3d.visualization.draw_geometries([mesh], window_name='Mesh with Estimated Normals', width=1200, height=800)

# Step 8: Export the generated mesh to a PLY file
ply_file_path_read = file_path  # Assuming ply_file_path_read should be based on the input file
ply_file_path_write = f"RESULTS/{ply_file_path_read.split('/')[-1].split('.')[0]}_mesh.ply"
o3d.io.write_triangle_mesh(ply_file_path_write, mesh)
print("Mesh saved to: {}".format(ply_file_path_write))
