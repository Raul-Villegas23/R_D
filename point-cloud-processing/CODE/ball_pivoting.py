# Import libraries
import numpy as np
import open3d as o3d

# Load a point cloud
ply_file_path_read = "DATA/ITC_firstfloor.ply"
point_cloud = o3d.io.read_point_cloud(ply_file_path_read)
# Get the point cloud to the center of the coordinate system
point_cloud = point_cloud.translate(-point_cloud.get_center())

# Compute normals if PCD does not have
if point_cloud.has_normals() == False:
    print("Estimating normals...")
    point_cloud.estimate_normals()
else:
    print("PCD already has normals. So, skip estimating normals")

# Orient the computed normals w.r.t to the tangent plane. 
# This step will solve the normal direction issue. If this step is skipped, there might be holes in the mesh surfaces.
o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(point_cloud, 10)

# Estimate radius for rolling ball
distances = point_cloud.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist   
print("Minimum neighbour distance = {:.6f}".format(np.min(distances)))
print("Maximum neighbour distance = {:.6f}".format(np.max(distances))) 
print("Average neighbour distance = {:.6f}".format(np.mean(distances)))

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                                    point_cloud, 
                                    o3d.utility.DoubleVector([radius, radius * 2]))

print("PCD Detail: {}".format(point_cloud))
print("Mesh Details: {}".format(mesh))

# Visualize and save the mesh generated from point cloud
mesh.paint_uniform_color(np.array([0.5, 1.0, 0.5])) # to uniformly color the surface
o3d.visualization.draw_geometries([mesh], window_name='mesh with estimated normals', width=1200, height=800)

#Export the mesh to a file with a formmatted string
ply_file_path_write = f"RESULTS/{ply_file_path_read.split('/')[-1].split('.')[0]}_mesh.ply"
o3d.io.write_triangle_mesh(ply_file_path_write, mesh)
print("Mesh saved to: {}".format(ply_file_path_write))