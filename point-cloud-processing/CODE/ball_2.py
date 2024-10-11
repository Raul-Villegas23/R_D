# Import libraries
import numpy as np
import open3d as o3d

# Load a point cloud
ply_file_path_read = "DATA/ITC_firstfloor.ply"
point_cloud = o3d.io.read_point_cloud(ply_file_path_read)
# Get the point cloud to the center of the coordinate system
point_cloud = point_cloud.translate(-point_cloud.get_center())
# point_cloud = point_cloud.uniform_down_sample(every_k_points=10) # if there are huge points
point_cloud.scale(1 / np.max(point_cloud.get_max_bound() - point_cloud.get_min_bound()),center=point_cloud.get_center()) # Fit to unit cube
o3d.io.write_point_cloud("bunny-pcd-unitcube.ply", point_cloud, write_ascii=True)

# Estimate radius for rolling ball
distances = point_cloud.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
print(f"Average neighbour distance = {np.mean(distances):.6}")

# Guess different voxel sizes based on points' distances
voxelsize = np.round(avg_dist, 5)
voxelsize_half = voxelsize/2
voxelsize_double = voxelsize*2

print(f"Different voxel size considered based on points' distances: {voxelsize}, {voxelsize_double}, {voxelsize_half}")

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxelsize)
print(f"Voxel center in X,Y,Z = {voxel_grid.get_center()}")
print(f"Sample Voxels: \n {voxel_grid.get_voxels()[:3]}")
o3d.visualization.draw_geometries([voxel_grid], width=1200, height=800)
o3d.io.write_voxel_grid("bunny-pcd-to-voxel-001-unitcube.ply", voxel_grid, write_ascii=True)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxelsize_half)
o3d.visualization.draw_geometries([voxel_grid], width=1200, height=800)
o3d.io.write_voxel_grid("bunny-pcd-to-voxel-002-unitcube.ply", voxel_grid, write_ascii=True)
print(f"Voxel center in X,Y,Z = {voxel_grid.get_center()}")
print(f"Sample Voxels: \n {voxel_grid.get_voxels()[:3]}")

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxelsize_double)
o3d.visualization.draw_geometries([voxel_grid], width=1200, height=800)
o3d.io.write_voxel_grid("bunny-pcd-to-voxel-003-unitcube.ply", voxel_grid, write_ascii=True)
print(f"Voxel center in X,Y,Z = {voxel_grid.get_center()}")
print(f"Sample Voxels: \n {voxel_grid.get_voxels()[:3]}")