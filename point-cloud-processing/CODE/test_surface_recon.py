import open3d as o3d
import numpy as np

def create_mesh_from_point_cloud_poisson(point_cloud, depth=10):
    """
    Convert a point cloud into a triangle mesh using Poisson surface reconstruction.
    
    Args:
    - point_cloud (o3d.geometry.PointCloud): The point cloud to be converted to a mesh.
    - depth (int): The depth of the Poisson reconstruction (higher values give more detail).
    
    Returns:
    - mesh (o3d.geometry.TriangleMesh): The reconstructed triangle mesh.
    """
    
    # Step 1: Estimate normals (required for Poisson reconstruction)
    print("Estimating normals for Poisson reconstruction...")
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=50))
    point_cloud.orient_normals_consistent_tangent_plane(100)
    
    # Step 2: Perform Poisson surface reconstruction
    print(f"Performing Poisson surface reconstruction with depth={depth}...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    
    return mesh

def create_mesh_from_point_cloud_ball_pivoting(point_cloud, radii=[0.05, 0.1, 0.2]):
    """
    Convert a point cloud into a triangle mesh using the Ball Pivoting algorithm.
    
    Args:
    - point_cloud (o3d.geometry.PointCloud): The point cloud to be converted to a mesh.
    - radii (list of floats): List of radii for Ball Pivoting (controls the level of detail).
    
    Returns:
    - mesh (o3d.geometry.TriangleMesh): The reconstructed triangle mesh.
    """
    
    # Step 1: Estimate normals (required for Ball Pivoting)
    print("Estimating normals for Ball Pivoting...")
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=50))
    
    # Step 2: Perform Ball Pivoting surface reconstruction
    print(f"Performing Ball Pivoting with radii={radii}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector(radii)
    )
    
    return mesh

# Example usage
# Load a point cloud from file (e.g., .ply or .xyz file)
point_cloud = o3d.io.read_point_cloud("RESULTS/cropped_point_cloud_with_rgb.ply")

# Generate a mesh using Poisson surface reconstruction
poisson_mesh = create_mesh_from_point_cloud_poisson(point_cloud, depth=10)


ball_pivoting_mesh = create_mesh_from_point_cloud_ball_pivoting(point_cloud, radii=[0.5, 1.0, 2.0])

# Visualize the resulting mesh
o3d.visualization.draw_geometries([poisson_mesh])
o3d.visualization.draw_geometries([ball_pivoting_mesh])

