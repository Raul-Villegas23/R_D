import open3d as o3d
import copy
import numpy as np

def refine_alignment_with_icp(source_mesh, target_mesh, threshold=1.0, max_iterations=500):
    """Refine alignment between source and target meshes using ICP."""
    source = copy.deepcopy(source_mesh)
    target = copy.deepcopy(target_mesh)
    
    # Convert meshes to point clouds for ICP
    source_point_cloud = source.sample_points_uniformly(number_of_points=10000)
    target_point_cloud = target.sample_points_uniformly(number_of_points=10000)

    # Apply ICP directly without an initial transformation
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_point_cloud, target_point_cloud, threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    # Apply the transformation to the source mesh
    source.transform(reg_p2p.transformation)

    return source, reg_p2p.transformation

def get_transformation_matrix(initial_mesh, final_mesh, threshold=1.0, max_iterations=500):
    """
    Calculate the transformation matrix that aligns the initial mesh with the final aligned mesh.
    
    Parameters:
    initial_mesh (o3d.geometry.TriangleMesh): The initial mesh before transformation.
    final_mesh (o3d.geometry.TriangleMesh): The final aligned mesh after transformation.
    threshold (float): Maximum distance threshold between corresponding points.
    max_iterations (int): Maximum number of ICP iterations.

    Returns:
    np.ndarray: Transformation matrix (4x4) that aligns the initial mesh with the final mesh.
    """
    # Convert meshes to point clouds for alignment comparison
    initial_point_cloud = initial_mesh.sample_points_uniformly(number_of_points=10000)
    final_point_cloud = final_mesh.sample_points_uniformly(number_of_points=10000)

    # Perform ICP registration to get the transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        initial_point_cloud, final_point_cloud, threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    return reg_p2p.transformation
