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
