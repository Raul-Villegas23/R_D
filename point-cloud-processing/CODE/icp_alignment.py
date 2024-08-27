import open3d as o3d
import numpy as np
import copy
import logging

def refine_alignment_with_icp(source_mesh, target_mesh, threshold=2.0, max_iterations=1000, convergence_threshold=1e-4, sample_points=10000, initial_transformation=None, multiple_passes=True):
    """
    Refines the alignment of a source mesh to a target mesh using Iterative Closest Point (ICP) registration.
    Parameters:
    - source_mesh: The source mesh to be aligned.
    - target_mesh: The target mesh to align to.
    - threshold: The distance threshold for correspondences.
    - max_iterations: The maximum number of iterations for ICP.
    - convergence_threshold: The convergence threshold for ICP.
    - sample_points: The number of points to sample from the meshes
    - initial_transformation: Initial transformation matrix for the ICP process.
    - multiple_passes: Whether to perform multiple ICP passes with decreasing thresholds.

    Returns:
    - source: The transformed source mesh after alignment.
    - final_transformation: The final transformation matrix used for alignment.
    """

    logging.info("Starting ICP registration...")
    

    source = copy.deepcopy(source_mesh)
    target = copy.deepcopy(target_mesh)

    logging.info(f"Sampling {sample_points} points from source and target meshes.")
    source_point_cloud = source.sample_points_uniformly(number_of_points=sample_points)
    target_point_cloud = target.sample_points_uniformly(number_of_points=sample_points)

    source_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if initial_transformation is None:
        initial_transformation = np.identity(4)

    final_transformation = initial_transformation

    if multiple_passes:
        for i, factor in enumerate([1.0, 0.5, 0.25], start=1):
            current_threshold = threshold * factor
            logging.info(f"ICP Pass {i} with threshold {current_threshold:.4f}")
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_point_cloud, target_point_cloud, current_threshold, final_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=convergence_threshold)
            )
            final_transformation = reg_p2p.transformation
            logging.info(f"Pass {i} - Fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_point_cloud, target_point_cloud, threshold, final_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=convergence_threshold)
        )
        final_transformation = reg_p2p.transformation
        logging.info(f"Single Pass ICP - Fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")

    source.transform(final_transformation)

    logging.info("ICP registration completed.")

    return source, final_transformation