import open3d as o3d
import numpy as np
import copy
import logging

def refine_alignment_with_icp(source_mesh, target_mesh, threshold=2.0, initial_max_iterations=50, max_iterations_limit=5000, convergence_threshold=1e-4, sample_points=10000):
    """
    Refines the alignment of a source mesh to a target mesh using Iterative Closest Point (ICP) registration.

    Parameters:
    - source_mesh: The source mesh to be aligned.
    - target_mesh: The target mesh to align to.
    - threshold: Distance threshold to consider for ICP.
    - initial_max_iterations: The starting number of maximum iterations for ICP.
    - max_iterations_limit: The maximum limit of iterations for ICP.
    - convergence_threshold: The threshold for determining convergence based on transformation change.
    - sample_points: Number of points to sample from the mesh for ICP.

    Returns:
    - aligned_source: The transformed source mesh after alignment.
    - final_transformation: The final transformation matrix used for alignment.
    """
    logging.info("Starting ICP refinement...")

    # Copy meshes to avoid modifying originals
    source = copy.deepcopy(source_mesh)
    target = copy.deepcopy(target_mesh)

    # Convert meshes to point clouds for ICP registration
    logging.info(f"Sampling {sample_points} points from source and target meshes.")
    source_point_cloud = source.sample_points_uniformly(number_of_points=sample_points)
    target_point_cloud = target.sample_points_uniformly(number_of_points=sample_points)

    last_transformation = np.identity(4)

    # Perform ICP registration with dynamic iteration control
    for max_iterations in range(initial_max_iterations, max_iterations_limit + 1, 50):
        logging.info(f"Performing ICP with max_iterations={max_iterations}...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_point_cloud, target_point_cloud, threshold,
            last_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )

        transformation_change = np.linalg.norm(reg_p2p.transformation - last_transformation)
        logging.info(f"Transformation change: {transformation_change:.6f}")

        if transformation_change < convergence_threshold:
            logging.info(f"ICP converged with {max_iterations} iterations, final transformation change: {transformation_change:.6f}")
            source.transform(reg_p2p.transformation)
            logging.info(f"Final registration fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")
            return source, reg_p2p.transformation

        last_transformation = reg_p2p.transformation

    logging.warning(f"ICP reached the maximum iteration limit of {max_iterations_limit} without full convergence.")
    source.transform(reg_p2p.transformation)

    logging.info(f"Final registration fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")
    return source, reg_p2p.transformation