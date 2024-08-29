import open3d as o3d
import numpy as np
import copy
import logging
from typing import Optional, Tuple

def refine_alignment_with_icp(
    source_mesh: o3d.geometry.TriangleMesh, 
    target_mesh: o3d.geometry.TriangleMesh, 
    threshold: float = 2.0, 
    max_iterations: int = 1000, 
    convergence_threshold: float = 1e-4, 
    sample_points: int = 10000, 
    initial_transformation: Optional[np.ndarray] = None, 
    multiple_passes: bool = True,
    fitness_threshold: float = 0.6,
    rmse_threshold: float = 0.3
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Refines the alignment of a source mesh to a target mesh using Iterative Closest Point (ICP) registration.
    
    Parameters:
    - source_mesh: The source mesh to be aligned.
    - target_mesh: The target mesh to align to.
    - threshold: The distance threshold for correspondences.
    - max_iterations: The maximum number of iterations for ICP.
    - convergence_threshold: The convergence threshold for ICP.
    - sample_points: The number of points to sample from the meshes.
    - initial_transformation: Initial transformation matrix for the ICP process.
    - multiple_passes: Whether to perform multiple ICP passes with decreasing thresholds.
    - fitness_threshold: The minimum acceptable fitness value.
    - rmse_threshold: The maximum acceptable RMSE value.

    Returns:
    - source: The transformed source mesh after alignment.
    - final_transformation: The final transformation matrix used for alignment.
    """

    def perform_icp(threshold: float) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float, float]:
        final_transformation = initial_transformation if initial_transformation is not None else np.identity(4)

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

        return source, final_transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

    logging.info("Starting ICP registration...")
    
    # Step 1: Copy the source and target meshes to avoid modifying the originals
    source = copy.deepcopy(source_mesh)
    target = copy.deepcopy(target_mesh)

    # Step 2: Sample points from the meshes
    logging.info(f"Sampling {sample_points} points from source and target meshes.")
    source_point_cloud = source.sample_points_uniformly(number_of_points=sample_points)
    target_point_cloud = target.sample_points_uniformly(number_of_points=sample_points)

    source_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Perform ICP registration with the initial threshold
    source, final_transformation, fitness, rmse = perform_icp(threshold)

    # Check if the fitness and RMSE meet the thresholds
    if fitness < fitness_threshold or rmse > rmse_threshold:
        logging.warning("Initial ICP did not meet the fitness or RMSE thresholds. Retrying with doubled threshold...")
        # Retry the ICP process with doubled threshold
        source, final_transformation, fitness, rmse = perform_icp(threshold * 2)

        # Check if the thresholds are met after the second attempt
        if fitness < fitness_threshold or rmse > rmse_threshold:
            logging.warning("Second ICP attempt did not meet the thresholds. Returning the best result obtained.")

    # Step 3: Apply the final transformation
    source.transform(final_transformation)

    logging.info("ICP registration completed.")
    logging.info(f"Final Fitness: {fitness}, Final RMSE: {rmse}")

    return source, final_transformation
