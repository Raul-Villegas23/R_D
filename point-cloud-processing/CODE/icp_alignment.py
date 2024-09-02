import open3d as o3d
import numpy as np
import copy
import logging
from typing import Optional, Tuple
from scipy.spatial import KDTree

def refine_alignment_with_icp(
    source_mesh: o3d.geometry.TriangleMesh, 
    target_mesh: o3d.geometry.TriangleMesh, 
    threshold: float = 1.0, 
    max_iterations: int = 1000, 
    initial_transformation: Optional[np.ndarray] = None
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Refines the alignment of a source mesh to a target mesh using basic Iterative Closest Point (ICP) registration.
    
    Parameters:
    - source_mesh: The source mesh to be aligned.
    - target_mesh: The target mesh to align to.
    - threshold: The distance threshold for correspondences.
    - max_iterations: The maximum number of iterations for ICP.
    - initial_transformation: Initial transformation matrix for the ICP process.

    Returns:
    - source: The transformed source mesh after alignment.
    - final_transformation: The final transformation matrix used for alignment.
    """

    logging.info("Starting basic ICP registration...")

    # Step 1: Copy the source and target meshes to avoid modifying the originals
    source = copy.deepcopy(source_mesh)
    target = copy.deepcopy(target_mesh)

    # Step 2: Sample points from the meshes
    source_point_cloud = source.sample_points_uniformly(number_of_points=10000)
    target_point_cloud = target.sample_points_uniformly(number_of_points=10000)

    # Estimate normals
    source_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Visualize the point clouds
    # Paint the source point cloud
    source_point_cloud.paint_uniform_color([1, 0.706, 0])
    # Paint the target point cloud
    target_point_cloud.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_point_cloud, target_point_cloud])

    # Step 3: Set the initial transformation matrix
    if initial_transformation is None:
        initial_transformation = np.identity(4)

    # Step 4: Perform ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_point_cloud, target_point_cloud, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    final_transformation = reg_p2p.transformation

    # Step 5: Apply the final transformation to the source mesh
    source.transform(final_transformation)

    logging.info("ICP registration completed.")
    logging.info(f"Final Fitness: {reg_p2p.fitness:.4f}, Final RMSE: {reg_p2p.inlier_rmse:.4f}")

    return source, final_transformation

def refine_alignment_with_multipass_icp(
    source_mesh: o3d.geometry.TriangleMesh, 
    target_mesh: o3d.geometry.TriangleMesh, 
    threshold: float = 1.0, 
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

    def perform_icp(threshold: float) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float, float, int]:
        final_transformation = initial_transformation if initial_transformation is not None else np.identity(4)
        total_iterations = 0

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

                # Estimate used iterations based on fitness convergence
                if reg_p2p.fitness >= fitness_threshold:
                    used_iterations = int(max_iterations * reg_p2p.fitness)
                else:
                    used_iterations = max_iterations  # Assume max_iterations if not converged

                total_iterations += used_iterations
                logging.info(f"Pass {i} - Fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")
        else:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_point_cloud, target_point_cloud, threshold, final_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=convergence_threshold)
            )
            final_transformation = reg_p2p.transformation

            if reg_p2p.fitness >= fitness_threshold:
                used_iterations = int(max_iterations * reg_p2p.fitness)
            else:
                used_iterations = max_iterations  # Assume max_iterations if not converged

            total_iterations += used_iterations
            logging.info(f"Single Pass ICP - Fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")

        return source, final_transformation, reg_p2p.fitness, reg_p2p.inlier_rmse, total_iterations

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
    source, final_transformation, fitness, rmse, iterations = perform_icp(threshold)

    # Check if the fitness and RMSE meet the thresholds
    if fitness < fitness_threshold or rmse > rmse_threshold:
        logging.warning("Initial ICP did not meet the fitness or RMSE thresholds. Retrying with doubled threshold...")
        # Retry the ICP process with doubled threshold
        source, final_transformation, fitness, rmse, retry_iterations = perform_icp(threshold * 2)
        iterations += retry_iterations

        # Check if the thresholds are met after the second attempt
        if fitness < fitness_threshold or rmse > rmse_threshold:
            logging.warning("Second ICP attempt did not meet the thresholds. Returning the best result obtained.")

    # Step 3: Apply the final transformation
    source.transform(final_transformation)

    logging.info("ICP registration completed.")
    logging.info(f"Final Fitness: {fitness}, Final RMSE: {rmse}, Total Iterations: {iterations}")

    return source, final_transformation

def best_fit_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the least-squares best-fit transform that maps points A to points B.
    
    Returns:
    - R: rotation matrix (3x3)
    - t: translation vector (3x1)
    """
    assert A.shape == B.shape

    # Get number of dimensions
    m = A.shape[1]

    # Calculate centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Subtract centroids
    AA = A - centroid_A
    BB = B - centroid_B

    # Calculate covariance matrix
    H = AA.T @ BB

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Calculate rotation matrix
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = Vt.T @ U.T

    # Calculate translation vector
    t = centroid_B.T - R @ centroid_A.T

    return R, t

def icp(
    A: np.ndarray, 
    B: np.ndarray, 
    threshold: float = 2.0,
    max_iterations: int = 1000, 
    tolerance: float = 1e-4,
    initial_transformation: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative Closest Point (ICP) algorithm with thresholding.

    Args:
    - A: Source point cloud (N x 3).
    - B: Target point cloud (M x 3).
    - threshold: Distance threshold for nearest neighbor search.
    - max_iterations: Maximum number of iterations for ICP.
    - tolerance: Convergence tolerance.
    - initial_transformation: Initial transformation matrix.

    Returns:
    - A_transformed: Transformed source point cloud.
    - R: Final rotation matrix.
    - t: Final translation vector.
    """

    # Ensure A and B have the same number of points by downsampling the larger one
    if A.shape[0] > B.shape[0]:
        indices = np.random.choice(A.shape[0], B.shape[0], replace=False)
        A = A[indices]
    elif B.shape[0] > A.shape[0]:
        indices = np.random.choice(B.shape[0], A.shape[0], replace=False)
        B = B[indices]

    # Get number of dimensions
    m = A.shape[1]

    # Initialize the transformation matrix
    R = np.eye(m)
    t = np.zeros((m,))

    if initial_transformation is not None:
        R_init = initial_transformation[:m, :m]
        t_init = initial_transformation[:m, m]
        A = (R_init @ A.T).T + t_init

    prev_error = float('inf')

    for i in range(max_iterations):
        # Apply the transformation to A
        A_transformed = (R @ A.T).T + t

        # Find the nearest neighbors between the current source and target point clouds
        tree = KDTree(B)
        distances, indices = tree.query(A_transformed, distance_upper_bound=threshold)

        # Filter out points with no valid correspondences (those beyond the threshold)
        valid_mask = distances < threshold
        A_valid = A_transformed[valid_mask]
        B_valid = B[indices[valid_mask]]

        if len(A_valid) < 3:
            print("Too few valid points, stopping ICP.")
            break

        # Compute the transformation between the current source and nearest neighbors
        R_new, t_new = best_fit_transform(A_valid, B_valid)

        # Update the transformation matrix
        R = R_new @ R
        t = R_new @ t + t_new

        # Check for convergence
        mean_error = np.mean(distances[valid_mask])
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Apply the final transformation
    A_transformed = (R @ A.T).T + t

    return A_transformed, R, t