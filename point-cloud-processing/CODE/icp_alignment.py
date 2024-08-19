import open3d as o3d
import copy
import numpy as np

def refine_alignment_with_icp(source_mesh, target_mesh, threshold=1.0, initial_max_iterations=50, max_iterations_limit=5000, convergence_threshold=1e-4):

    source = copy.deepcopy(source_mesh)
    target = copy.deepcopy(target_mesh)

    # Convert meshes to point clouds for ICP
    source_point_cloud = source.sample_points_uniformly(number_of_points=10000)
    target_point_cloud = target.sample_points_uniformly(number_of_points=10000)

    # Initialize variables for dynamic iteration control
    last_transformation = np.identity(4)
    max_iterations = initial_max_iterations

    while max_iterations <= max_iterations_limit:
        # Apply ICP with current max_iterations
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_point_cloud, target_point_cloud, threshold,
            last_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )

        # Compute change in transformation matrix
        transformation_change = np.linalg.norm(reg_p2p.transformation - last_transformation)

        # Check for convergence
        if transformation_change < convergence_threshold:
            # print(f"ICP converged with {max_iterations} iterations, change: {transformation_change:.6f}")
            source.transform(reg_p2p.transformation)
            # print(reg_p2p)
            return source, reg_p2p.transformation

        # Update last transformation and increase max_iterations
        last_transformation = reg_p2p.transformation
        max_iterations += 50  # Increment iterations for next attempt

    print(f"ICP reached maximum iteration limit of {max_iterations_limit} without full convergence.")
    source.transform(reg_p2p.transformation)
    return source, reg_p2p.transformation
