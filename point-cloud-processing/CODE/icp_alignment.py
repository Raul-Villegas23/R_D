import open3d as o3d
import numpy as np
import copy
import logging

def preprocess_point_cloud(mesh, voxel_size):
    """
    Converts a mesh to a point cloud, then downsamples, estimates normals, and computes FPFH features.
    
    Parameters:
    - mesh: The input TriangleMesh.
    - voxel_size: The voxel size for downsampling.
    
    Returns:
    - downsampled_pcd: The downsampled point cloud.
    - fpfh: The FPFH features of the downsampled point cloud.
    """
    print(":: Converting mesh to point cloud.")
    pcd = mesh.sample_points_uniformly(number_of_points=100000)

    print(":: Downsampling with a voxel size of %.3f." % voxel_size)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    print(":: Estimating normals.")
    downsampled_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))

    print(":: Computing FPFH features.")
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))
    
    return downsampled_pcd, fpfh


def refine_alignment_with_icp(source_mesh, target_mesh, threshold=2.0, max_iterations=1000, convergence_threshold=1e-4, sample_points=10000, initial_transformation=None, multiple_passes=True):
    logging.info("Starting ICP registration...")

    source = copy.deepcopy(source_mesh)
    target = copy.deepcopy(target_mesh)

    logging.info(f"Sampling {sample_points} points from source and target meshes.")
    source_point_cloud = source.sample_points_uniformly(number_of_points=sample_points)
    target_point_cloud = target.sample_points_uniformly(number_of_points=sample_points)

    source_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([source_point_cloud, target_point_cloud])

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