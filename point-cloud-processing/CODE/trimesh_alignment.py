import trimesh
import numpy as np
import logging
import trimesh.registration
from typing import Optional, Tuple
from trimesh.transformations import rotation_matrix
from trimesh.registration import icp


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Voxel downsample the input point cloud.

    Parameters:
    - points: The input point cloud (Nx3 array).
    - voxel_size: The size of the voxel grid.

    Returns:
    - downsampled_points: The downsampled point cloud.
    """
    # Quantize the point cloud based on the voxel size
    quantized = np.floor(points / voxel_size).astype(np.int32)

    # Find unique quantized points (effectively downsampling)
    unique_quantized, unique_indices = np.unique(quantized, axis=0, return_index=True)

    # Use the unique indices to get the downsampled points
    downsampled_points = points[unique_indices]

    return downsampled_points


def refine_alignment_with_icp_trimesh(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    threshold: float = 0.01,
    max_iterations: int = 1000,
    initial_transformation: Optional[np.ndarray] = None,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Refines the alignment of a source mesh to a target mesh using ICP, compares normal and 180-degree flipped GLB mesh.

    Parameters:
    - source_mesh: The source mesh to be aligned (trimesh.Trimesh).
    - target_mesh: The target mesh to align to (trimesh.Trimesh).
    - threshold: The distance threshold for ICP convergence.
    - max_iterations: The maximum number of iterations for ICP.
    - initial_transformation: Initial transformation matrix for the ICP process.

    Returns:
    - best_source_mesh: The transformed source mesh after the best alignment.
    - best_transformation: The final transformation matrix used for alignment.
    """

    logging.info("Starting basic ICP registration with Trimesh...")

    # Step 1: Copy the source and target meshes to avoid modifying the originals
    source_copy = source_mesh.copy()
    target_copy = target_mesh.copy()

    # Step 2: Sample points from the meshes using Trimesh's sample method
    source_points = source_copy.sample(10000)
    target_points = target_copy.sample(10000)

    # Step 3: Downsample points using voxel downsampling
    voxel_size = 0.05  # Define the voxel size
    source_points_downsampled = voxel_downsample(np.array(source_points), voxel_size)
    target_points_downsampled = voxel_downsample(np.array(target_points), voxel_size)

    # Step 4: Set the initial transformation matrix
    if initial_transformation is None:
        initial_transformation = np.eye(4)

    # Step 5: Perform ICP registration without flipping
    matrix_normal, aligned_source_points_normal, cost_normal = icp(
        source_points_downsampled,
        target_points_downsampled,
        initial=initial_transformation,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    logging.info(f"Normal ICP completed with cost: {cost_normal:.6f}")

    # Step 6: Apply 180-degree rotation around the Z-axis from the center of the mesh
    flip_180_z = rotation_matrix(np.radians(180), [0, 0, 1], source_copy.centroid)

    # Apply the rotation to the source mesh
    source_copy.apply_transform(flip_180_z)

    # Step 7: Perform ICP registration with the flipped mesh
    source_points_flipped = source_copy.sample(10000)  # Re-sample after transformation
    source_points_flipped = np.array(source_points_flipped)  # Convert to ndarray
    source_points_flipped_downsampled = voxel_downsample(
        source_points_flipped, voxel_size
    )

    matrix_flipped, aligned_source_points_flipped, cost_flipped = icp(
        source_points_flipped_downsampled,
        target_points_downsampled,
        initial=initial_transformation,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    logging.info(f"Flipped ICP completed with cost: {cost_flipped:.6f}")

    # Step 8: Compare the ICP costs and choose the better transformation
    if cost_flipped < cost_normal:
        # Use flipped transformation
        best_transformation = matrix_flipped @ flip_180_z
        best_source_mesh = source_copy.copy()  # Flipped and transformed mesh
        logging.info(f"Flipped transformation chosen with cost: {cost_flipped:.6f}")
    else:
        # Use normal transformation
        best_transformation = matrix_normal
        best_source_mesh = source_mesh.copy()
        best_source_mesh.apply_transform(best_transformation)
        logging.info(f"Normal transformation chosen with cost: {cost_normal:.6f}")

    return best_source_mesh, best_transformation
