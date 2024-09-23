import trimesh
import numpy as np
import logging
from trimesh.transformations import rotation_matrix
from trimesh.registration import icp
from typing import Optional, Tuple


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Voxel downsample the input point cloud by reducing points into voxel grid cells.
    This reduces the number of points by keeping only one point per voxel.
    """
    quantized = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(quantized, axis=0, return_index=True)
    return points[unique_indices]


def sample_and_downsample(
    mesh: trimesh.Trimesh, num_samples: int, voxel_size: float
) -> np.ndarray:
    """
    Sample points from a mesh and downsample the point cloud using voxel downsampling.
    """
    points = mesh.sample(num_samples)
    return voxel_downsample(np.array(points), voxel_size)


def hierarchical_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    voxel_sizes: list,
    threshold: float,
    max_iterations: int,
    initial_transformation: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Perform hierarchical (multi-resolution) ICP, starting with a coarse alignment and refining it at finer resolutions.

    Parameters:
    - source_points: Source point cloud to align.
    - target_points: Target point cloud to align to.
    - voxel_sizes: List of voxel sizes to use for hierarchical ICP (coarse to fine).
    - threshold: Distance threshold for ICP convergence.
    - max_iterations: Maximum number of iterations for each ICP stage.
    - initial_transformation: Initial transformation matrix for ICP.

    Returns:
    - best_matrix: The final transformation matrix after alignment.
    - best_cost: The cost of the final alignment.
    """
    current_transformation = initial_transformation

    # Downsample source and target once using the smallest voxel size to avoid redundant calculations
    smallest_voxel_size = min(voxel_sizes)
    source_downsampled = voxel_downsample(source_points, smallest_voxel_size)
    target_downsampled = voxel_downsample(target_points, smallest_voxel_size)

    for voxel_size in voxel_sizes:
        if voxel_size != smallest_voxel_size:
            # Further downsample the pre-downsampled points
            source_voxel_down = voxel_downsample(source_downsampled, voxel_size)
            target_voxel_down = voxel_downsample(target_downsampled, voxel_size)
        else:
            # Use the smallest downsampled points directly
            source_voxel_down = source_downsampled
            target_voxel_down = target_downsampled

        # Perform ICP
        # logging.info(
        #     f"Running ICP with voxel size: {voxel_size}, {len(source_voxel_down)} points"
        # )
        matrix, _, cost = icp(
            source_voxel_down,
            target_voxel_down,
            initial=current_transformation,
            threshold=threshold,
            max_iterations=max_iterations,
        )
        current_transformation = matrix  # Update transformation with the result

    return current_transformation, cost


def refine_alignment_with_icp_trimesh(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    threshold: float = 0.001,
    max_iterations: int = 1000,
    initial_transformation: Optional[np.ndarray] = None,
    voxel_sizes: list = [0.1, 0.05, 0.02],  # Coarse to fine resolution
    num_samples: int = 10000,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Refines the alignment of a source mesh to a target mesh using hierarchical ICP (multi-resolution),
    and compares normal and 180-degree flipped versions of the source mesh.

    Parameters:
    - source_mesh: The source mesh to be aligned.
    - target_mesh: The target mesh to align to.
    - threshold: Distance threshold for ICP convergence.
    - max_iterations: Maximum number of ICP iterations.
    - initial_transformation: Initial transformation matrix for ICP.
    - voxel_sizes: List of voxel sizes for hierarchical ICP.
    - num_samples: Number of points to sample from the meshes.

    Returns:
    - best_source_mesh: The transformed source mesh after the best alignment.
    - best_transformation: The transformation matrix that gave the best alignment.
    """
    logging.info("Starting ICP registration...")

    # Sample points once from both meshes
    source_points = np.array(source_mesh.sample(num_samples))
    target_points = np.array(target_mesh.sample(num_samples))

    # Initialize transformation matrix
    if initial_transformation is None:
        initial_transformation = np.eye(4)

    # Perform hierarchical ICP on normal orientation
    # logging.info("Performing ICP for normal orientation...")
    matrix_normal, cost_normal = hierarchical_icp(
        source_points,
        target_points,
        voxel_sizes=voxel_sizes,
        threshold=threshold,
        max_iterations=max_iterations,
        initial_transformation=initial_transformation,
    )
    # logging.info(f"Normal ICP completed with cost: {cost_normal:.6f}")

    # Apply 180-degree flip around Z-axis (apply the transformation directly to the points)
    # logging.info("Performing ICP for flipped orientation...")
    flip_180_z = rotation_matrix(np.radians(180), [0, 0, 1], source_mesh.centroid)
    source_points_flipped = trimesh.transform_points(source_points, flip_180_z)

    # Perform hierarchical ICP on flipped orientation
    matrix_flipped, cost_flipped = hierarchical_icp(
        source_points_flipped,
        target_points,
        voxel_sizes=voxel_sizes,
        threshold=threshold,
        max_iterations=max_iterations,
        initial_transformation=initial_transformation,
    )
    # logging.info(f"Flipped ICP completed with cost: {cost_flipped:.6f}")

    # Choose the best transformation based on cost
    if cost_flipped < cost_normal:
        best_transformation = matrix_flipped @ flip_180_z
        best_source_mesh = source_mesh.copy()
        best_source_mesh.apply_transform(best_transformation)
        logging.info(f"Flipped transformation chosen with cost: {cost_flipped:.6f}")
    else:
        best_transformation = matrix_normal
        best_source_mesh = source_mesh.copy()
        best_source_mesh.apply_transform(best_transformation)
        logging.info(f"Normal transformation chosen with cost: {cost_normal:.6f}")

    return best_source_mesh, best_transformation
