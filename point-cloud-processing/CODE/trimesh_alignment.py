import trimesh
import numpy as np
import logging
import trimesh.registration
from typing import Optional, Tuple

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
    threshold: float =  1.0, # 1e-6 = 0.000001
    max_iterations: int = 10000000, 
    initial_transformation: Optional[np.ndarray] = None
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Refines the alignment of a source mesh to a target mesh using basic Iterative Closest Point (ICP) registration.
    
    Parameters:
    - source_mesh: The source mesh to be aligned (trimesh.Trimesh).
    - target_mesh: The target mesh to align to (trimesh.Trimesh).
    - threshold: The distance threshold for ICP convergence.
    - max_iterations: The maximum number of iterations for ICP.
    - initial_transformation: Initial transformation matrix for the ICP process.
    
    Returns:
    - source_mesh: The transformed source mesh after alignment.
    - final_transformation: The final transformation matrix used for alignment.
    """

    logging.info("Starting basic ICP registration with Trimesh...")

    # Step 1: Copy the source and target meshes to avoid modifying the originals
    source_copy = source_mesh.copy()
    target_copy = target_mesh.copy()

    # Step 2: Sample points from the meshes using Trimesh's sample method
    source_points = source_copy.sample(10000)  # Sampling 10,000 points from the source mesh
    target_points = target_copy.sample(10000)  # Sampling 10,000 points from the target mesh

    # Step 3: Downsample points using voxel downsampling
    voxel_size = 0.05  # Define the voxel size
    source_points_downsampled = voxel_downsample(source_points, voxel_size)
    target_points_downsampled = voxel_downsample(target_points, voxel_size)

    # Step 4: Set the initial transformation matrix
    if initial_transformation is None:
        initial_transformation = np.eye(4)

    # Step 5: Perform ICP registration using Trimesh's ICP implementation
    matrix, aligned_source_points, cost = trimesh.registration.icp(
        source_points_downsampled, target_points_downsampled, 
        initial=initial_transformation, 
        threshold=threshold, max_iterations=max_iterations
    )

    logging.info(f"ICP registration completed with cost: {cost:.6f}")
    logging.info(f"Final ICP transformation matrix:\n{matrix}")

    # Step 6: Apply the final transformation to the source mesh
    source_copy.apply_transform(matrix)

    return source_copy, matrix
