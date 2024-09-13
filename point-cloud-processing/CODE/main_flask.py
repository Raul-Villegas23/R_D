# main.py

import logging
import time
import gc
from typing import List, Dict
import numpy as np

from trimesh_fetcher import process_feature_list
from trimesh_processor import load_and_transform_glb_model_trimesh, align_trimesh_centers, apply_optimal_params_trimesh
from trimesh_alignment import refine_alignment_with_icp_trimesh
from trimesh_transformations_utils import compute_z_offset, apply_z_offset
from transformation_utils import accumulate_transformations, calculate_rotation_z
from geolocation import extract_latlon_orientation_from_mesh
from geometry_utils import extract_2d_perimeter, optimize_rotation_and_translation
from trimesh_visualization import visualize_trimesh_objects

def process_glb_and_bag(
    feature_ids: List[str],
    glb_file_path: str,
    collections_url: str,
    collection_id: str
) -> Dict[str, float]:
    """Process a single GLB and BAG feature IDs."""
    bag_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if bag_mesh and scale is not None and translate is not None and reference_system is not None:
        # Load the GLB model and apply transformations
        glb_mesh, initial_transformation, origin = load_and_transform_glb_model_trimesh(glb_file_path, translate)

        if glb_mesh:
            if initial_transformation is not None:
                transformations: List[np.ndarray] = [initial_transformation]
            else:
                transformations: List[np.ndarray] = []

            # Align mesh centers of the BAG and GLB meshes
            glb_mesh, center_alignment_translation = align_trimesh_centers(bag_mesh, glb_mesh)
            transformations.append(center_alignment_translation)

            # Optimize rotation and translation between the BAG and GLB meshes
            perimeter1 = extract_2d_perimeter(bag_mesh)
            perimeter2 = extract_2d_perimeter(glb_mesh)

            optimal_params = optimize_rotation_and_translation(perimeter1, perimeter2)

            # Ensure optimal_params is valid
            if optimal_params is not None and len(optimal_params) > 0:
                optimal_angle, optimal_tx, optimal_ty = optimal_params
                glb_mesh, t = apply_optimal_params_trimesh(glb_mesh, optimal_angle, optimal_tx, optimal_ty)
                transformations.append(t)


            # Refine alignment with ICP
            glb_mesh, icp_transformation = refine_alignment_with_icp_trimesh(glb_mesh, bag_mesh)
            transformations.append(icp_transformation)

            # Fix the height offset
            z_offset = compute_z_offset(bag_mesh, glb_mesh)
            apply_z_offset(glb_mesh, z_offset)
            transformations.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z_offset], [0, 0, 0, 1]], dtype=np.float64))

            final_transformation_matrix = accumulate_transformations(transformations)
            rotation = calculate_rotation_z(final_transformation_matrix)
            print(f"Rotation: {rotation}")
            # Extract latitude, longitude, and orientation
            lon, lat = extract_latlon_orientation_from_mesh(final_transformation_matrix, reference_system, origin)
            print(f"Latitude: {lat}, Longitude: {lon}")

            # Visualize the results
            # visualize_trimesh_objects(bag_mesh, glb_mesh)

            # Save the aligned GLB mesh to a file inside RESULTS folder
            glb_mesh.export(f"RESULTS/{glb_file_path.split('/')[-1].split('.')[0]}_aligned.ply")

            # Return the result as a dictionary
            return {
                "latitude": lat,
                "longitude": lon,
                "rotation": rotation
            }
    return {}
