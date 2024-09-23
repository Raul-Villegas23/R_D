# Libraries for API requests and data processing
import logging
import time
import gc
from typing import List, Tuple, Optional, Dict

# Libraries for point cloud processing
import numpy as np

# Import the necessary functions from the custom modules
from trimesh_fetcher import process_feature_list, print_memory_usage
from geolocation import extract_latlon_orientation_from_mesh
from trimesh_transformations_utils import compute_z_offset, apply_z_offset
from trimesh_visualization import color_transformed_origin_vertex, visualize_trimesh_objects
from trimesh_alignment import refine_alignment_with_icp_trimesh
from geometry_utils import extract_2d_perimeter, optimize_rotation_and_translation
from trimesh_processor import load_and_transform_glb_model_trimesh, align_trimesh_centers, apply_optimal_params_trimesh
from trimesh_transformations_utils import accumulate_transformations, calculate_rotation_z

def process_glb_and_bag(
    feature_ids: List[str],
    glb_dataset: str,
    collections_url: str,
    collection_id: str
) -> None:
    """Process a single GLB and BAG feature IDs.

    Args:
        feature_ids (List[str]): A list of feature IDs from the BAG dataset.
        glb_dataset (str): The filename of the GLB dataset.
        collections_url (str): The base URL for the API collections.
        collection_id (str): The ID of the collection being processed.
    """
    bag_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if bag_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        glb_model_path = data_folder + glb_dataset

        # Load the GLB model and apply transformations
        glb_mesh, initial_transformation, origin = load_and_transform_glb_model_trimesh(glb_model_path, translate)
        
        # colored_mesh = color_transformed_origin_vertex(glb_mesh, np.eye(4))
        # # Export the modified mesh with the name of the GLB dataset
        # colored_mesh.export(f"RESULTS/{glb_dataset.split('.')[0]}_colored.ply")


        if glb_mesh:
            # Initialize the list of transformations with the initial transformation matrix from the GLB model
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

            # Use ICP with trimesh to refine the alignment
            glb_mesh, icp_transformation = refine_alignment_with_icp_trimesh(glb_mesh, bag_mesh)

            # Append the ICP transformation to the list of transformations
            transformations.append(icp_transformation)
            

            # Fix the height offset between the BAG and GLB meshes
            z_offset = compute_z_offset(bag_mesh, glb_mesh)
            apply_z_offset(glb_mesh, z_offset)

            transformations.append(np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, z_offset],
                [0, 0, 0, 1]
            ], dtype=np.float64))

            # Accumulate and save the final transformation matrix
            final_transformation_matrix = accumulate_transformations(transformations)
            logging.info(f"\nFinal transformation matrix:\n{final_transformation_matrix}")
            
            # Calculate the rotation around the Z-axis to align the GLB model with the BAG model
            rotation = calculate_rotation_z(final_transformation_matrix)
            logging.info(f"Optimal rotation angle around Z-axis: {rotation:.5f} radians")

            # Extract latitude, longitude, and orientation
            lon, lat = extract_latlon_orientation_from_mesh(final_transformation_matrix, reference_system, origin)
            logging.info(f"Latitude: {lat}, Longitude: {lon}, Rotation: {rotation} radians")

            # Save the aligned GLB mesh to a file inside RESULTS folder
            colored_mesh = color_transformed_origin_vertex(glb_mesh, final_transformation_matrix)
            # Export the modified mesh with the name of the GLB dataset
            colored_mesh.export(f"RESULTS/{glb_dataset.split('.')[0]}_colored.ply")

            # Save the bag mesh to a file inside RESULTS folder with the colored origin vertex
            colored_bag_mesh = color_transformed_origin_vertex(bag_mesh, np.eye(4))
            colored_bag_mesh.export(f"RESULTS/{glb_dataset.split('.')[0]}_bag_colored.ply")

            # Write final transformation matrix to file
            np.savetxt(f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt", final_transformation_matrix)

            # Visualize with matplotlib
            bag_perimeter = extract_2d_perimeter(bag_mesh)
            glb_perimeter = extract_2d_perimeter(glb_mesh)
            # visualize_2d_perimeters(bag_perimeter, glb_perimeter)

            visualize_trimesh_objects(bag_mesh, glb_mesh)

            # Free memory
            del bag_mesh, glb_mesh, transformations
            gc.collect()

def main() -> None:
    start_time = time.time()
    print_memory_usage("start")

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'

    tasks: List[Dict[str, List[str]]] = [
        {
            "feature_ids": ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"],
            "glb_dataset": "pijlkruid.glb"
            # "glb_dataset": "pijlkruidstraat11-13-15.glb"
        },
        {
            "feature_ids": ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"],
            "glb_dataset": "pijlkruid37-37.glb"
        },
        {
            "feature_ids": ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"],
            "glb_dataset": "rietstraat31-33.glb"
        }
    ]

    for task in tasks:
        feature_ids = task["feature_ids"]
        glb_dataset = task["glb_dataset"]
        logging.info(f"Processing GLB dataset {glb_dataset}")
        process_glb_and_bag(feature_ids, glb_dataset, collections_url, collection_id)
        print_memory_usage(f"after processing {glb_dataset}")
        print("\n")
        # Print the time per task
        logging.info(f"Time for processing {glb_dataset}: {time.time() - start_time:.3f} seconds")

    logging.info(f"Total elapsed time: {time.time() - start_time:.3f} seconds")
    print_memory_usage("end")

    gc.collect()

if __name__ == "__main__":
    main()
