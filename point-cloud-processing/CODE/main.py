# Libraries for API requests and data processing
import logging
import time
import gc
from typing import List, Tuple, Optional, Dict

# Libraries for point cloud processing
import numpy as np
import open3d as o3d

# Import the necessary functions from the custom modules
from fetcher import process_feature_list, print_memory_usage
from geolocation import extract_latlon_orientation_from_mesh
from mesh_processor import load_and_transform_glb_model, align_mesh_centers
from transformation_utils import accumulate_transformations, compute_z_offset, apply_z_offset
from visualization_utils import visualize_meshes_with_height_coloring
from icp_alignment import refine_alignment_with_icp, refine_alignment_with_multipass_icp
# Typing import
from numpy.typing import NDArray

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

        # Load the glb model, translate it close to the 3dbag model and transform it to the same coordinate system
        glb_mesh, initial_transformation = load_and_transform_glb_model(glb_model_path, translate)

        if glb_mesh:
            # Initialize the list of transformations with the initial transformation matrix from the GLB model
            transformations: List[NDArray[np.float64]] = [initial_transformation]

            # Use ICP to refine the alignment between the BAG and GLB meshes
            glb_mesh, icp_transformation = refine_alignment_with_icp(glb_mesh, bag_mesh)

            # Append the ICP transformation to the list of transformations
            transformations.append(icp_transformation)

            # Fix the height offset between the BAG and GLB meshes
            z_offset = compute_z_offset(bag_mesh, glb_mesh)
            apply_z_offset(glb_mesh, z_offset)

            # Append the z-offset transformation to the list of transformations
            transformations.append(np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, z_offset],
                [0, 0, 0, 1]
            ], dtype=np.float64))

            # Extract the latitude, longitude, and orientation from the transformed GLB mesh
            lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
            logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")
            with open(f"RESULTS/{glb_dataset.split('.')[0]}_lat_lon_orientation.txt", "w") as file:
                file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")

            # Accumulate the transformations and save the final transformation matrix to a text file
            final_transformation_matrix = accumulate_transformations(transformations)
            logging.info(f"\nFinal transformation matrix:\n{final_transformation_matrix}")

            transformation_matrix_filename = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
            np.savetxt(transformation_matrix_filename, final_transformation_matrix)

            # Visualize the BAG and GLB meshes with height coloring for debugging purposes
            visualize_meshes_with_height_coloring(bag_mesh, glb_mesh, colormap_1='YlOrRd', colormap_2='YlGnBu')

            del bag_mesh, glb_mesh, transformations  # Delete the meshes and transformations to free up memory
            gc.collect()  # Explicitly call garbage collection

def main() -> None:
    start_time = time.time()
    print_memory_usage("start")

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'

    tasks: List[Dict[str, List[str]]] = [
        {
            "feature_ids": ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"],
            "glb_dataset": "pijlkruidstraat11-13-15.glb"
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

    # Process each task in the list with the corresponding feature IDs and GLB dataset
    for task in tasks:
        feature_ids = task["feature_ids"]
        glb_dataset = task["glb_dataset"]
        logging.info(f"Processing GLB dataset {glb_dataset}")
        process_glb_and_bag(feature_ids, glb_dataset, collections_url, collection_id)
        print_memory_usage(f"after processing {glb_dataset}")
        print("\n")

    logging.info(f"Total elapsed time: {time.time() - start_time:.3f} seconds")
    print_memory_usage("end")

    gc.collect()

if __name__ == "__main__":
    main()
