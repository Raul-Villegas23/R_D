# Libraries for API requests and data processing
import logging
import time

# Libraries for point cloud processing
import numpy as np
import open3d as o3d

# Import the necessary functions from the custom modules
from fetcher import process_feature_list, print_memory_usage
from geolocation import extract_latlon_orientation_from_mesh
from mesh_processor import load_and_transform_glb_model
from transformation_utils import accumulate_transformations, compute_z_offset, apply_z_offset
from visualization_utils import visualize_meshes_with_height_coloring
from icp_alignment import refine_alignment_with_icp

def process_glb_and_bag(feature_ids, glb_dataset, collections_url, collection_id):
    # Process the feature list with the 3D BAG API and combine the meshes into a single mesh using the PAND collection ID
    bag_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if bag_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        glb_model_path = data_folder + glb_dataset

        # Load and transform the GLB model using the translation from the BAG data and apply the necessary coordinate system transformations
        glb_mesh, initial_transformation = load_and_transform_glb_model(glb_model_path, translate)

        if glb_mesh:

            # Apply the initial transformation to the list of transformations to get the final transformation matrix later
            transformations = [initial_transformation]

            glb_mesh, icp_transformation = refine_alignment_with_icp(glb_mesh, bag_mesh)
            transformations.append(icp_transformation)

            # Compute the z-offset between the BAG and GLB meshes and apply it to the GLB mesh
            z_offset = compute_z_offset(bag_mesh, glb_mesh)
            apply_z_offset(glb_mesh, z_offset)
            transformations.append(np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, z_offset],
                [0, 0, 0, 1]
            ]))

            # Extract the latitude, longitude, and orientation from the GLB mesh
            lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
            logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")

            # Accumulate the transformations to get the final transformation matrix
            final_transformation_matrix = accumulate_transformations(transformations)
            logging.info(f"Final transformation matrix:\n{final_transformation_matrix}")

            # Save the transformation matrix, latitude, longitude, and orientation to files
            transformation_matrix_filename = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
            np.savetxt(transformation_matrix_filename, final_transformation_matrix)
            with open(f"RESULTS/{glb_dataset.split('.')[0]}_lat_lon_orientation.txt", "w") as file:
                file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")

            # Visualize and save results
            # visualize_meshes_with_height_coloring(bag_mesh, glb_mesh, colormap_1="YlOrRd", colormap_2="YlGnBu_r")
            # bag_mesh += glb_mesh
            # combined_ply_filename = glb_dataset.replace('.glb', '_bag_glb.ply')
            # o3d.io.write_triangle_mesh(f"RESULTS/{combined_ply_filename}", bag_mesh, print_progress=True)

def main():
    start_time = time.time()
    print_memory_usage("start")  # Initial memory usage

    # Define the API URL and the collection ID
    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'

    # List of feature IDs and corresponding GLB datasets
    tasks = [
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

    # Process each GLB dataset and its corresponding BAG feature IDs
    for task in tasks:
        feature_ids = task["feature_ids"]
        glb_dataset = task["glb_dataset"]
        logging.info(f"Processing GLB dataset {glb_dataset}")
        process_glb_and_bag(feature_ids, glb_dataset, collections_url, collection_id)
        print_memory_usage(f"after processing {glb_dataset}")
        print("\n")

    logging.info(f"Total elapsed time: {time.time() - start_time:.3f} seconds")
    print_memory_usage("end")

if __name__ == "__main__":
    main()
