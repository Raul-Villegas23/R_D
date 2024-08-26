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


def main():
    start_time = time.time()
    print_memory_usage("start")  # Initial memory usage

    # Define the API URL and the collection ID and feature IDs to process the 3DBAG data
    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    # feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  # pijlkruidstraat11-13-15.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb

    # Process the feature list with the 3D BAG API and combine the meshes into a single meshusing the PAND collection ID
    bag_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    # Define the GLB model path
    # Use the scale, translate, and reference system from the feature data to transform the GLB model and align it with the 3DBAG mesh
    if bag_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        # glb_dataset = "pijlkruidstraat11-13-15.glb"
        # glb_dataset = "pijlkruid37-37.glb"
        glb_dataset = "rietstraat31-33.glb"
        glb_model_path = data_folder + glb_dataset

        # Enable post-processing to get the texture coordinates and normals of the GLB model e,g. enable_post_processing=True
        glb_mesh, initial_transformation = load_and_transform_glb_model(glb_model_path, translate)

        # Check if the GLB model was loaded successfully
        if glb_mesh:

            # Initialize a list to store the transformations applied to the GLB mesh 
            logging.info("Starting alignment process...")
            transformations = [initial_transformation]

            # Refine the alignment using ICP and calculate the final transformation matrix (source = GLB, target = 3DBAG)
            # glb_mesh, icp_transformation = full_registration_pipeline(glb_mesh, bag_mesh)
            glb_mesh, icp_transformation = refine_alignment_with_icp(glb_mesh, bag_mesh)
            transformations.append(icp_transformation)

            # Compute and apply the Z offset to align the floor of the GLB mesh with the 3DBAG mesh
            z_offset = compute_z_offset(bag_mesh, glb_mesh)
            apply_z_offset(glb_mesh, z_offset)
            transformations.append(np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, z_offset],
                [0, 0, 0, 1]
            ]))

            # Extract the latitude, longitude, and orientation from the GLB mesh vertices using the reference system
            lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
            logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")
            get
            # Accumulate the transformations to get the final transformation matrix
            final_transformation_matrix = accumulate_transformations(transformations)
            logging.info(f"Final transformation matrix:\n{final_transformation_matrix}")

            # Save the final transformation matrix and the lat, lon, and orientation to text files
            transformation_matrix_filename = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
            np.savetxt(transformation_matrix_filename, final_transformation_matrix)
            with open(f"RESULTS/{glb_dataset.split('.')[0]}_lat_lon_orientation.txt", "w") as file:
                file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")
    print_memory_usage("end")

    # Visualize the 3DBAG mesh and the aligned GLB mesh with height coloring
    visualize_meshes_with_height_coloring(bag_mesh, glb_mesh, colormap_1="YlOrRd", colormap_2="YlGnBu_r")

    bag_mesh += glb_mesh
    combined_ply_filename = glb_dataset.replace('.glb', '_bag_glb.ply')
    o3d.io.write_triangle_mesh(f"RESULTS/{combined_ply_filename}", bag_mesh, print_progress=True)
    print_memory_usage("after writing to file")
if __name__ == "__main__":
    main()
