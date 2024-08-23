# Libraries for API requests and data processing
import logging
import time

# Libraries for point cloud processing
import numpy as np
import open3d as o3d

 # Import the necessary functions from the custom modules
from fetcher import process_feature_list
from geolocation import extract_latlon_orientation_from_mesh
from mesh_processor import load_and_transform_glb_model, align_mesh_centers, apply_optimal_params
from geometry_utils import extract_2d_perimeter, optimize_rotation_and_translation
from transformation_utils import compute_z_offset, apply_z_offset, accumulate_transformations, create_center_based_transformation_matrix
from visualization_utils import visualize_meshes_with_height_coloring, visualize_2d_perimeters
from icp_alignment import refine_alignment_with_icp



def main():
    start_time = time.time()

    # Define the API URL and the collection ID and feature IDs to process the 3DBAG data
    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    # feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  # pijlkruidstraat11-13-15.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb

    # Process the feature list and combine the meshes into a single mesh
    # It returns the combined bag_mesh, scale, translate, and reference system from the feature data
    bag_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    # Define the GLB model path
    # Use the scale, translate, and reference system from the feature data to transform the GLB model and align it with the 3DBAG mesh
    if bag_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        # glb_dataset = "pijlkruidstraat11-13-15.glb"
        # glb_dataset = "pijlkruid37-37.glb"
        glb_dataset = "rietstraat31-33.glb"
        glb_model_path = data_folder + glb_dataset

        # Transform the GLB model and align its center with the combined mesh from 3DBAG
        # It returns the transformed glb_mesh and the initial_transformation matrix used to change the GLB model to the right-handed coordinate system
        glb_mesh, initial_transformation = load_and_transform_glb_model(glb_model_path, translate, enable_post_processing=False)

        # Check if the GLB model was loaded successfully
        if glb_mesh:

            # Initialize a list to store the transformations applied to the GLB mesh 
            transformations = [initial_transformation]

            # Align the centers of the two meshes to prepare for a rough alignment using the perimeter of the meshes
            glb_mesh, center_translation = align_mesh_centers(bag_mesh, glb_mesh)
            visualize_meshes_with_height_coloring(bag_mesh, glb_mesh, colormap_1="YlGnBu_r", colormap_2="YlOrRd")    


            # Create a transformation matrix based on the translation of the center of the GLB mesh to the center of the bag mesh
            center_translation_matrix = np.eye(4)
            center_translation_matrix[:3, 3] = center_translation
            transformations.append(center_translation_matrix) # Append the center translation matrix to the transformations list

            # Extract 2D perimeters from the 3DBAG and GLB meshes
            perimeter1, perimeter2 = extract_2d_perimeter(bag_mesh), extract_2d_perimeter(glb_mesh)

            # Find a rough rotation angle and translation vectors to align the perimeters using the error between their intersections as a metric
            optimal_params = optimize_rotation_and_translation(perimeter1, perimeter2)
            if optimal_params is not None:
                optimal_angle, optimal_tx, optimal_ty = optimal_params

                # Apply the optimal rotation and translation to the GLB mesh
                glb_mesh = apply_optimal_params(glb_mesh, optimal_angle, optimal_tx, optimal_ty)

                # Create a transformation matrix based on these parameters
                optimal_transformation_matrix = create_center_based_transformation_matrix(glb_mesh, optimal_angle, optimal_tx, optimal_ty)
                transformations.append(optimal_transformation_matrix) # Append the optimal transformation matrix to the transformations list

                try:
                    # Compute the Z offset needed to align the floor of the GLB mesh with the bag mesh
                    z_offset1 = compute_z_offset(bag_mesh, glb_mesh)
                    apply_z_offset(glb_mesh, z_offset1)

                    # Update the transformation matrix with the z_offset
                    z_offset_matrix1 = np.eye(4)
                    z_offset_matrix1[2, 3] = z_offset1
                    transformations.append(z_offset_matrix1)

                except ValueError as e:
                    logging.error(f"Error computing Z-offset: {e}")
                    return
                
                visualize_meshes_with_height_coloring(bag_mesh, glb_mesh)
                perimeter3 = extract_2d_perimeter(glb_mesh)
                visualize_2d_perimeters(perimeter1, perimeter3, perimeter2)    

                
                # Refine the alignment using ICP and calculate the final transformation matrix (source = GLB, target = 3DBAG)
                glb_mesh, icp_transformation = refine_alignment_with_icp(glb_mesh, bag_mesh)

                transformations.append(icp_transformation) # Append the ICP transformation matrix to the transformations list
                # Compute the Z offset needed to align the floor of the GLB mesh with the bag mesh after ICP
                z_offset2 = compute_z_offset(bag_mesh, glb_mesh)
                apply_z_offset(glb_mesh, z_offset2)

                # Update the transformation matrix with the z_offset
                z_offset_matrix2 = np.eye(4)
                z_offset_matrix2[2, 3] = z_offset2
                transformations.append(z_offset_matrix2)

                # Extract the latitude, longitude, and orientation from the GLB mesh vertices using the reference system
                lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
                logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")

                # Accumulate the transformations to get the final transformation matrix
                final_transformation_matrix = accumulate_transformations(transformations)
                logging.info(f"Final transformation matrix:\n{final_transformation_matrix}")

                # Save the optimal parameters for later use in a text file (optional)
                np.savetxt(f"RESULTS/{glb_dataset.split('.')[0]}_optimal_params.txt", [optimal_angle, optimal_tx, optimal_ty])

                # Save the final transformation matrix and the lat, lon, and orientation to text files (optional)
                transformation_matrix_filename = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
                np.savetxt(transformation_matrix_filename, final_transformation_matrix)
                with open(f"RESULTS/{glb_dataset.split('.')[0]}_lat_lon_orientation.txt", "w") as file:
                    file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")
                
                # Visualize the meshes with height coloring
                visualize_meshes_with_height_coloring(bag_mesh, glb_mesh, colormap_1="YlGnBu_r", colormap_2="YlOrRd")    
                o3d.visualization.draw_geometries([bag_mesh])
                o3d.visualization.draw_geometries([glb_mesh])

                # Save the GLB mesh as a Obj file
                obj_filename = glb_dataset.replace('.glb', '.obj')
                o3d.io.write_triangle_mesh(f"RESULTS/{obj_filename}", glb_mesh, print_progress=True)

                # Save the GLB mesh as a PLY file
                ply_filename = glb_dataset.replace('.glb', '.ply')
                o3d.io.write_triangle_mesh(f"RESULTS/{ply_filename}", glb_mesh,  print_progress=True)

                # Save the bag mesh as a PLY file with a distinct filename with the glb dataset name
                bag_ply_filename = glb_dataset.replace('.glb', '_bag.ply')
                o3d.io.write_triangle_mesh(f"RESULTS/{bag_ply_filename}", bag_mesh, print_progress=True)

                # Combine the glb and bag mesh into one and then save them as a PLY file
                bag_mesh += glb_mesh
                combined_ply_filename = glb_dataset.replace('.glb', '_bag_glb.plSy')
                o3d.io.write_triangle_mesh(f"RESULTS/{combined_ply_filename}", bag_mesh, print_progress=True)

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")
    


if __name__ == "__main__":
    main()
