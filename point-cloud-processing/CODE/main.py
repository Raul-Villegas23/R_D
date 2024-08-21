import logging
import time
import numpy as np
import open3d as o3d
from fetcher import process_feature_list
from geolocation import extract_latlon_orientation_from_mesh
from mesh_processor import load_and_transform_glb_model, align_mesh_centers, apply_optimal_params
from geometry_utils import extract_2d_perimeter, optimize_rotation_and_translation
from transformation_utils import compute_z_offset, apply_z_offset, accumulate_transformations, create_center_based_transformation_matrix
from visualization_utils import visualize_2d_perimeters, visualize_meshes_with_height_coloring
from icp_alignment import refine_alignment_with_icp



def main():
    start_time = time.time()

    # Define the API URL and the collection ID and feature IDs to process
    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    # feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  # Pijlkruidstraat 11, 13 and 15
    # feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb

    # Process the feature list and combine the meshes into a single mesh
    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if combined_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        # glb_dataset = "pijlkruidstraat11-13-15.glb"
        # glb_dataset = "pijlkruid37-37.glb"
        glb_dataset = "rietstraat31-33.glb"
        glb_model_path = data_folder + glb_dataset

        # Transform the GLB model and align its center with the combined mesh from 3DBAG
        glb_mesh, initial_transformation = load_and_transform_glb_model(glb_model_path, translate)

        if glb_mesh:

            # Initialize transformations with the initial transformation matrix
            transformations = [initial_transformation]

            # Align the centers of the two meshes
            glb_mesh, center_translation = align_mesh_centers(combined_mesh, glb_mesh)

            # Create a transformation matrix based on the center translation
            center_translation_matrix = np.eye(4)
            center_translation_matrix[:3, 3] = center_translation
            transformations.append(center_translation_matrix) # Append the center translation matrix

            # Extract 2D perimeters from the combined and GLB meshes
            perimeter1, perimeter2 = extract_2d_perimeter(combined_mesh), extract_2d_perimeter(glb_mesh)

            # Find the optimal rotation angle and translation to align the perimeters using the error between their intersections
            optimal_params = optimize_rotation_and_translation(perimeter1, perimeter2)
            if optimal_params is not None:
                optimal_angle, optimal_tx, optimal_ty = optimal_params

                # Save the optimal parameters for later use
                # np.savetxt(f"RESULTS/{glb_dataset.split('.')[0]}_optimal_params.txt", [optimal_angle, optimal_tx, optimal_ty])
                glb_mesh = apply_optimal_params(glb_mesh, optimal_angle, optimal_tx, optimal_ty)

                # Create a transformation matrix based on the optimal parameters
                optimal_transformation_matrix = create_center_based_transformation_matrix(glb_mesh, optimal_angle, optimal_tx, optimal_ty)
                transformations.append(optimal_transformation_matrix) # Append the optimal transformation matrix

                try:
                    # Compute the Z offset needed to align the floor of the GLB mesh with the combined mesh
                    z_offset1 = compute_z_offset(combined_mesh, glb_mesh)
                    apply_z_offset(glb_mesh, z_offset1)

                    # Update the transformation matrix with the z_offset
                    z_offset_matrix1 = np.eye(4)
                    z_offset_matrix1[2, 3] = z_offset1
                    transformations.append(z_offset_matrix1)

                except ValueError as e:
                    logging.error(f"Error computing Z-offset: {e}")
                    return
                
                # Refine the alignment using ICP and calculate the final transformation matrix
                glb_mesh, icp_transformation = refine_alignment_with_icp(glb_mesh, combined_mesh)
                transformations.append(icp_transformation) # Append the ICP transformation matrix

                try:
                    # Compute the Z offset needed to align the floor of the GLB mesh with the combined mesh after ICP
                    z_offset2 = compute_z_offset(combined_mesh, glb_mesh)
                    apply_z_offset(glb_mesh, z_offset2)

                    # Update the transformation matrix with the z_offset
                    z_offset_matrix2 = np.eye(4)
                    z_offset_matrix2[2, 3] = z_offset2
                    transformations.append(z_offset_matrix2)

                except ValueError as e:
                    logging.error(f"Error computing Z-offset: {e}")
                    return

                # Extract the latitude, longitude, and orientation from the GLB mesh vertices using the reference system
                lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
                logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")

                # Accumulate the transformations to get the final transformation matrix
                final_transformation_matrix = accumulate_transformations(transformations)
                logging.info(f"Final transformation matrix:\n{final_transformation_matrix}")

                # Save the final transformation matrix and the lat, lon, and orientation to text files
                # transformation_matrix_filename = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
                # np.savetxt(transformation_matrix_filename, final_transformation_matrix)
                # with open(f"RESULTS/{glb_dataset.split('.')[0]}_lat_lon_orientation.txt", "w") as file:
                #     file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")
                
                # Compute vertex normals after transformation
                # glb_mesh.compute_vertex_normals()

                visualize_meshes_with_height_coloring(combined_mesh, glb_mesh)

                # Save the glb mesh as a GLB file
                glb_filename = glb_dataset.replace('.glb', '_aligned.glb')
                o3d.io.write_triangle_mesh(f"RESULTS/{glb_filename}", glb_mesh, print_progress=True)

                # Save the GLB mesh as a Obj file
                obj_filename = glb_dataset.replace('.glb', '.obj')
                o3d.io.write_triangle_mesh(f"RESULTS/{obj_filename}", glb_mesh, print_progress=True)

                # Save the GLB mesh as a PLY file
                ply_filename = glb_dataset.replace('.glb', '.ply')
                o3d.io.write_triangle_mesh(f"RESULTS/{ply_filename}", glb_mesh,  print_progress=True)

                # Save the GLB mesh as a GLTF file
                gltf_filename = glb_dataset.replace('.glb', '.gltf')
                o3d.io.write_triangle_mesh(f"RESULTS/{gltf_filename}", glb_mesh, print_progress=True)

                # Save the combined mesh as a PLY file with a distinct filename with the glb dataset name
                combined_ply_filename = glb_dataset.replace('.glb', '_combined.ply')
                o3d.io.write_triangle_mesh(f"RESULTS/{combined_ply_filename}", combined_mesh, print_progress=True)

                # Combine the glb and combined meshes into one and then save them as a PLY file
                combined_mesh += glb_mesh
                combined_ply_filename = glb_dataset.replace('.glb', '_combined_meshes.ply')
                o3d.io.write_triangle_mesh(f"RESULTS/{combined_ply_filename}", combined_mesh, print_progress=True)


    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")
    # Visualize the combined and GLB meshes with height coloring
    # # Visualize the 2D perimeters and the intersection
    # perimeter3 = extract_2d_perimeter(glb_mesh) # Extract the 2D perimeter of the GLB mesh after alignment
    # visualize_2d_perimeters(perimeter1, perimeter3, perimeter2)

if __name__ == "__main__":
    main()
