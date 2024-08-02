import logging
import time
import numpy as np
import open3d as o3d
from fetcher import fetch_json
from mesh_processor import create_mesh_from_feature, load_and_transform_glb_model, align_mesh_centers
from geometry_utils import extract_2d_perimeter, extract_latlon_orientation_from_mesh, calculate_intersection_error
from transformation import optimize_rotation_and_translation, compute_z_offset, apply_z_offset, accumulate_transformations
from visualization import visualize_glb_and_combined_meshes, visualize_2d_perimeters, color_mesh_by_height, visualize_meshes_with_height_coloring
from icp_alignment import refine_alignment_with_icp
from shapely.geometry import Polygon
from shapely.affinity import rotate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_feature_list(collections_url, collection_id, feature_ids):
    """Process a list of feature IDs and combine their meshes."""
    meshes, scale, translate, reference_system = [], None, None, None

    for feature_id in feature_ids:
        feature_url = f"{collections_url}/{collection_id}/items/{feature_id}"
        feature = fetch_json(feature_url)
        if feature:
            mesh, scale, translate = create_mesh_from_feature(feature)
            if mesh:
                meshes.append(mesh)
            reference_system = feature['metadata'].get('metadata', {}).get('referenceSystem')

    if meshes:
        combined_mesh = sum(meshes, o3d.geometry.TriangleMesh())
        return combined_mesh, scale, translate, reference_system
    else:
        logging.error("No meshes to visualize.")
        return None, None, None, None

def main():
    start_time = time.time()

    collections_url = "https://api.3dbag.nl/collections"
    collection_id = 'pand'
    # feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]  # Pijlkruidstraat 11, 13 and 15
    feature_ids = ["NL.IMBAG.Pand.0141100000049153", "NL.IMBAG.Pand.0141100000049152"] # pijlkruid37-37.glb
    # feature_ids = ["NL.IMBAG.Pand.0141100000010853", "NL.IMBAG.Pand.0141100000010852"] # rietstraat31-33.glb
    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)

    if combined_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        # glb_dataset = "pijlkruidstraat11-13-15.glb"
        glb_dataset = "pijlkruid37-37.glb"
        # glb_dataset = "rietstraat31-33.glb"
        glb_model_path = data_folder + glb_dataset
        glb_mesh, initial_transformation = load_and_transform_glb_model(glb_model_path, translate)

        if glb_mesh:
            transformations = [initial_transformation]  # Start with the initial transformation

            # Align the centers of the two meshes
            glb_mesh, center_translation = align_mesh_centers(combined_mesh, glb_mesh)
            center_translation_matrix = np.eye(4)
            center_translation_matrix[:3, 3] = center_translation
            transformations.append(center_translation_matrix)

            # Extract 2D perimeters from the combined and GLB meshes
            perimeter1, perimeter2 = extract_2d_perimeter(combined_mesh), extract_2d_perimeter(glb_mesh)

            # Visualize the 3D and 2D meshes for comparison and alignment
            # visualize_glb_and_combined_meshes(combined_mesh, glb_mesh)
            # visualize_2d_perimeters(perimeter1, perimeter2, perimeter2)

            # Optimize the rotation angle and translation to align the perimeters
            optimal_params = optimize_rotation_and_translation(perimeter1, perimeter2)
            if optimal_params is not None:
                optimal_angle, optimal_tx, optimal_ty = optimal_params
                # Save the optimal parameters for later use
                np.savetxt(f"RESULTS/{glb_dataset.split('.')[0]}_optimal_params.txt", [optimal_angle, optimal_tx, optimal_ty])
                rotation_matrix = glb_mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(optimal_angle)))
                glb_mesh.rotate(rotation_matrix, center=glb_mesh.get_center())
                vertices = np.asarray(glb_mesh.vertices)
                vertices[:, :2] += [optimal_tx, optimal_ty]
                glb_mesh.vertices = o3d.utility.Vector3dVector(vertices)

                # Create the optimal transformation matrix
                rotation_matrix_hom = np.eye(4)
                rotation_matrix_hom[:3, :3] = rotation_matrix
                optimal_translation_matrix = np.eye(4)
                optimal_translation_matrix[0, 3] = optimal_tx
                optimal_translation_matrix[1, 3] = optimal_ty
                optimal_transformation = optimal_translation_matrix @ rotation_matrix_hom
                # transformations.append(optimal_transformation)
                print(f"Optimal Transformation Matrix:\n{optimal_transformation}")

                try:
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
                logging.info(f"ICP Transformation Matrix:\n{icp_transformation}")
                transformations.append(icp_transformation)

                try:
                    z_offset2 = compute_z_offset(combined_mesh, glb_mesh)
                    apply_z_offset(glb_mesh, z_offset2)

                    # Update the transformation matrix with the z_offset
                    z_offset_matrix2 = np.eye(4)
                    z_offset_matrix2[2, 3] = z_offset2
                    transformations.append(z_offset_matrix2)
                except ValueError as e:
                    logging.error(f"Error computing Z-offset: {e}")
                    return

                lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
                logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")

                # Accumulate the transformations to get the final transformation matrix
                final_transformation_matrix = accumulate_transformations(transformations)
                logging.info(f"Final Transformation Matrix:\n{final_transformation_matrix}")

                transformation_matrix_filename = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
                np.savetxt(transformation_matrix_filename, final_transformation_matrix)
                with open(f"RESULTS/{glb_dataset.split('.')[0]}_lat_lon_orientation.txt", "w") as file:
                    file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")

                # visualize_glb_and_combined_meshes(combined_mesh, glb_mesh)
                visualize_meshes_with_height_coloring(combined_mesh, glb_mesh)

                rotated_perimeter2 = rotate(Polygon(perimeter2), optimal_angle, origin='centroid')
                translated_rotated_perimeter2 = np.array(rotated_perimeter2.exterior.coords) + [optimal_tx, optimal_ty]
                visualize_2d_perimeters(perimeter1, translated_rotated_perimeter2, perimeter2)

                error = calculate_intersection_error(optimal_params, perimeter1, perimeter2)
                logging.info(f"Intersection Error after optimization: {error:.5f}")

                ply_filename = glb_dataset.replace('.glb', '.ply')
                o3d.io.write_triangle_mesh(f"RESULTS/{ply_filename}", glb_mesh)

    logging.info(f"Elapsed time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()
