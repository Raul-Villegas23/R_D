import logging
import time
import numpy as np
import open3d as o3d
from fetcher import fetch_json
from mesh_processor import create_mesh_from_feature, load_and_transform_glb_model, align_mesh_centers
from geometry_utils import extract_2d_perimeter, extract_latlon_orientation_from_mesh, calculate_intersection_error
from transformation import optimize_rotation_and_translation, compute_z_offset, apply_z_offset, calculate_transformation_matrix, accumulate_transformations
from visualization import visualize_glb_and_combined_meshes, visualize_2d_perimeters, color_mesh_by_height
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
    feature_ids = ["NL.IMBAG.Pand.0141100000048693", "NL.IMBAG.Pand.0141100000048692", "NL.IMBAG.Pand.0141100000049132"]
    combined_mesh, scale, translate, reference_system = process_feature_list(collections_url, collection_id, feature_ids)
    
    if combined_mesh and scale is not None and translate is not None and reference_system is not None:
        data_folder = "DATA/"
        glb_dataset = "pijlkruidstraat11-13-15.glb"
        glb_model_path = data_folder + glb_dataset
        glb_mesh = load_and_transform_glb_model(glb_model_path, translate)
        if glb_mesh:
            glb_mesh, center_translation = align_mesh_centers(combined_mesh, glb_mesh)
            perimeter1, perimeter2 = extract_2d_perimeter(combined_mesh), extract_2d_perimeter(glb_mesh)

            visualize_glb_and_combined_meshes(combined_mesh, glb_mesh)
            visualize_2d_perimeters(perimeter1, perimeter2, perimeter2)

            optimal_params = optimize_rotation_and_translation(perimeter1, perimeter2)
            if optimal_params is not None:
                optimal_angle, optimal_tx, optimal_ty = optimal_params
                rotation_matrix = glb_mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(optimal_angle)))
                glb_mesh.rotate(rotation_matrix, center=glb_mesh.get_center())
                vertices = np.asarray(glb_mesh.vertices)
                vertices[:, :2] += [optimal_tx, optimal_ty]
                glb_mesh.vertices = o3d.utility.Vector3dVector(vertices)

                try:
                    z_offset1 = compute_z_offset(combined_mesh, glb_mesh)
                    apply_z_offset(glb_mesh, z_offset1)
                except ValueError as e:
                    print(f"Error computing Z-offset: {e}")
                    return
                
                glb_mesh, icp_transformation = refine_alignment_with_icp(glb_mesh, combined_mesh)
                logging.info(f"ICP Transformation Matrix:\n{icp_transformation}")

                try:
                    z_offset2 = compute_z_offset(combined_mesh, glb_mesh)
                    apply_z_offset(glb_mesh, z_offset2)
                except ValueError as e:
                    print(f"Error computing Z-offset: {e}")
                    return
                
                lon, lat, orientation = extract_latlon_orientation_from_mesh(glb_mesh, reference_system)
                logging.info(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}, Orientation: {orientation:.5f} degrees")
                
                initial_transformation = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
                transformation_matrix = calculate_transformation_matrix(initial_transformation, optimal_angle, translate, center_translation, z_offset1)
                final_transformation_matrix = icp_transformation @ transformation_matrix
                logging.info(f"Final Transformation Matrix:\n{final_transformation_matrix}")

                transformation_matrix_filename = f"RESULTS/{glb_dataset.split('.')[0]}_transformation_matrix.txt"
                np.savetxt(transformation_matrix_filename, final_transformation_matrix)
                with open(f"RESULTS/{glb_dataset.split('.')[0]}_lat_lon_orientation.txt", "w") as file:
                    file.write(f"Latitude: {lat:.5f}\nLongitude: {lon:.5f}\nOrientation: {orientation:.5f}")

                visualize_glb_and_combined_meshes(combined_mesh, glb_mesh)
                glb_mesh = color_mesh_by_height(glb_mesh)
                combined_mesh = color_mesh_by_height(combined_mesh)
                o3d.visualization.draw_geometries([combined_mesh, glb_mesh], window_name="3D BAG and GLB Meshes", width=800, height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=True)

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
