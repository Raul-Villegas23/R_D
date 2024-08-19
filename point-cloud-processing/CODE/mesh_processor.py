import numpy as np
import open3d as o3d
import logging

def apply_transformation(mesh, transformation_matrix):
    """Apply transformation matrix to the mesh."""
    vertices = np.asarray(mesh.vertices)
    transformed_vertices = (transformation_matrix[:3, :3] @ vertices.T).T + transformation_matrix[:3, 3]
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    mesh.compute_vertex_normals()
    return mesh

def create_mesh_from_feature(feature):
    """Create a mesh object using only the highest LoD from feature data."""
    if 'vertices' in feature['feature']:
        vertices = np.array(feature['feature']['vertices'])
        transform = feature['metadata'].get('transform', {})
        scale = np.array(transform.get('scale', [1, 1, 1]))
        translate = np.array(transform.get('translate', [0, 0, 0]))
        vertices = vertices * scale + translate

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        city_objects = feature['feature'].get('CityObjects', {})
        max_lod = None
        max_lod_geom = None

        # Find the highest LoD
        for obj in city_objects.values():
            for geom in obj.get('geometry', []):
                lod = geom.get('lod', None)
                if max_lod is None or (lod is not None and float(lod) > float(max_lod)):
                    max_lod = lod
                    max_lod_geom = geom

        if max_lod_geom:
            # print(f"Using highest LoD: {max_lod}")

            triangle_count = 0  # To keep track of the number of triangles created

            for boundary_group in max_lod_geom.get('boundaries', []):
                # Each boundary_group may contain multiple boundaries (each a polygon)
                for boundary in boundary_group:
                    if isinstance(boundary[0], list):  # Check if the boundary is nested
                        for sub_boundary in boundary:
                            # print(f"  Sub-boundary: {sub_boundary}")  # Debugging: Print the sub-boundary structure
                            if len(sub_boundary) >= 3:  # Ensure there are at least 3 points to form a triangle
                                for i in range(1, len(sub_boundary) - 1):
                                    mesh.triangles.append([sub_boundary[0], sub_boundary[i], sub_boundary[i + 1]])
                                    triangle_count += 1
                                    # Print the triangle that is being added
                                    # print(f"Triangle {triangle_count} (LoD {max_lod}): [{sub_boundary[0]}, {sub_boundary[i]}, {sub_boundary[i + 1]}]")
                    else:
                        # print(f"  Boundary: {boundary}")  # Debugging: Print the boundary structure
                        if len(boundary) >= 3:  # Ensure there are at least 3 points to form a triangle
                            for i in range(1, len(boundary) - 1):
                                mesh.triangles.append([boundary[0], boundary[i], boundary[i + 1]])
                                triangle_count += 1
                                # Print the triangle that is being added
                                # print(f"Triangle {triangle_count} (LoD {max_lod}): [{boundary[0]}, {boundary[i]}, {boundary[i + 1]}]")

            mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
            # print(f"Total number of triangles created: {triangle_count}")
            return mesh, scale, translate
        else:
            logging.error("No geometry data found for the highest LoD.")
            return None, None, None
    else:
        logging.error("No vertices found in the feature data.")
        return None, None, None

def load_and_transform_glb_model(file_path, translate):
    """Load and transform a GLB model."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        logging.error("The GLB model has no vertices or triangles.")
        return None, None

    vertices = np.asarray(mesh.vertices)

    # Transformation to convert the GLB model to the right-handed coordinate system
    transformation_matrix = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    vertices = np.dot(vertices, transformation_matrix.T) + translate
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    
    # Create the initial transformation matrix as a homogeneous transformation
    initial_transformation = np.eye(4)
    initial_transformation[:3, :3] = transformation_matrix
    initial_transformation[:3, 3] = translate
    
    return mesh, initial_transformation


def align_mesh_centers(mesh1, mesh2):
    """Align the center of mesh2 to mesh1."""
    center1 = mesh1.get_center()
    center2 = mesh2.get_center()
    translation = center1 - center2
    vertices = np.asarray(mesh2.vertices) + translation
    mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh2, translation  # Return the translation used for alignment

def apply_optimal_params(mesh, optimal_angle, optimal_tx, optimal_ty):
    """Apply the optimal rotation and translation to the mesh."""
    # Apply optimal rotation
    rotation_matrix = mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(optimal_angle)))
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    
    # Apply optimal translation
    vertices = np.asarray(mesh.vertices)
    vertices[:, :2] += [optimal_tx, optimal_ty]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Transformation matrix for the optimal parameters
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [optimal_tx, optimal_ty, 0]
    
    return mesh
