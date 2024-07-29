import numpy as np
import open3d as o3d
import logging

def create_mesh_from_feature(feature):
    """Create a mesh object from feature data."""
    if 'vertices' in feature['feature']:
        vertices = np.array(feature['feature']['vertices'])
        transform = feature['metadata'].get('transform', {})
        scale = np.array(transform.get('scale', [1, 1, 1]))
        translate = np.array(transform.get('translate', [0, 0, 0]))
        vertices = vertices * scale + translate

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        city_objects = feature['feature'].get('CityObjects', {})
        for obj in city_objects.values():
            for geom in obj.get('geometry', []):
                for boundary in geom.get('boundaries', []):
                    for i in range(1, len(boundary[0]) - 1):
                        mesh.triangles.append([boundary[0][0], boundary[0][i], boundary[0][i + 1]])

        mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
        return mesh, scale, translate
    else:
        logging.error("No vertices found in the feature data.")
        return None, None, None

def load_and_transform_glb_model(file_path, translate):
    """Load and transform a GLB model."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        logging.error("The GLB model has no vertices or triangles.")
        return None

    vertices = np.asarray(mesh.vertices)
    transformation_matrix = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) # Transform the GLB model to the right-handed coordinate system
    vertices = np.dot(vertices, transformation_matrix.T) + translate
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh

def align_mesh_centers(mesh1, mesh2):
    """Align the center of mesh2 to mesh1."""
    center1 = mesh1.get_center()
    center2 = mesh2.get_center()
    translation = center1 - center2
    vertices = np.asarray(mesh2.vertices) + translation
    mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh2, translation  # Return the translation used for alignment
