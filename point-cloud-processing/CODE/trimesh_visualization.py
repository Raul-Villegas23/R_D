import trimesh
import numpy as np
import matplotlib.pyplot as plt

def apply_height_coloring(mesh: trimesh.Trimesh, colormap_name: str = 'YlGnBu'):
    """
    Apply height-based coloring to a mesh based on the Z-coordinate.

    Parameters:
    - mesh: trimesh.Trimesh object.
    - colormap_name: Name of the colormap to use (from matplotlib).
    """
    # Get vertex heights (Z-coordinates)
    heights = mesh.vertices[:, 2]

    # Normalize heights to [0, 1] range
    min_height, max_height = heights.min(), heights.max()
    normalized_heights = (heights - min_height) / (max_height - min_height)

    # Apply the colormap
    colormap = plt.get_cmap(colormap_name)
    colors = colormap(normalized_heights)[:, :3]  # Get RGB values from the colormap

    # Assign colors to the mesh vertices
    mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)

def visualize_trimesh_objects(bag_mesh: trimesh.Trimesh, glb_mesh: trimesh.Trimesh, glb_downsample_factor: int = 1000):
    """
    Visualize two 3D meshes (BAG and GLB) using Trimesh's built-in viewer, each with different colors.
    BAG mesh will have height-based coloring.

    Parameters:
    - bag_mesh: trimesh.Trimesh object for the BAG mesh.
    - glb_mesh: trimesh.Trimesh object for the GLB mesh.
    - glb_downsample_factor: Downsampling factor for the GLB mesh.
    """

    # Apply height coloring to the BAG mesh
    apply_height_coloring(bag_mesh, colormap_name='YlGnBu')

    # Assign a solid color to the GLB mesh (e.g., green)
    glb_mesh.visual.vertex_colors = [0, 255, 0, 255]  # Green for GLB mesh 

    # Create a scene and add both meshes
    scene = trimesh.Scene()
    scene.add_geometry(bag_mesh, geom_name="BAG Mesh (Height Colored)")
    scene.add_geometry(glb_mesh, geom_name="GLB Mesh")

    # Visualize using Trimesh's viewer
    scene.show()

# Example usage:
# visualize_trimesh_objects(bag_mesh, glb_mesh)


def color_transformed_origin_vertex(
    mesh: trimesh.Trimesh, final_transformation_matrix: np.ndarray
) -> trimesh.Trimesh:
    """
    Color the vertex closest to the transformed (0, 0, 0) origin in the mesh with a red dot.

    Parameters:
        mesh (trimesh.Trimesh): The mesh object.
        final_transformation_matrix (np.ndarray): The accumulated transformation matrix applied to the mesh.
        
    Returns:
        trimesh.Trimesh: A new mesh object with the vertex closest to the transformed origin colored red.
    """
    # Step 1: Define the origin point (0, 0, 0) in the mesh's local coordinate system
    mesh_origin = np.array([0.0, 0.0, 0.0, 1.0])  # Homogeneous coordinates for transformation

    # Step 2: Apply the final transformation matrix to the mesh origin to get the transformed coordinates
    transformed_origin_homogeneous = final_transformation_matrix @ mesh_origin
    transformed_origin = transformed_origin_homogeneous[:3]  # Extract (x, y, z) from homogeneous coordinates

    # Step 3: Calculate the distance from the transformed origin to each vertex in the mesh
    vertices = np.asarray(mesh.vertices)
    distances = np.linalg.norm(vertices - transformed_origin, axis=1)

    # Step 4: Get the index of the vertex closest to the transformed origin
    origin_vertex_index = np.argmin(distances)

    # Step 5: Create a new vertex colors array with the same length as the number of vertices
    vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)  # Initialize as black (or transparent)

    # Set the color of the closest vertex to red (RGB + Alpha)
    vertex_colors[origin_vertex_index] = [255, 0, 0, 255]  # Red with full opacity

    # Step 6: Create a new visual (color or texture-based, depending on the mesh) with the vertex colors
    if hasattr(mesh.visual, 'texture'):
        texture = mesh.visual.texture
        new_visual = trimesh.visual.TextureVisuals(
            texture=texture,
            vertex_colors=vertex_colors
        )
    else:
        new_visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)

    # Step 7: Create a new mesh with the updated visual
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, visual=new_visual)
    
    return new_mesh

