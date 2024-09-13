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


def color_origin_vertex(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Color the vertex at (0, 0, 0) in the mesh with a big red dot.

    Args:
        mesh (trimesh.Trimesh): The mesh object to be processed.

    Returns:
        trimesh.Trimesh: A new mesh object with the (0, 0, 0) vertex colored red.
    """
    # Extract vertices
    vertices = np.asarray(mesh.vertices)
    
    # Calculate the distance from (0, 0, 0) for each vertex
    distances = np.linalg.norm(vertices, axis=1)
    
    # Get the index of the vertex closest to (0, 0, 0)
    origin_vertex_index = np.argmin(distances)
    
    # Create a new vertex colors array with the same length as vertices
    vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    
    # Set the color for the (0, 0, 0) vertex to red
    vertex_colors[origin_vertex_index] = [255, 0, 0, 255]  # Red with full opacity
    
    # Create a new mesh with vertex colors
    # Use TextureVisuals if the mesh has textures, otherwise use ColorVisuals
    if hasattr(mesh.visual, 'texture'):
        texture = mesh.visual.texture
        new_visual = trimesh.visual.TextureVisuals(
            texture=texture,
            vertex_colors=vertex_colors
        )
    else:
        new_visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
    
    # Create a new mesh with the updated visual
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, visual=new_visual)
    
    return new_mesh

