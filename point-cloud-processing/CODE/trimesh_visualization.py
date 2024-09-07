import trimesh
import numpy as np
import matplotlib.pyplot as plt

def apply_height_coloring(mesh: trimesh.Trimesh, colormap_name: str = 'viridis'):
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
    apply_height_coloring(bag_mesh, colormap_name='viridis')

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
