import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_meshes_as_points(glb_mesh: trimesh.Trimesh, bag_mesh: trimesh.Trimesh, downsample_factor: int = 1000):
    """
    Visualize two 3D meshes (GLB and BAG) as point clouds using a 3D plot with matplotlib.
    
    Parameters:
    - glb_mesh: trimesh.Trimesh object for the GLB mesh.
    - bag_mesh: trimesh.Trimesh object for the BAG mesh.
    - downsample_factor: Factor by which to downsample the GLB mesh points for better performance.
    """
    # Check the number of vertices in both meshes
    print(f"GLB mesh has {len(glb_mesh.vertices)} vertices")
    print(f"BAG mesh has {len(bag_mesh.vertices)} vertices")

    # Ensure there are enough points in both meshes
    if len(glb_mesh.vertices) < 2 or len(bag_mesh.vertices) < 2:
        print("One of the meshes has too few points to display.")
        return

    # Downsample GLB mesh only if it has too many points
    glb_points = glb_mesh.vertices[::downsample_factor] if len(glb_mesh.vertices) > downsample_factor else glb_mesh.vertices
    bag_points = bag_mesh.vertices  # No downsampling for BAG mesh

    print(f"GLB points after downsampling: {glb_points.shape}")
    print(f"BAG points (no downsampling): {bag_points.shape}")

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot GLB mesh points
    ax.scatter(glb_points[:, 0], glb_points[:, 1], glb_points[:, 2], c='b', label='GLB Mesh Points', alpha=0.6)

    # Plot BAG mesh points
    ax.scatter(bag_points[:, 0], bag_points[:, 1], bag_points[:, 2], c='g', label='BAG Mesh Points', alpha=0.6)

    # Set labels and title
    ax.set_title('GLB and BAG Meshes as Point Clouds')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    ax.legend()

    # Show plot
    plt.show()

