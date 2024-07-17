import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the GLB model from the DATA folder
glb_path = 'DATA/model.glb'
glb_model = o3d.io.read_triangle_mesh(glb_path)

# Downsample the vertices using quadric decimation
target_number_of_vertices = 1000  # Adjust the number of vertices as needed
glb_model = glb_model.simplify_quadric_decimation(target_number_of_vertices)

def visualize_transformed_mesh(mesh):
    """Visualize and animate the transformed GLB mesh using Matplotlib."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Transformation matrix: Reflect the x-axis and remap y to z
    transformation_matrix = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    
    # Apply the transformation to the vertices
    transformed_vertices = vertices.dot(transformation_matrix)
    
    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D surface plot using plot_trisurf with a colormap
    trisurf = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, cmap='winter', edgecolor='k', alpha=0.5)
    
    # Set axis limits based on the range of vertices
    xlim = (min(vertices[:, 0].min(), transformed_vertices[:, 0].min()), max(vertices[:, 0].max(), transformed_vertices[:, 0].max()))
    ylim = (min(vertices[:, 1].min(), transformed_vertices[:, 1].min()), max(vertices[:, 1].max(), transformed_vertices[:, 1].max()))
    zlim = (min(vertices[:, 2].min(), transformed_vertices[:, 2].min()), max(vertices[:, 2].max(), transformed_vertices[:, 2].max()))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Update function for animation
    def update(angle):
        nonlocal transformed_vertices
        
        # Rotate the transformed vertices around the z-axis
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
        
        rotated_vertices = transformed_vertices.dot(rotation_matrix.T)
        
        ax.clear()
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, cmap='winter', edgecolor='k', alpha=0.5)
        ax.plot_trisurf(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], triangles=triangles, cmap='summer', edgecolor='k', alpha=0.5)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        return ax,

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

    # Show the plot
    plt.show()

# Call the function to visualize and animate the transformed mesh
visualize_transformed_mesh(glb_model)
