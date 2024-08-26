import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from geometry_utils import calculate_centroid, compute_orientation


def visualize_2d_perimeters(perimeter1, perimeter2):
    """Visualize three 2D perimeters, their centroids, orientations, and longest edges using Matplotlib."""
    fig, ax = plt.subplots()
    ax.plot(perimeter1[:, 0], perimeter1[:, 1], 'r-', label='3D BAG Mesh Perimeter')
    ax.plot(perimeter2[:, 0], perimeter2[:, 1], 'b-', label='GLB Mesh Perimeter')
    
    # Calculate and plot centroids
    centroid1 = calculate_centroid(perimeter1)
    centroid2 = calculate_centroid(perimeter2)

    
    ax.plot(centroid1[0], centroid1[1], 'ro', label='Centroid 3D BAG Mesh')
    ax.plot(centroid2[0], centroid2[1], 'bo', label='Centroid GLB Mesh')


    # Compute and display orientations
    orientation2, longest_edge2 = compute_orientation(perimeter2)

    # ax.text(centroid1[0], centroid1[1], f'{orientation1:.1f}°', color='red', fontsize=12, ha='right')
    ax.text(centroid2[0], centroid2[1], f'{orientation2:.1f}°', color='blue', fontsize=12, ha='right')
    # ax.text(centroid3[0], centroid3[1], f'{orientation3:.1f}°', color='green', fontsize=12, ha='right')

    # Plot the longest edges
    if longest_edge2[0] is not None and longest_edge2[1] is not None:
        ax.plot([longest_edge2[0][0], longest_edge2[1][0]], [longest_edge2[0][1], longest_edge2[1][1]], 'b--', linewidth=1, label='Longest Edge GLB Mesh')

    # Plot orientation lines from centroid to the direction given by orientation angle
    def plot_orientation_line(centroid, orientation, color):
        length = 2.0  # Length of the orientation line
        end_x = centroid[0] + length * np.cos(np.radians(orientation))
        end_y = centroid[1] + length * np.sin(np.radians(orientation))
        ax.plot([centroid[0], end_x], [centroid[1], end_y], color=color, linestyle='--')

    plot_orientation_line(centroid2, orientation2, 'blue')

    # Adjust legend position to be outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_title('2D Perimeters')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make room for the legend
    plt.show()

def visualize_meshes_with_height_coloring(bag_mesh, glb_mesh, colormap_1='YlGnBu', colormap_2='YlOrRd'):

    def color_mesh_by_height(mesh, colormap_name):
        vertices = np.asarray(mesh.vertices)
        heights = vertices[:, 2]
        min_height = np.min(heights)
        max_height = np.max(heights)
        normalized_heights = (heights - min_height) / (max_height - min_height)
        colors = plt.get_cmap(colormap_name)(normalized_heights)[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    # Color a copy of the meshes based on height with different colormaps
    bag_mesh_colored = color_mesh_by_height(bag_mesh, colormap_1)
    glb_mesh_colored = color_mesh_by_height(glb_mesh, colormap_2)

    # Visualize the meshes using Open3D
    o3d.visualization.draw_geometries(
        [bag_mesh_colored, glb_mesh_colored],
        window_name="3D BAG and GLB Meshes",
        width=1000,
        height=800,
        left=50,
        top=50,
    )