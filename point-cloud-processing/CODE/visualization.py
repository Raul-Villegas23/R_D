import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import rotate
from geometry_utils import calculate_centroid, compute_orientation

def visualize_glb_and_combined_meshes(mesh1, mesh2):
    """Visualize the GLB and combined meshes using Matplotlib."""
    vertices1 = np.asarray(mesh1.vertices)
    triangles1 = np.asarray(mesh1.triangles)
    
    # Simplify the second mesh for visualization purposes
    if mesh2.has_triangle_uvs():
        mesh2.triangle_uvs = o3d.utility.Vector2dVector([])
    
    mesh2 = mesh2.simplify_quadric_decimation(1000) #1000 is the number of vertices after simplification
    vertices2 = np.asarray(mesh2.vertices)
    triangles2 = np.asarray(mesh2.triangles)

    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], c='k', marker='o', s=5, label='3D BAG Mesh Vertices')

    # Create a 3D surface plot using plot_trisurf with a colormap
    ax.plot_trisurf(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], triangles=triangles1, cmap='viridis', edgecolor='k', alpha=0.5)
    ax.plot_trisurf(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], triangles=triangles2, cmap='viridis', edgecolor='k', alpha=0.5)  
    
    # Auto scale to the mesh size
    scale = np.concatenate((vertices1, vertices2)).flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    # Figure name
    ax.set_title('3D BAG and GLB Meshes')
    
    # Set axis limits based on the range of vertices
    xlim = (min(vertices1[:, 0].min(), vertices2[:, 0].min()), max(vertices1[:, 0].max(), vertices2[:, 0].max()))
    ylim = (min(vertices1[:, 1].min(), vertices2[:, 1].min()), max(vertices1[:, 1].max(), vertices2[:, 1].max()))
    zlim = (min(vertices1[:, 2].min(), vertices2[:, 2].min()), max(vertices1[:, 2].max(), vertices2[:, 2].max()))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Show the plot
    plt.show()

def visualize_2d_perimeters(perimeter1, perimeter2, perimeter3):
    """Visualize three 2D perimeters, their centroids, orientations, and longest edges using Matplotlib."""
    fig, ax = plt.subplots()
    ax.plot(perimeter1[:, 0], perimeter1[:, 1], 'r-', label='3D BAG Mesh Perimeter')
    ax.plot(perimeter2[:, 0], perimeter2[:, 1], 'b-', label='GLB Mesh Perimeter')
    ax.plot(perimeter3[:, 0], perimeter3[:, 1], 'g--', label='Non-aligned Perimeter')  # Dashed lines for the third perimeter

    # Calculate and plot centroids
    centroid1 = calculate_centroid(perimeter1)
    centroid2 = calculate_centroid(perimeter2)
    centroid3 = calculate_centroid(perimeter3)
    
    ax.plot(centroid1[0], centroid1[1], 'ro', label='Centroid 3D BAG Mesh')
    ax.plot(centroid2[0], centroid2[1], 'bo', label='Centroid GLB Mesh')
    ax.plot(centroid3[0], centroid3[1], 'go', label='Centroid Non-aligned')

    # Compute and display orientations
    orientation1, longest_edge1 = compute_orientation(perimeter1)
    orientation2, longest_edge2 = compute_orientation(perimeter2)
    orientation3, longest_edge3 = compute_orientation(perimeter3)

    # ax.text(centroid1[0], centroid1[1], f'{orientation1:.1f}°', color='red', fontsize=12, ha='right')
    ax.text(centroid2[0], centroid2[1], f'{orientation2:.1f}°', color='blue', fontsize=12, ha='right')
    # ax.text(centroid3[0], centroid3[1], f'{orientation3:.1f}°', color='green', fontsize=12, ha='right')

    # Plot north and east direction arrow (adjust the coordinates as needed)
    # ax.plot([centroid2[0], centroid2[0]], [centroid2[1], centroid2[1] + 6], 'k--', linewidth= 0.5)
    # ax.plot([centroid2[0], centroid2[0] + 6], [centroid2[1], centroid2[1]], 'k--', linewidth= 0.5)
 
    # Plot the longest edges
    # if longest_edge1[0] is not None and longest_edge1[1] is not None:
    #     ax.plot([longest_edge1[0][0], longest_edge1[1][0]], [longest_edge1[0][1], longest_edge1[1][1]], 'r--', linewidth=2, label='Longest Edge 3D BAG Mesh')
    if longest_edge2[0] is not None and longest_edge2[1] is not None:
        ax.plot([longest_edge2[0][0], longest_edge2[1][0]], [longest_edge2[0][1], longest_edge2[1][1]], 'b--', linewidth=1, label='Longest Edge GLB Mesh')
    # if longest_edge3[0] is not None and longest_edge3[1] is not None:
    #     ax.plot([longest_edge3[0][0], longest_edge3[1][0]], [longest_edge3[0][1], longest_edge3[1][1]], 'g--', linewidth=2, label='Longest Edge Non-optimized')

    # Plot orientation lines from centroid to the direction given by orientation angle
    def plot_orientation_line(centroid, orientation, color):
        length = 2.0  # Length of the orientation line
        end_x = centroid[0] + length * np.cos(np.radians(orientation))
        end_y = centroid[1] + length * np.sin(np.radians(orientation))
        ax.plot([centroid[0], end_x], [centroid[1], end_y], color=color, linestyle='--')

    # plot_orientation_line(centroid1, orientation1, 'red')
    plot_orientation_line(centroid2, orientation2, 'blue')
    # plot_orientation_line(centroid3, orientation3, 'green')
    # Adjust legend position to be outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_title('2D Perimeters')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make room for the legend
    plt.show()

def color_mesh_by_height(mesh):
    vertices = np.asarray(mesh.vertices)
    heights = vertices[:, 2]
    min_height = np.min(heights)
    max_height = np.max(heights)
    normalized_heights = (heights - min_height) / (max_height - min_height)
    colors = plt.get_cmap('plasma')(normalized_heights)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def visualize_meshes_with_height_coloring(combined_mesh, glb_mesh):
    """
    Color the meshes by height and visualize them using Open3D.

    Parameters:
    - combined_mesh (open3d.geometry.TriangleMesh): The combined mesh to be visualized.
    - glb_mesh (open3d.geometry.TriangleMesh): The GLB mesh to be visualized.
    """
    # Color the meshes based on height
    glb_mesh = color_mesh_by_height(glb_mesh)
    combined_mesh = color_mesh_by_height(combined_mesh)

    # Visualize the meshes using Open3D
    o3d.visualization.draw_geometries(
        [combined_mesh, glb_mesh],
        window_name="3D BAG and GLB Meshes",
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=True
    )