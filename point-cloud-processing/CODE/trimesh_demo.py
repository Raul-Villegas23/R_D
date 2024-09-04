import trimesh
import numpy as np
import trimesh.registration as reg
import matplotlib.pyplot as plt

from trimesh_alignment import refine_alignment_with_icp

# Function to generate random points on a sphere
def generate_sphere_points(radius: float, count: int = 1000) -> np.ndarray:
    """Generate random points on a sphere's surface."""
    points = np.random.randn(count, 3)
    points /= np.linalg.norm(points, axis=1).reshape(-1, 1)
    return points * radius

# Generate two spheres: one as the source and the other as the target (rotated and translated)
source_points = generate_sphere_points(radius=1.0, count=1000)
target_points = generate_sphere_points(radius=1.0, count=1000)

# Apply a transformation to the target points (rotation and translation)
rotation_angle = np.radians(30)  # Rotate by 30 degrees
rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, [0, 0, 1])

translation_vector = [0.5, 0.5, 0.2]
translation_matrix = trimesh.transformations.translation_matrix(translation_vector)

transformation_matrix = np.dot(translation_matrix, rotation_matrix)

# Transform the target points
target_points_transformed = trimesh.transformations.transform_points(target_points, transformation_matrix)

# Run ICP to align the source to the transformed target
matrix, aligned_source_points, cost = reg.icp(source_points, target_points_transformed, max_iterations=100)

print("Final Transformation Matrix from ICP:\n", matrix)
print("Final Cost (error):", cost)

# Visualization using matplotlib - 2D Projection
def plot_2d_projection(points, title, ax, color, marker):
    """Plot 2D projection of points onto the XY plane."""
    ax.scatter(points[:, 0], points[:, 1], color=color, label=title, alpha=0.6, marker=marker)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

# Create 2D projections of source, target, and aligned points
fig, ax = plt.subplots()

plot_2d_projection(source_points, 'Source Points (Original)', ax, 'blue', 'o')
plot_2d_projection(target_points_transformed, 'Target Points (Transformed)', ax, 'green', 'x')
plot_2d_projection(aligned_source_points, 'Aligned Source Points', ax, 'red', '^')

plt.legend()
plt.show()
