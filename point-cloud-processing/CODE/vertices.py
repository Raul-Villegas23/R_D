import trimesh
import numpy as np

# Load the .glb model
mesh = trimesh.load('DATA/pijlkruid2.glb')

# Define the origin point in the local coordinate system
origin = np.array([0, 0, 0])

# Function to find the closest vertex to the origin
def find_closest_vertex(vertices):
    distances = np.linalg.norm(vertices - origin, axis=1)
    closest_vertex_index = np.argmin(distances)
    closest_vertex = vertices[closest_vertex_index]
    closest_distance = distances[closest_vertex_index]
    
    return closest_vertex, closest_distance

# Function to create a small sphere at a vertex for visualization
def create_sphere_at_vertex(vertex, radius=0.005):
    # Create a sphere with the given radius at the vertex location
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    # Move the sphere to the vertex position
    sphere.apply_translation(vertex)
    return sphere

# Visualize the scene and vertices
def visualize_mesh_with_vertices(mesh, closest_vertex):
    # Transform the mesh to the right-handed coordinate system
    mesh.apply_transform([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Create a scene to hold the mesh and the vertex marker
    scene = trimesh.Scene()

    # Add the mesh to the scene
    if isinstance(mesh, trimesh.Scene):
        scene = mesh
    else:
        scene.add_geometry(mesh)

    # Create a sphere at the closest vertex for visualization
    closest_vertex_sphere = create_sphere_at_vertex(closest_vertex, radius=0.025)
    scene.add_geometry(closest_vertex_sphere)

    # Create an axis at (0, 0, 0) for reference
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=2.0)
    scene.add_geometry(axis)

    # Show the final scene with the mesh, axis, and vertex marker
    scene.show()

# Check if the model contains multiple meshes (scene) or a single mesh
if isinstance(mesh, trimesh.Scene):
    # For scenes with multiple meshes
    for name, geometry in mesh.geometry.items():
        print(f"Checking mesh: {name}")
        vertices = geometry.vertices

        # Find the closest vertex to the origin
        closest_vertex, closest_distance = find_closest_vertex(vertices)
        print(f"Closest vertex to (0, 0, 0) in mesh {name}: {closest_vertex}, Distance: {closest_distance}")

        # Visualize the mesh along with the closest vertex
        visualize_mesh_with_vertices(geometry, closest_vertex)
else:
    # Single mesh case
    vertices = mesh.vertices

    # Find the closest vertex to the origin
    closest_vertex, closest_distance = find_closest_vertex(vertices)
    print(f"Closest vertex to (0, 0, 0): {closest_vertex}, Distance: {closest_distance}")

    # Visualize the mesh along with the closest vertex
    visualize_mesh_with_vertices(mesh, closest_vertex)
