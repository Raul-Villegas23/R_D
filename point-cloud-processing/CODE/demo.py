import trimesh
import numpy as np
import matplotlib.pyplot as plt

def visualize_meshes(source, target):
    """
    Visualize source and target meshes for comparison.
    """
    # Create a scene with the source and target meshes
    scene = trimesh.Scene()
    scene.add_geometry(source, transform=np.eye(4))  # Original mesh
    scene.add_geometry(target, transform=np.eye(4))  # Target mesh
    scene.show()

def main():
    # Create a simple source mesh (a cube)
    source_mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    # Create a simple target mesh (a cube translated by [2, 2, 2])
    target_mesh = trimesh.creation.box(extents=[1, 1, 1])
    target_mesh.apply_translation([2, 2, 2])
    
    # Visualize the meshes before ICP
    print("Visualizing meshes before ICP...")
    visualize_meshes(source_mesh, target_mesh)
    
    # Sample points from the meshes
    source_points = source_mesh.sample(1000)  # Sample 1000 points from source
    target_points = target_mesh.sample(1000)  # Sample 1000 points from target
    
    # Perform ICP registration
    print("Performing ICP registration...")
    matrix, aligned_source_points, cost = trimesh.registration.icp(
        source_points, target_points, 
        threshold=1.0, 
        initial=np.eye(4)
    )
    
    print(f"ICP completed with cost: {cost:.4f}")
    print("Applying the ICP transformation to the source mesh...")
    
    # Apply the ICP transformation to the source mesh
    source_mesh.apply_transform(matrix)
    
    # Visualize the aligned meshes
    print("Visualizing meshes after ICP...")
    visualize_meshes(source_mesh, target_mesh)

if __name__ == "__main__":
    main()
