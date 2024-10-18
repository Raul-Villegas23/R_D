import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import rotate
from scipy.optimize import minimize
from typing import List, Tuple, Optional
import trimesh


def extract_2d_perimeter(mesh) -> np.ndarray:
    """Extract the 2D perimeter of the mesh by projecting onto the xy-plane and computing the convex hull."""
    vertices = np.asarray(mesh.vertices)[:, :2]
    hull = ConvexHull(vertices)
    perimeter_points = vertices[hull.vertices]
    return np.vstack([perimeter_points, perimeter_points[0]])


def generate_angle_guesses(
    angle_range: Tuple[float, float], step: float
) -> List[List[float]]:
    """Generate a list of initial guesses for angles."""
    angles = np.arange(angle_range[0], angle_range[1], step)
    return [[angle, 0.0, 0.0] for angle in angles]


def optimize_rotation_and_translation(
    perimeter1: np.ndarray, perimeter2: np.ndarray
) -> Optional[np.ndarray]:
    """Optimize rotation angle and translation to align two perimeters."""
    angle_range: Tuple[float, float] = (
        -180,
        180,
    )  # Define the range of angles to search for the optimal solution
    angle_step: float = 45.0  # Define the step size for the angles

    # Generate the initial guesses for angles
    initial_guesses: List[List[float]] = generate_angle_guesses(angle_range, angle_step)
    bounds: List[Tuple[float, float]] = [
        (-180, 180),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
    ]
    best_result: Optional[minimize.OptimizeResult] = None
    lowest_error: float = float("inf")
    method: str = "L-BFGS-B"

    for initial_guess in initial_guesses:
        result = minimize(
            calculate_intersection_error,
            initial_guess,
            args=(perimeter1, perimeter2),
            method=method,
            bounds=bounds,
        )
        if result.success and result.fun < lowest_error:
            best_result, lowest_error = result, result.fun

    return best_result.x if best_result else None


def calculate_intersection_error(
    params: np.ndarray, perimeter1: np.ndarray, perimeter2: np.ndarray
) -> float:
    """Calculate the error between intersections of two perimeters after rotating and translating one."""
    angle, tx, ty = params
    rotated_perimeter2 = rotate(Polygon(perimeter2), angle, origin="centroid")
    translated_perimeter2 = np.array(rotated_perimeter2.exterior.coords) + [tx, ty]
    poly1, poly2 = Polygon(perimeter1), Polygon(translated_perimeter2)
    intersection = poly1.intersection(poly2)
    union = poly1.union(poly2)
    return 1 - (intersection.area / union.area) if union.area != 0 else 0


def apply_optimal_params_trimesh(
    mesh: trimesh.Trimesh, optimal_angle: float, optimal_tx: float, optimal_ty: float
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Apply the optimal rotation and translation to the mesh, based on the mesh center.

    Parameters:
    - mesh: The Trimesh object to be transformed.
    - optimal_angle: The optimal rotation angle around the Z-axis (in degrees).
    - optimal_tx: The optimal translation in the X direction.
    - optimal_ty: The optimal translation in the Y direction.

    Returns:
    - mesh: The transformed mesh with the applied rotation and translation.
    - transformation_matrix: The 4x4 transformation matrix applied to the mesh.
    """

    # Convert angle from degrees to radians
    angle_rad = np.radians(optimal_angle)

    # Create a 4x4 identity transformation matrix
    transformation_matrix = np.eye(4)

    # Rotation matrix around the Z-axis
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )

    # Translate to the origin, rotate, and translate back
    mesh_center = mesh.centroid

    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -mesh_center

    translation_back = np.eye(4)
    translation_back[:3, 3] = mesh_center

    # Create a 4x4 rotation matrix from the 3x3 rotation matrix
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix

    # Combine the transformations: T_back * R * T_origin
    combined_transform = translation_back @ rotation_matrix_4x4 @ translation_to_origin

    # Apply translation in the X and Y directions
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = [optimal_tx, optimal_ty, 0]

    # Combine the rotation and translation transformations
    transformation_matrix = translation_matrix @ combined_transform

    # Apply the final transformation to the mesh
    mesh.apply_transform(transformation_matrix)

    return mesh, transformation_matrix
