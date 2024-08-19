import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import rotate
from pyproj import Transformer
from scipy.optimize import minimize

def extract_2d_perimeter(mesh):
    """Extract the 2D perimeter of the mesh by projecting onto the xy-plane and computing the convex hull."""
    vertices = np.asarray(mesh.vertices)[:, :2]
    hull = ConvexHull(vertices)
    perimeter_points = vertices[hull.vertices]
    return np.vstack([perimeter_points, perimeter_points[0]])

def optimize_rotation_and_translation(perimeter1, perimeter2):
    """Optimize rotation angle and translation to align two perimeters."""
    initial_guesses = [[-45.0, 0.0, 0.0], [45.0, 0.0, 0.0], [90.0, 0.0, 0.0], [-90.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    bounds = [(-180, 180), (-np.inf, np.inf), (-np.inf, np.inf)]
    best_result, lowest_error = None, float('inf')

    for initial_guess in initial_guesses:
        result = minimize(calculate_intersection_error, initial_guess, args=(perimeter1, perimeter2), method='L-BFGS-B', bounds=bounds)
        if result.success and result.fun < lowest_error:
            best_result, lowest_error = result, result.fun

    return best_result.x if best_result else None


def calculate_intersection_error(params, perimeter1, perimeter2):
    """Calculate the error between intersections of two perimeters after rotating and translating one."""
    angle, tx, ty = params
    rotated_perimeter2 = rotate(Polygon(perimeter2), angle, origin='centroid')
    translated_perimeter2 = np.array(rotated_perimeter2.exterior.coords) + [tx, ty]
    poly1, poly2 = Polygon(perimeter1), Polygon(translated_perimeter2)
    intersection = poly1.intersection(poly2)
    union = poly1.union(poly2)
    return 1 - (intersection.area / union.area) if union.area != 0 else 0

def calculate_centroid(perimeter):
    """Calculate the centroid of a given perimeter using Shapely."""
    polygon = Polygon(perimeter)
    centroid = polygon.centroid
    return np.array([centroid.x, centroid.y])

def compute_orientation(vertices):
    """Compute the orientation of the building based on the azimuth angle of the longest edge relative to the north."""
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    
    max_length = 0
    orientation_angle = 0
    longest_edge = (None, None)
    
    for i in range(len(hull_vertices)):
        for j in range(i + 1, len(hull_vertices)):
            vec = hull_vertices[j] - hull_vertices[i]
            length = np.linalg.norm(vec)
            if length > max_length:
                max_length = length
                # Calculate the azimuth angle relative to the north (y-axis)
                orientation_angle = (np.degrees(np.arctan2(vec[1], vec[0])) + 360) % 360
                longest_edge = (hull_vertices[i], hull_vertices[j])
    
    return orientation_angle, longest_edge
