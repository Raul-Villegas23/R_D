from pyproj import Transformer
import numpy as np
from typing import Tuple


def extract_latlon_orientation_from_mesh(
    final_transformation_matrix: np.ndarray,
    reference_system: str,
) -> Tuple[float, float]:
    """
    Extract longitude and latitude from the transformed mesh origin (0,0,0).

    Parameters:
        mesh (trimesh.Trimesh): The mesh object.
        final_transformation_matrix (np.ndarray): The accumulated transformation matrix applied to the mesh.
        reference_system (str): The EPSG reference system for the mesh.

    Returns:
        Tuple[float, float]: The longitude and latitude of the transformed origin.
    """
    # Step 1: Define the origin point (0, 0, 0) in the mesh's local coordinate system
    mesh_origin = np.array(
        [0.0, 0.0, 0.0, 1.0]
    )  # Homogeneous coordinates for transformation

    # Step 2: Apply the final transformation matrix to the mesh origin to get the transformed coordinates
    transformed_origin_homogeneous = final_transformation_matrix @ mesh_origin
    transformed_origin = transformed_origin_homogeneous[
        :3
    ]  # Extract (x, y, z) from homogeneous coordinates

    # Step 3: Convert the transformed origin's (x, y) to latitude/longitude
    # Extract the EPSG code from the reference system
    epsg_code = reference_system.split("/")[-1]

    # Use pyproj to convert the coordinates to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

    # Convert the transformed origin's x, y coordinates to lon/lat (WGS84)
    lon, lat = transformer.transform(transformed_origin[0], transformed_origin[1])

    # Step 4: Return the longitude and latitude
    return lon, lat
