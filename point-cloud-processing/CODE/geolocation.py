from pyproj import Transformer
from geopy.geocoders import Nominatim
import numpy as np
from scipy.spatial import ConvexHull

def transform_coordinates(lat, lon, reference_system):
    """Transform coordinates to EPSG:7415 if they are not already in that reference system."""
    epsg_code = reference_system.split('/')[-1]
    if epsg_code != '7415':
        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:7415", always_xy=True)
        lon, lat = transformer.transform(lon, lat)
    return lat, lon

def get_geo_location(lat, lon, reference_system):
    """Given latitude and longitude, return the geo location using Nominatim."""
    # Ensure coordinates are in EPSG:7415
    lat, lon = transform_coordinates(lat, lon, reference_system)
    
    # Use Nominatim geolocator
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.reverse((lat, lon), exactly_one=True)
    
    if location:
        address = location.address
        print(f"Address: {address}")
        return address
    else:
        print("Unable to retrieve location information.")
        return None
    
def extract_latlon_orientation_from_mesh(mesh, reference_system):
    """Extract longitude, latitude, and orientation from mesh vertices."""
    vertices = np.asarray(mesh.vertices)
    epsg_code = reference_system.split('/')[-1]
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    latlon_vertices = np.array([transformer.transform(x, y) for x, y, z in vertices])
    centroid = np.mean(latlon_vertices, axis=0)
    hull = ConvexHull(latlon_vertices)
    hull_vertices = latlon_vertices[hull.vertices]
    longest_edge = max(((hull_vertices[i], hull_vertices[j]) for i in range(len(hull_vertices)) for j in range(i+1, len(hull_vertices))), key=lambda edge: np.linalg.norm(edge[1] - edge[0]))
    orientation_angle = (np.degrees(np.arctan2(longest_edge[1][1] - longest_edge[0][1], longest_edge[1][0] - longest_edge[0][0])) + 360) % 360

    return centroid[1], centroid[0], orientation_angle