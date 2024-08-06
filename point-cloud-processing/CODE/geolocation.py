from pyproj import Transformer
from geopy.geocoders import Nominatim

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