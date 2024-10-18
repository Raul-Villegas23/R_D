from geopy.geocoders import Nominatim

# Initialize the Nominatim geolocator
geolocator = Nominatim(user_agent="geo_locator")

# Coordinates in WGS84 (latitude, longitude format)
coordinates_wgs84 = [
    (47.97459675175307, 3.313493244432963),  # Point 1
    (47.974596250117685, 3.3134934699050387),  # Point 2
    (47.97459741094898, 3.313493084697589),  # Point 3
    (47.974597581602275, 3.3134929853130366),  # Point 4
    (47.97459762369825, 3.3134929300498204)   # Point 5
]

# Reverse geocoding to get addresses
for lat, lon in coordinates_wgs84:
    location = geolocator.reverse((lat, lon), exactly_one=True)
    if location:
        print(f"Coordinates: ({lat}, {lon}) => Address: {location.address}")
    else:
        print(f"Coordinates: ({lat}, {lon}) => Address not found")
