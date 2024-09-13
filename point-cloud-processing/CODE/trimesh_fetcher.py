import requests
import logging
import psutil
import trimesh
import numpy as np
import concurrent.futures
from typing import Optional, Tuple, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_json(session: requests.Session, url: str) -> Optional[Dict[str, Any]]:
    """Fetch JSON data from the given URL using a session for connection pooling."""
    try:
        response = session.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Status code: {response.status_code}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except Exception as err:
        logging.error(f"An unexpected error occurred: {err}")
    return None

def process_feature_list(collections_url: str, collection_id: str, feature_ids: List[str]) -> Tuple[Optional[trimesh.Trimesh], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Process a list of feature IDs and combine their trimesh meshes."""
    meshes = []
    scale, translate, reference_system = None, None, None

    # Use requests.Session for connection pooling
    with requests.Session() as session:
        # Concurrent fetching and processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_and_process_feature, session, collections_url, collection_id, feature_id.strip()) for feature_id in feature_ids]

            for future in concurrent.futures.as_completed(futures):
                mesh, feature_scale, feature_translate, ref_system = future.result()
                if mesh:
                    meshes.append(mesh)
                if ref_system:
                    reference_system = ref_system
                if feature_scale is not None:
                    scale = feature_scale
                if feature_translate is not None:
                    translate = feature_translate

    if meshes:
        # Combine all meshes into one using trimesh
        bag_mesh = trimesh.util.concatenate(meshes)
        return bag_mesh, scale, translate, reference_system
    else:
        logging.error("No meshes to visualize.")
        return None, None, None, None

def fetch_and_process_feature(session: requests.Session, collections_url: str, collection_id: str, feature_id: str) -> Tuple[Optional[trimesh.Trimesh], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Fetch and process a single feature."""
    feature_url = f"{collections_url}/{collection_id}/items/{feature_id}"
    feature = fetch_json(session, feature_url)
    logging.info(f"Processing feature: {feature_id}")

    if feature:
        mesh, scale, translate = create_trimesh_from_feature(feature)
        reference_system = feature['metadata'].get('metadata', {}).get('referenceSystem')
        return mesh, scale, translate, reference_system
    else:
        return None, None, None, None

def print_memory_usage(step: str):
    """Print the current memory usage at a given step."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage at {step}: {mem_info.rss / 1024 ** 2:.2f} MB")

def create_trimesh_from_feature(feature: Dict[str, Any]) -> Tuple[Optional[trimesh.Trimesh], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a Trimesh object using only the highest LoD from feature data.

    Parameters:
    - feature: A dictionary containing the feature data with vertices and geometry.

    Returns:
    - mesh: The created Trimesh object.
    - scale: The scale applied to the vertices.
    - translate: The translation applied to the vertices.
    """
    if 'vertices' in feature['feature']:
        vertices = np.array(feature['feature']['vertices'])
        transform = feature['metadata'].get('transform', {})
        scale = np.array(transform.get('scale', [1, 1, 1]))
        translate = np.array(transform.get('translate', [0, 0, 0]))
        vertices = vertices * scale + translate

        city_objects = feature['feature'].get('CityObjects', {})
        max_lod_geom = get_highest_lod_geometry(city_objects)

        if max_lod_geom:
            faces = generate_faces(max_lod_geom)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh, scale, translate
        else:
            logging.error("No geometry data found for the highest LoD.")
            return None, None, None
    else:
        logging.error("No vertices found in the feature data.")
        return None, None, None

def get_highest_lod_geometry(city_objects: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the geometry with the highest LoD."""
    max_lod = None
    max_lod_geom = None
    for obj in city_objects.values():
        for geom in obj.get('geometry', []):
            lod = geom.get('lod', None)
            if max_lod is None or (lod is not None and float(lod) > float(max_lod)):
                max_lod = lod
                max_lod_geom = geom
    return max_lod_geom

def generate_faces(geometry: Dict[str, Any]) -> np.ndarray:
    """Generate faces from the geometry's boundary data in an optimized way."""
    faces = []
    for boundary_group in geometry.get('boundaries', []):
        for boundary in boundary_group:
            if isinstance(boundary[0], list):
                for sub_boundary in boundary:
                    if len(sub_boundary) >= 3:
                        faces.extend([[sub_boundary[0], sub_boundary[i], sub_boundary[i + 1]]
                                      for i in range(1, len(sub_boundary) - 1)])
            else:
                if len(boundary) >= 3:
                    faces.extend([[boundary[0], boundary[i], boundary[i + 1]]
                                  for i in range(1, len(boundary) - 1)])
    return np.array(faces)
