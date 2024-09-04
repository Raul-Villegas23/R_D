import requests
import logging
import psutil
import trimesh
from trimesh_processor import create_trimesh_from_feature

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_json(url):
    """Fetch JSON data from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Status code: {response.status_code}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except Exception as err:
        logging.error(f"An unexpected error occurred: {err}")
    return None

def process_feature_list(collections_url, collection_id, feature_ids):
    """Process a list of feature IDs and combine their trimesh meshes."""
    meshes, scale, translate, reference_system = [], None, None, None

    for feature_id in feature_ids:
        feature_url = f"{collections_url}/{collection_id}/items/{feature_id}"
        feature = fetch_json(feature_url)
        logging.info(f"Processing feature: {feature_id}")
        if feature:
            mesh, scale, translate = create_trimesh_from_feature(feature)  # Using the trimesh version
            if mesh:
                meshes.append(mesh)
            reference_system = feature['metadata'].get('metadata', {}).get('referenceSystem')

    if meshes:
        # Combine all meshes into one using trimesh
        bag_mesh = trimesh.util.concatenate(meshes)
        return bag_mesh, scale, translate, reference_system
    else:
        logging.error("No meshes to visualize.")
        return None, None, None, None

def print_memory_usage(step):
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage at {step}: {mem_info.rss / 1024 ** 2:.2f} MB")