import requests
import logging
import psutil

import open3d as o3d
from mesh_processor import create_mesh_from_feature
from typing import Any, Dict, Optional

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

def process_feature_list(collections_url, collection_id, feature_ids):
    """Process a list of feature IDs and combine their meshes."""
    meshes, scale, translate, reference_system = [], None, None, None
    # Use requests.Session for connection pooling
    with requests.Session() as session:
        for feature_id in feature_ids:
            feature_url = f"{collections_url}/{collection_id}/items/{feature_id}"
            feature = fetch_json(session, feature_url)
            logging.info(f"Processing feature: {feature_id}")
            if feature:
                mesh, feature_scale, feature_translate = create_mesh_from_feature(feature)
                if mesh:
                    meshes.append(mesh)
                if feature_scale is not None:
                    scale = feature_scale
                if feature_translate is not None:
                    translate = feature_translate
                reference_system = feature['metadata'].get('metadata', {}).get('referenceSystem')
        if meshes:
            bag_mesh = sum(meshes, o3d.geometry.TriangleMesh())
            return bag_mesh, scale, translate, reference_system
        else:
            logging.error("No meshes to visualize.")
            return None, None, None, None
    
def print_memory_usage(step):
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage at {step}: {mem_info.rss / 1024 ** 2:.2f} MB")