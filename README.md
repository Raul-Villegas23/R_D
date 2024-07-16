# 3D Mesh Processing and Visualization

This project processes 3D building features, creates meshes, aligns them with a GLB model, and visualizes the results.

## Features

- Fetches building feature data from a specified URL.
- Creates 3D meshes from feature data.
- Loads and transforms a GLB model.
- Aligns and optimizes the orientation of the GLB model with the building meshes.
- Extracts latitude, longitude, and orientation of the meshes.
- Visualizes 3D meshes and 2D perimeters.
- Retrieves geographic location using latitude and longitude.

## Requirements

- Python 3.9.19
- `requests`
- `numpy`
- `open3d`
- `matplotlib`
- `scipy`
- `shapely`
- `pyproj`
- `geopy`

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/Raul-Villegas23/3d-mesh-processing.git
   cd 3d-mesh-processing
   ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Place your GLB model file (model.glb) in the DATA directory.

2. Run the script:
    ```sh
    python mesh_alignment.py
    ```

