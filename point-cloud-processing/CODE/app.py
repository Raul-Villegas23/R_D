# app.py

from flask import Flask, request, jsonify, render_template
import os
from main_flask import process_glb_and_bag

# Initialize Flask app and specify the template folder (outside CODE directory)
app = Flask(__name__, template_folder='../templates')

# Configure upload folder
UPLOAD_FOLDER = 'DATA/'
RESULTS_FOLDER = 'RESULTS/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Get feature IDs and GLB file from the form
    feature_ids = request.form.get('feature_ids').split(',')
    glb_file = request.files['glb_file']

    if glb_file and feature_ids:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], glb_file.filename)
        glb_file.save(file_path)

        # Call the processing function
        collections_url = "https://api.3dbag.nl/collections"
        collection_id = 'pand'

        result = process_glb_and_bag(feature_ids, file_path, collections_url, collection_id)

        # Return result as JSON
        return jsonify(result)

    return 'Error: Missing feature IDs or GLB file.'

if __name__ == '__main__':
    app.run(debug=True)
