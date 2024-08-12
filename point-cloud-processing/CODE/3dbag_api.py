import urllib.request
import json
import os
from pathlib import Path
from cjio import cityjson

# Hardcoded pand ID and output folder
PAND_ID = "NL.IMBAG.Pand.1655100000500568"  # Replace with the desired pand ID
OUTPUT_FOLDER = "output_folder"  # Replace with your desired output folder

def fetch_and_save_pand_data(pand_id, output_folder):
    # Construct the URL using the given pand ID
    myurl = f"https://api.3dbag.nl/collections/pand/items/{pand_id}"
    
    try:
        # Open the URL and read the response
        with urllib.request.urlopen(myurl) as response:
            j = json.loads(response.read().decode('utf-8'))
            
            # Ensure the output folder exists
            os.makedirs(output_folder, exist_ok=True)
            
            # Define the output file path
            output_file = os.path.join(output_folder, f"{pand_id}.city.jsonl")
            
            # Open the output file and write the data
            with open(output_file, "w") as my_file:
                # Write metadata if available
                if "metadata" in j:
                    my_file.write(json.dumps(j["metadata"], ensure_ascii=False) + "\n")
                
                # Write the feature if available
                if "feature" in j:
                    my_file.write(json.dumps(j["feature"], ensure_ascii=False) + "\n")
                
                # Write features if available
                if "features" in j:
                    for f in j["features"]:
                        my_file.write(json.dumps(f, ensure_ascii=False) + "\n")
            
            print(f"Data for pand ID {pand_id} has been saved to {output_file}")
    
    except urllib.error.HTTPError as e:
        print(f"HTTP error occurred: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        print(f"URL error occurred: {e.reason}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def convert_cityjsonseq_to_cityjson(input_file, output_file):
    cityjson_data = {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": {},
        "geographicalExtent": [],
        "vertices": []
    }
    
    try:
        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json_obj = json.loads(line)
                    
                    # Add components to the CityJSON data
                    if "CityObjects" in json_obj:
                        cityjson_data["CityObjects"].update(json_obj["CityObjects"])
                    
                    if "geographicalExtent" in json_obj:
                        cityjson_data["geographicalExtent"] = json_obj["geographicalExtent"]
                    
                    if "vertices" in json_obj:
                        cityjson_data["vertices"] = json_obj["vertices"]
                    
                    if "metadata" in json_obj:
                        cityjson_data["metadata"] = json_obj["metadata"]
                    
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON decoding error: {e}")
        
        if not cityjson_data["CityObjects"]:
            print("No CityObjects found in the JSON Lines file.")
        if not cityjson_data["vertices"]:
            print("No vertices found in the JSON Lines file.")
        
        with open(output_file, "w") as f:
            json.dump(cityjson_data, f, indent=4)
        
        print(f"CityJSON data has been saved to {output_file}")
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON Lines file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Fetch and save data
    fetch_and_save_pand_data(PAND_ID, OUTPUT_FOLDER)
    
    # Define input and output paths for conversion
    cityjsonseq_file = os.path.join(OUTPUT_FOLDER, f"{PAND_ID}.city.jsonl")
    cityjson_file = os.path.join(OUTPUT_FOLDER, f"{PAND_ID}.city.json")
    
    # Convert CityJSONSeq to CityJSON
    convert_cityjsonseq_to_cityjson(cityjsonseq_file, cityjson_file)
    
    # Optionally, you can load and inspect the CityJSON file
    try:
        cm = cityjson.load(path=cityjson_file)
        
        for co_id, co in cm.cityobjects.items():
            print(f"Found CityObject {co_id} of type {co.type} and instance of {type(co)}")
    except FileNotFoundError as e:
        print(f"File not found during CityJSON loading: {e}")
    except Exception as e:
        print(f"An error occurred during CityJSON loading: {e}")
