from src.utils.constants import EVENTS
import json
import glob
import os
import tqdm

DATAPATH = "data/CrisisMMD_v2.0/json/{}_final_data.json"

# Step 1: Build the id-to-text mapping from the source files
id_to_text = dict()
for e in tqdm.tqdm(EVENTS):
    with open(DATAPATH.format(e), 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())  # Parse each line as JSON
                id_key = data.get("id")  # Extract the 'id'
                text_value = data.get("text")  # Extract the 'text'
                if id_key and text_value:  # Ensure both 'id' and 'text' exist
                    id_to_text[id_key] = text_value
            except json.JSONDecodeError as e:
                lines = line.split(",")
                for l in lines:
                    if l[:5] == "\"id\":":
                        id_key = int(l[5:])
                    if l[:7] == "\"text\":":
                        text_value = l[8:-1]
                    if id_key and text_value:  # Ensure both 'id' and 'text' exist
                        id_to_text[id_key] = text_value
                        continue
            except Exception as ex:
                print(f"Error processing data in {DATAPATH.format(e)}: {line}\nError: {ex}")
                continue

# Step 2: Process each JSONL file in the target directory
for filepath in tqdm.tqdm(glob.glob("data/CrisisMMD_v2.0_baseline_split/event_data_splits_w_text/*/*.jsonl")):
    # Temporary output file to write updated JSONL content
    temp_filepath = filepath.replace("w_text", "") + ".tmp"

    with open(filepath, 'r') as infile, open(temp_filepath, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())  # Parse the JSON line
                image_path = data.get("image")  # Extract the image field
                if image_path:
                    id_value = int(image_path[image_path.rfind("/") + 1:-6])  # Extract the ID from the image path

                    # Add or update the 'text' field if the ID exists in the mapping
                    if id_value in id_to_text:
                        data["text"] = id_to_text[id_value]

                # Write the updated JSON object back to the file
                outfile.write(json.dumps(data) + "\n")
            except Exception as ex:
                print(f"Error processing line in {filepath}: {line}\nError: {ex}")
                continue

    # Replace the original file with the updated file
    os.replace(temp_filepath, filepath)

print("Processing completed!")
