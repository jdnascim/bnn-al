import json
import glob
import os
import tqdm
import pandas as pd

# Paths for the TSV data source and JSONL target files
DATAPATH = "data/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_{}.tsv"
JSONL_PATH = "data/CrisisMMD_v2.0_baseline_split/event_data_splits/*/{}.jsonl"

# Possible dataset splits
SPLITS = ["train", "dev", "test"]

# Step 1: Build the ID-to-text mapping from the TSV files
id_to_text = {}

for split in SPLITS:
    tsv_path = DATAPATH.format(split)
    if not os.path.exists(tsv_path):
        print(f"⚠️ Warning: TSV file not found for {split}: {tsv_path}")
        continue

    df = pd.read_csv(tsv_path, sep="\t")  # Read TSV file
    for _, row in df.iterrows():
        id_key = row.get("image")  # Extract the 'tweet_id' field
        text_value = row.get("tweet_text")  # Extract the 'tweet_text' field
        if pd.notna(id_key) and pd.notna(text_value):  # Ensure both exist
            id_to_text[id_key] = text_value

print(f"✅ Loaded {len(id_to_text)} tweet-text mappings from TSV files.")

# Step 2: Process each JSONL file in the target directory
for split in SPLITS:
    jsonl_files = glob.glob(JSONL_PATH.format(split))
    
    for filepath in tqdm.tqdm(jsonl_files, desc=f"Processing {split} JSONL files"):
        temp_filepath = filepath + ".tmp"

        with open(filepath, 'r') as infile, open(temp_filepath, 'w') as outfile:
            for line in infile:
                try:
                    data = json.loads(line.strip())  # Parse JSON
                    image_path = data.get("image")  # Extract image field
                    if image_path:

                        # Add or update the 'text' field if the ID exists in the mapping
                        if image_path in id_to_text:
                            data["text"] = id_to_text[image_path]
                        else:
                            print(image_path)

                    # Write the updated JSON object back to the file
                    outfile.write(json.dumps(data) + "\n")
                except Exception as ex:
                    print(f"⚠️ Error processing line in {filepath}: {line}\nError: {ex}")
                    continue

        # Replace the original file with the updated file
        os.replace(temp_filepath, filepath)

print("✅ Processing completed")
