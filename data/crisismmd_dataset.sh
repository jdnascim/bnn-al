#!/bin/bash

# Define the URL
URL="https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"

# Define the filename
FILENAME="CrisisMMD_v2.0.tar.gz"

# Define the directory to extract the files
EXTRACT_DIR="CrisisMMD_v2.0"

# Download the file
wget "$URL" -O "$FILENAME"

# Check if download was successful
if [ $? -eq 0 ]; then
    # Extract the files
    tar -xvf "$FILENAME" -C "$EXTRACT_DIR"

    # Remove the downloaded tar.gz file if extraction was successful
    if [ $? -eq 0 ]; then
        rm "$FILENAME"
    else
        echo "Extraction failed. Keeping the tar.gz file."
    fi
else
    echo "Download failed."
fi
